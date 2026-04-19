import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import List

import av
import librosa
import numpy as np
import requests
from ghoshell_common.contracts import Workspace
from ghoshell_container import INSTANCE, IoCContainer, Provider
from ghoshell_moss import Message, Text
from reachy_mini_dances_library import DanceMove
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES

from framework.abcd.agent_event import CTMLAgentEvent, ReactAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.components.sound import Sound

logger = logging.getLogger(__name__)

_BILIBILI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.bilibili.com",
}

# Max duration (seconds) for a single song. Longer clips are likely compilations.
_MAX_SINGLE_SONG_DURATION = 600

# Head reset after each dance adds ~0.5s
_HEAD_RESET_OVERHEAD = 0.5

# ------------------------------------------------------------------
# Audio analysis
# ------------------------------------------------------------------


def _analyze_audio(path: str) -> dict:
    """Analyze a local audio file and return duration_s, bpm, and beat_times."""
    result: dict = {"duration_s": 0.0, "bpm": 0, "beat_times": []}
    try:
        container = av.open(path)
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            container.close()
            return result

        if stream.duration and stream.time_base:
            result["duration_s"] = round(float(stream.duration * stream.time_base), 1)

        sample_rate = stream.rate or 22050
        samples: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray().mean(axis=0)  # mono
            samples.append(arr)
        container.close()

        if not samples:
            return result

        audio = np.concatenate(samples).astype(np.float32)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)

        result["bpm"] = int(round(float(np.atleast_1d(tempo)[0])))
        result["beat_times"] = beat_times.tolist()
    except Exception:
        logger.exception("Failed to analyze audio: %s", path)
    return result


# ------------------------------------------------------------------
# MusicSearch class
# ------------------------------------------------------------------


class MusicSearch:
    """Search music via Bilibili and play through the existing Sound component."""

    def __init__(self, sound: Sound, workspace: Workspace, eventbus: EventBus) -> None:
        self._sound = sound
        self._eventbus = eventbus
        self._cache_dir = Path(workspace.runtime().abspath()) / "music_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / "index.json"
        self._index: dict = self._load_index()
        self._session: requests.Session | None = None

        # Playlist state
        self._playlist: list[dict] = []
        self._playlist_index: int = 0
        self._current_song: dict | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Beat-sync: playback start time (monotonic) for beat alignment
        self._playback_start_time: float | None = None

        # Prefetch state
        self._pending_results: list[dict] = []  # Search results not yet downloaded
        self._prefetch_task: asyncio.Task | None = None

        # Duration limiting
        self._duration_timer: asyncio.Task | None = None
        self._per_song_duration: float = -1.0  # -1 means play full song

        # Continuous mode: when True, system prompts LLM to pick next song after each finishes
        self._continuous: bool = False

        # Guard flag: prevents _on_playback_finish callback from firing during intentional stops
        self._stopping: bool = False

        # Set by stop_music() to reject stale continuous callbacks from queued events
        self._music_stopped: bool = False

        # Register playback finish callback
        self._sound.set_on_finish(self._on_playback_finish)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to load music cache index, starting fresh")
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _lookup_cache_by_bvid(self, bvid: str) -> dict | None:
        for entry in self._index.values():
            if entry.get("bvid") == bvid and entry.get("local_path") and Path(entry["local_path"]).exists():
                return entry
        return None

    def _lookup_cache(self, query: str) -> dict | None:
        entry = self._index.get(query)
        if entry and entry.get("local_path") and Path(entry["local_path"]).exists():
            return entry
        return None

    # ------------------------------------------------------------------
    # Bilibili session + search + audio extraction (synchronous)
    # ------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(_BILIBILI_HEADERS)
            try:
                self._session.get("https://www.bilibili.com", timeout=10)
            except Exception:
                logger.warning("Failed to fetch Bilibili homepage for cookies")
        return self._session

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text)

    def _search_bilibili(self, query: str, count: int = 5) -> list[dict]:
        url = "https://api.bilibili.com/x/web-interface/search/type"
        params = {"search_type": "video", "keyword": query, "page_size": count}
        try:
            resp = self._get_session().get(url, params=params, timeout=10).json()
        except Exception:
            logger.exception("Bilibili search request failed")
            return []

        if resp.get("code") != 0 or not resp.get("data", {}).get("result"):
            return []

        results = []
        for item in resp["data"]["result"][:count]:
            results.append({
                "bvid": item.get("bvid", ""),
                "title": self._strip_html(item.get("title", "")),
                "author": item.get("author", ""),
                "duration": item.get("duration", ""),
            })
        return results

    def _get_audio_url(self, bvid: str) -> str | None:
        try:
            info_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
            info = self._get_session().get(info_url, timeout=10).json()
            cid = info["data"]["cid"]
            play_url = (
                f"https://api.bilibili.com/x/player/playurl"
                f"?bvid={bvid}&cid={cid}&fnval=16&fnver=0&fourk=1"
            )
            play = self._get_session().get(play_url, timeout=10).json()
            dash = play.get("data", {}).get("dash")
            if dash and dash.get("audio"):
                return dash["audio"][0]["baseUrl"]
        except Exception:
            logger.exception("Failed to get audio URL for bvid=%s", bvid)
        return None

    def _download_audio(self, audio_url: str, dest: Path) -> bool:
        try:
            resp = self._get_session().get(audio_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.writelines(resp.iter_content(chunk_size=8192))
            return True
        except Exception:
            logger.exception("Failed to download audio to %s", dest)
            return False

    # ------------------------------------------------------------------
    # Song download + analyze
    # ------------------------------------------------------------------

    async def _prepare_song(self, result: dict) -> dict | None:
        """Download and analyze a single search result. Returns song dict or None."""
        bvid = result["bvid"]
        title = self._strip_html(result.get("title", "未知"))
        artist = result.get("author", "未知")

        cached = self._lookup_cache_by_bvid(bvid)
        if cached and cached.get("bpm") and cached.get("beat_times") is not None:
            return {
                "title": cached["title"], "artist": cached["artist"],
                "bpm": cached["bpm"], "duration_s": cached.get("duration_s", 0),
                "beat_times": cached.get("beat_times", []),
                "local_path": cached["local_path"], "bvid": bvid,
            }

        audio_url = await asyncio.to_thread(self._get_audio_url, bvid)
        if not audio_url:
            logger.warning("No audio URL for %s (%s)", title, bvid)
            return None

        safe_name = re.sub(r'[^\w\-]', '_', title)[:50]
        cache_file = self._cache_dir / f"{safe_name}_{int(time.time())}.m4a"
        ok = await asyncio.to_thread(self._download_audio, audio_url, cache_file)
        if not ok:
            return None

        info = await asyncio.to_thread(_analyze_audio, str(cache_file))
        duration_s = info.get("duration_s", 0)

        if duration_s > _MAX_SINGLE_SONG_DURATION:
            logger.info("Skipping %s — duration %.0fs exceeds max", title, duration_s)
            cache_file.unlink(missing_ok=True)
            return None

        self._index[bvid] = {
            "title": title, "artist": artist, "source": "bilibili",
            "bvid": bvid, "local_path": str(cache_file),
            "cached_at": int(time.time()),
            "bpm": info.get("bpm", 0), "duration_s": duration_s,
            "beat_times": info.get("beat_times", []),
        }
        self._save_index()

        return {
            "title": title, "artist": artist,
            "bpm": info.get("bpm", 0), "duration_s": duration_s,
            "beat_times": info.get("beat_times", []),
            "local_path": str(cache_file), "bvid": bvid,
        }

    # ------------------------------------------------------------------
    # Prefetch: download next song in background while current plays
    # ------------------------------------------------------------------

    async def _prefetch_next(self) -> None:
        """Download and analyze the next pending search result in background."""
        while self._pending_results:
            result = self._pending_results.pop(0)
            try:
                song = await self._prepare_song(result)
                if song is not None:
                    self._playlist.append(song)
                    logger.info("Prefetched: %s (%.0fs)", song["title"], song["duration_s"])
                    return
            except Exception:
                logger.exception("Prefetch failed for %s", result.get("title", "?"))

    def _start_prefetch(self) -> None:
        """Start a background prefetch task if there are pending results."""
        if not self._pending_results:
            return
        if self._prefetch_task and not self._prefetch_task.done():
            return
        self._prefetch_task = asyncio.create_task(self._prefetch_next())

    # ------------------------------------------------------------------
    # Playback finish callback & playlist progression
    # ------------------------------------------------------------------

    async def _reset_playback(self) -> None:
        """Stop music and clear ALL playback/playlist state. Called at the start of every play_music."""
        # Cancel duration timer
        if self._duration_timer and not self._duration_timer.done():
            self._duration_timer.cancel()
            self._duration_timer = None
        # Cancel prefetch
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            self._prefetch_task = None
        # Stop sound with _stopping guard so callback is a no-op
        if self._current_song is not None:
            self._stopping = True
            await self._sound.stop_sound()
            self._stopping = False
        # Clear all state
        self._current_song = None
        self._playback_start_time = None
        self._playlist = []
        self._playlist_index = 0
        self._pending_results = []
        self._per_song_duration = -1.0

    def _on_playback_finish(self, source: str) -> None:
        """Called from file_player worker thread when playback finishes normally."""
        if self._loop is None or self._current_song is None or self._stopping:
            return
        asyncio.run_coroutine_threadsafe(self._handle_song_finish(), self._loop)

    def _start_duration_timer(self, song: dict) -> float:
        """Start duration timer if per_song_duration is set. Returns the effective remaining_s."""
        song_dur = song.get("duration_s", 0)
        if self._per_song_duration > 0:
            remaining_s = min(self._per_song_duration, song_dur)
            if remaining_s < song_dur:
                self._duration_timer = asyncio.create_task(
                    self._stop_after_duration(remaining_s)
                )
            return remaining_s
        return song_dur

    async def _handle_song_finish(self) -> None:
        """Handle song finish: play next in playlist or notify end."""
        if self._current_song is None:
            return  # Already handled or reset — nothing to do
        # Cancel any duration timer
        if self._duration_timer and not self._duration_timer.done():
            self._duration_timer.cancel()
            self._duration_timer = None
        # Clear remaining dance commands from previous song
        await self._eventbus.put(CTMLAgentEvent(
            ctml='<clear chan="reachy_mini"/>',
            priority=2,
        ))

        self._playlist_index += 1

        if self._playlist_index < len(self._playlist):
            # More songs ready — transition to next
            prev = self._current_song
            next_song = self._playlist[self._playlist_index]
            self._current_song = next_song

            await self._sound.play_sound(next_song["local_path"])
            self._playback_start_time = time.monotonic()
            remaining_s = self._start_duration_timer(next_song)

            # Push transition speech, then choreography for new song
            await self._push_song_transition_event(prev, next_song)
            await self._push_choreography_event(next_song, remaining_s=remaining_s)

            # Prefetch the one after that
            self._start_prefetch()

        elif self._pending_results:
            # Next song not in playlist yet — download now
            prev = self._current_song
            song = None
            while self._pending_results and song is None:
                result = self._pending_results.pop(0)
                song = await self._prepare_song(result)

            if song:
                self._playlist.append(song)
                self._current_song = song
                await self._sound.play_sound(song["local_path"])
                self._playback_start_time = time.monotonic()
                remaining_s = self._start_duration_timer(song)
                await self._push_song_transition_event(prev, song)
                await self._push_choreography_event(song, remaining_s=remaining_s)
                self._start_prefetch()
            else:
                # All remaining downloads failed
                last = self._current_song
                self._current_song = None
                self._playback_start_time = None
                self._playlist = []
                self._playlist_index = 0
                self._pending_results = []
                await self._push_song_end_event(last["title"], last["artist"])
        else:
            # No more songs in playlist
            last = self._current_song
            self._current_song = None
            self._playback_start_time = None
            self._playlist = []
            self._playlist_index = 0
            await self._push_song_end_event(last["title"], last["artist"])

    async def _push_song_end_event(self, title: str, artist: str) -> None:
        """Notify the LLM that a song has finished playing."""
        if self._continuous:
            prompt = (
                f"歌曲 {title} - {artist} 已播放完毕。"
                f"\n请简短自然地评价这首歌（一两句话即可），不要重复固定话术，"
                f"然后继续用play_music选择下一首歌并播放，不要停下来问用户。"
            )
        else:
            prompt = (
                f"歌曲 {title} - {artist} 已播放完毕。"
                f"\n请简短自然地评价这首歌（一两句话即可），不要重复固定话术。"
                f"\n可以问问用户还想听什么。"
            )
        await self._eventbus.put(ReactAgentEvent(
            messages=[Message.new(role="system").with_content(Text(text=prompt))],
            priority=0,
        ))

    async def _push_song_transition_event(self, prev: dict, next_song: dict) -> None:
        """Push a ReactAgentEvent for the LLM to speak a transition line."""
        prompt = (
            f"歌曲 {prev['title']} - {prev['artist']} 播完了。"
            f"\n现在开始播放: {next_song['title']} - {next_song['artist']}。"
            f"\n请用一句话自然过渡（简短评价上一首或期待下一首）。"
            f"\n**只说一句过渡语，不要输出dance/emotion等CTML动作标签**，编舞请求会单独发给你。"
        )
        await self._eventbus.put(ReactAgentEvent(
            messages=[Message.new(role="system").with_content(Text(text=prompt))],
            priority=1,
        ))

    # ------------------------------------------------------------------
    # LLM choreography (one event per song, accurate duration)
    # ------------------------------------------------------------------

    async def _push_choreography_event(self, song: dict, remaining_s: float | None = None) -> None:
        """Ask the LLM to freely choreograph using normal CTML commands.

        :param song: song dict with title, artist, bpm, duration_s
        :param remaining_s: if set, use this as the duration instead of full song duration
        """
        title = song.get("title", "")
        artist = song.get("artist", "")
        bpm = song.get("bpm", 0)
        duration_s = remaining_s if remaining_s is not None else song.get("duration_s", 0)
        beat_dur = round(60.0 / bpm, 2) if bpm > 0 else 0.5

        # Build dance duration table (fixed at DanceMove.default_bpm=114)
        internal_beat_dur = 60.0 / DanceMove.default_bpm
        dance_lines = []
        for name, move in AVAILABLE_MOVES.items():
            _func, _params, meta = move
            beats = meta.get("default_duration_beats", 4)
            dance_dur = round(internal_beat_dur * beats, 1)
            total_dur = round(dance_dur + _HEAD_RESET_OVERHEAD, 1)
            dance_lines.append(f"  {name}: {total_dur}s")
        dance_table = "\n".join(dance_lines)

        prompt = (
            f"歌曲 {title} - {artist} 正在播放（BPM={bpm}，剩余时长{duration_s}秒）。\n"
            f"请编排覆盖整首歌剩余时长的舞蹈动作序列。\n"
            f"\n"
            f"=== 节拍信息 ===\n"
            f"每拍时长: {beat_dur}s（BPM={bpm}）\n"
            f"系统会自动将每个动作对齐到最近的节拍点。\n"
            f"=== RGB灯光（当有rgb通道的时，优先输出rgb:bpm_flash命令）\n"
            f"- <rgb:bpm_flash bpm=... mode=... duration=...>"
            f"\n"
            f"=== 动作时长表（固定，不随歌曲BPM变化）===\n"
            f"{dance_table}\n"
            f"  emotion: ~1.5s\n"
            f"  head_move: 由duration参数决定\n"
            f"  antennas_move: 由duration参数决定\n"
            f"\n"
            f"=== 可用动作CTML命令 ===\n"
            f"- <jetarm:motion ...> 机械臂舞蹈动作\n"
            f'- <reachy_mini:dance name="..."/>  舞蹈动作\n'
            f'- <reachy_mini:emotion emoji="..."/>  表情动作\n'
            f'- <reachy_mini:head_move yaw=".." pitch=".." duration=".."/>  头部律动\n'
            f'- <reachy_mini:antennas_move left=".." right=".." duration=".."/>  天线摆动\n'
            f'- <sleep duration=".."/>  节拍停顿（用beat的倍数：{beat_dur}s=一拍，{round(beat_dur * 2, 2)}s=两拍）\n'
            f"\n"
            f"=== 编排要求 ===\n"
            f"1. 直接输出完整的动作序列，总时长覆盖{duration_s}秒（不要使用loop，逐个写出每个动作）\n"
            f"2. 动作要多样化，不要连续重复相同动作，尽量用到不同的dance\n"
            f"3. 用<sleep duration=\"{beat_dur}\"/>在动作间插入节拍停顿，踩准节奏\n"
            f"4. head_move和antennas_move穿插在dance之间增加表现力\n"
            f"5. 根据歌曲风格搭配（快歌→headbanger_combo/chicken_peck，慢歌→pendulum_swing/side_to_side_sway）\n"
            f"6. 高潮段密集动作，安静段多用sleep留白\n"
            f"7. 不要说话，只输出CTML动作标签\n"
            f"8. 必须使用机械臂动作\n"
        )
        logger.info("Pushing choreography event for %s (BPM=%d, %.1fs)", title, bpm, duration_s)
        await self._eventbus.put(ReactAgentEvent(
            messages=[Message.new(role="system").with_content(Text(text=prompt))],
            priority=0,
        ))

    async def _stop_after_duration(self, duration_s: float) -> None:
        """Stop playback after specified duration."""
        try:
            await asyncio.sleep(duration_s)
            if self._current_song is None:
                return
            # Set _stopping so the worker-thread callback becomes a no-op
            self._stopping = True
            await self._sound.stop_sound()
            self._stopping = False
            self._duration_timer = None
            await self._handle_song_finish()
        except asyncio.CancelledError:
            self._stopping = False
        except Exception:
            self._stopping = False
            logger.exception("Error in duration timer")

    # ------------------------------------------------------------------
    # Context messages (music-playing state for LLM)
    # ------------------------------------------------------------------

    async def context_messages(self) -> List[Message]:
        """Return context messages about current music playback state."""
        if self._current_song is None:
            return []
        msg = Message.new(role="user", name="__music__")
        title = self._current_song.get("title", "")
        artist = self._current_song.get("artist", "")
        bpm = self._current_song.get("bpm", 0)
        total = len(self._playlist) + len(self._pending_results)
        idx = self._playlist_index + 1
        playlist_info = f"（第{idx}/{total}首）" if total > 1 else ""
        msg.with_content(
            Text(text=(
                f"⚠️ 音乐正在播放: {title} - {artist}{playlist_info}（BPM={bpm}）。"
                f"当收到编舞请求时，请用dance/emotion/head_move/antennas_move等CTML命令自由编排舞蹈动作。"
                f"编排过程中不要说话，只输出CTML动作标签。"
                f"歌曲结束时系统会自动通知你。"
            )),
        )
        return [msg]

    # ------------------------------------------------------------------
    # Public async API (exposed as CTML commands)
    # ------------------------------------------------------------------

    async def play_music(self, query: str, count: int = 1, duration: float = -1.0, continuous: str = "False") -> str:
        """搜索并播放音乐。query 必须是具体的「歌手+歌名」或「歌名」。

        禁止将用户的模糊描述（如"下雨天适合听的歌"）直接作为 query。
        当用户给出心情/场景/风格描述时，先推荐具体歌曲，再用具体歌名调用。

        :param query: 具体歌名或歌手+歌名，如"周杰伦 晴天"、"光年之外"
        :param count: 播放歌曲数量，默认1
        :param duration: 不要传此参数，除非用户明确要求只听指定秒数。默认-1播完整首歌
        :param continuous: 连续播放模式。当用户要求播放多首歌曲时设为True，播完后系统会提示你继续选歌
        """
        continuous = continuous == "True"

        self._loop = asyncio.get_running_loop()

        # Reject stale continuous callbacks that arrive after stop_music
        if self._music_stopped:
            self._music_stopped = False
            if continuous:
                return "音乐已停止。"

        # Always reset ALL state first — clean slate for every play request
        await self._reset_playback()

        # Store session settings
        self._per_song_duration = duration
        if continuous:
            self._continuous = True
        # Don't set _continuous = False here; only stop_music clears it

        # 1. Check cache (single song)
        if count == 1:
            cached = self._lookup_cache(query)
            if cached and cached.get("local_path") and Path(cached["local_path"]).exists():
                local_path = cached["local_path"]
                if cached.get("bpm") and cached.get("beat_times") is not None:
                    info = {
                        "bpm": cached["bpm"],
                        "duration_s": cached.get("duration_s", 0),
                        "beat_times": cached.get("beat_times", []),
                    }
                else:
                    info = await asyncio.to_thread(_analyze_audio, local_path)
                    cached["bpm"] = info.get("bpm", 0)
                    cached["duration_s"] = info.get("duration_s", 0)
                    cached["beat_times"] = info.get("beat_times", [])
                    self._save_index()
                title = cached["title"]
                artist = cached["artist"]
                song = {
                    "title": title, "artist": artist,
                    "bpm": info["bpm"], "duration_s": info["duration_s"],
                    "beat_times": info["beat_times"], "local_path": local_path,
                }
                self._playlist = [song]
                self._playlist_index = 0
                self._current_song = song
                await self._sound.play_sound(local_path)
                self._playback_start_time = time.monotonic()
                remaining_s = self._start_duration_timer(song)
                await self._push_choreography_event(song, remaining_s=remaining_s)
                return (
                    f"正在播放: {title} - {artist}（缓存）。"
                    f"时长: {remaining_s}秒，BPM: {info['bpm']}。"
                )

        # 2. Search Bilibili
        search_count = max(count * 2, 5)
        results = await asyncio.to_thread(self._search_bilibili, query, search_count)
        if not results:
            return f"没有找到 '{query}' 相关的音乐"

        # 3. Download ONLY the first song → play immediately
        first_song = None
        remaining_results = list(results)
        while remaining_results and first_song is None:
            result = remaining_results.pop(0)
            first_song = await self._prepare_song(result)

        if not first_song:
            return f"找到了 '{query}' 但无法获取可用的音频"

        # 4. Set up playlist with first song, save remaining for prefetch
        self._playlist = [first_song]
        self._playlist_index = 0
        self._current_song = first_song
        # Keep up to (count - 1) remaining results for prefetch
        self._pending_results = remaining_results[:count - 1]

        # 5. Start playing immediately + push choreography event
        await self._sound.play_sound(first_song["local_path"])
        self._playback_start_time = time.monotonic()
        remaining_s = self._start_duration_timer(first_song)
        await self._push_choreography_event(first_song, remaining_s=remaining_s)

        # 6. Start prefetching next song in background
        self._start_prefetch()

        # 7. Build result string
        total = 1 + len(self._pending_results)
        if total == 1:
            return (
                f"正在播放: {first_song['title']} - {first_song['artist']}。"
                f"时长: {remaining_s}秒，BPM: {first_song['bpm']}。"
            )
        return (
            f"正在播放: {first_song['title']} - {first_song['artist']}"
            f"（时长: {remaining_s}秒，BPM: {first_song['bpm']}）。"
            f"\n还有{len(self._pending_results)}首歌在后台准备中，会自动接续播放。"
        )

    async def search_music(self, query: str, count: int = 5) -> str:
        """搜索音乐返回结果列表，不自动播放。用于让用户选择。

        :param query: 搜索关键词，如"周杰伦"
        :param count: 返回的结果数量，默认5
        """
        results = await asyncio.to_thread(self._search_bilibili, query, count)
        if not results:
            return f"没有找到 '{query}' 相关的音乐"

        lines = [f"搜索 '{query}' 的结果:"]
        for i, item in enumerate(results, 1):
            title = item.get("title", "未知")
            author = item.get("author", "未知")
            duration = item.get("duration", "")
            lines.append(f"{i}. {title} - {author} ({duration})")
        return "\n".join(lines)

    async def pause_music(self) -> str:
        """暂停音乐播放。"""
        await self._sound.pause_sound()
        return "音乐已暂停。"

    async def stop_music(self) -> str:
        """停止音乐播放。"""
        self._music_stopped = True  # Block stale continuous callbacks
        await self._reset_playback()
        self._continuous = False  # Explicitly stop continuous mode
        # Clear remaining dance commands on reachy_mini channel
        await self._eventbus.put(CTMLAgentEvent(
            ctml='<clear chan="reachy_mini"/>',
            priority=2,
        ))
        return "音乐已停止。"

    async def resume_music(self) -> str:
        """恢复音乐播放，并继续配合动作。"""
        status = await asyncio.to_thread(self._sound._player.status)
        remaining_s = 0.0
        if status.duration_s and status.position_s:
            remaining_s = round(max(0, status.duration_s - status.position_s), 1)

        await self._sound.resume_sound()
        self._playback_start_time = time.monotonic()  # reset for beat-sync

        # Push a new choreography event with accurate remaining time
        if self._current_song and remaining_s > 0:
            await self._push_choreography_event(self._current_song, remaining_s=remaining_s)

        return f"音乐已恢复播放，剩余约{remaining_s:.0f}秒。"


class MusicSearchProvider(Provider[MusicSearch]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        sound = con.force_fetch(Sound)
        ws = con.force_fetch(Workspace)
        eventbus = con.force_fetch(EventBus)
        return MusicSearch(sound, ws, eventbus)
