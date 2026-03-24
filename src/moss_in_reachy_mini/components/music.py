import asyncio
import json
import logging
import re
import time
from pathlib import Path

import av
import librosa
import numpy as np
import requests
from ghoshell_common.contracts import Workspace
from ghoshell_container import INSTANCE, IoCContainer, Provider
from ghoshell_moss import Message, Text

from framework.abcd.agent_event import ReactAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.components.sound import Sound

logger = logging.getLogger(__name__)

_BILIBILI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.bilibili.com",
}


def _analyze_audio(path: str) -> dict:
    """Analyze a local audio file and return duration_s, bpm, and beat_times."""
    result: dict = {"duration_s": 0.0, "bpm": 0, "beat_times": []}
    try:
        # Decode with PyAV (handles m4a, mp3, etc.)
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

        # Use librosa for beat tracking (operates on numpy array directly)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)

        result["bpm"] = int(round(float(np.atleast_1d(tempo)[0])))
        result["beat_times"] = beat_times.tolist()
    except Exception:
        logger.exception("Failed to analyze audio: %s", path)
    return result


def _format_play_result(title: str, artist: str, info: dict, from_cache: bool = False) -> str:
    """Format play_music return string — status only, no dance suggestions."""
    bpm = info.get("bpm", 0)
    duration_s = info.get("duration_s", 0)
    cache_tag = "（缓存）" if from_cache else ""
    return f"正在播放: {title} - {artist}{cache_tag}。时长: {duration_s}秒，BPM: {bpm}。舞蹈编排已自动触发，不要在本轮回复中输出任何dance命令。"


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

    def _lookup_cache(self, query: str) -> dict | None:
        entry = self._index.get(query)
        if entry and entry.get("local_path") and Path(entry["local_path"]).exists():
            return entry
        return None

    # ------------------------------------------------------------------
    # Bilibili session + search + audio extraction (synchronous – run via to_thread)
    # ------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        """Get or create a requests.Session with Bilibili cookies."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(_BILIBILI_HEADERS)
            # Visit homepage to obtain required cookies (buvid3, etc.)
            try:
                self._session.get("https://www.bilibili.com", timeout=10)
            except Exception:
                logger.warning("Failed to fetch Bilibili homepage for cookies")
        return self._session

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text)

    def _search_bilibili(self, query: str, count: int = 5) -> list[dict]:
        """Search Bilibili for music videos, return list of result dicts."""
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
        """Get the best audio stream URL for a Bilibili video."""
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
        """Download audio stream to a local file. Bilibili requires Referer."""
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
    # Public async API (exposed as CTML commands)
    # ------------------------------------------------------------------

    async def _push_choreography_event(self, title: str, artist: str, info: dict) -> None:
        """Push a ReactAgentEvent to trigger the LLM to generate dance choreography."""
        bpm = info.get("bpm", 0)
        duration_s = info.get("duration_s", 0)
        beat_times: list[float] = info.get("beat_times", [])
        if bpm <= 0 or duration_s <= 0:
            return

        beat_duration = 60.0 / bpm
        dance_4beat_s = round(4 * beat_duration, 1)
        total_beats = len(beat_times)
        # How many 4-beat dances fit in the song
        dance_count = max(1, total_beats // 4)

        if bpm > 120:
            style = "活泼欢快"
            dances = "headbanger_combo, chicken_peck, grid_snap, sharp_side_tilt, yeah_nod"
        elif bpm > 80:
            style = "律动感"
            dances = "groovy_sway_and_roll, side_to_side_sway, polyrhythm_combo, uh_huh_tilt, simple_nod"
        else:
            style = "优雅舒缓"
            dances = "pendulum_swing, head_tilt_roll, interwoven_spirals, side_glance_flick, dizzy_spin"

        prompt = (
            f"音乐正在播放: {title} - {artist}，BPM={bpm}，时长={duration_s}秒，"
            f"总共{total_beats}拍，每拍{beat_duration:.2f}秒，每个dance占4拍（{dance_4beat_s}秒），"
            f"整首歌可容纳约{dance_count}个dance。风格: {style}。推荐dance: {dances}。"
            f"\n请立即输出动作编排配合音乐。**必须大量使用loop来节省输出**，结构如下："
            f"\n1. 开场（1-2个动作）"
            f"\n2. 主体用loop覆盖大部分时长，每个loop体内放5-8个不同动作"
            f"\n3. 中间可换多组loop体现变化"
            f"\n4. 结尾（1-2个动作）"
            f"\n总输出控制在15个标签以内。"
            f"\n示例（{duration_s}秒歌曲）："
            f"\n<reachy_mini:emotion name=\"cheerful1\"/>"
            f"<loop times=\"{max(1, dance_count // 3)}\">"
            f"...动作ctml"
            f"</loop>"
            f"<loop times=\"{max(1, dance_count // 3)}\">"
            f"...动作ctml"
            f"</loop>"
            f"<reachy_mini:dance name=\"head_tilt_roll\"/>"
        )
        await self._eventbus.put(ReactAgentEvent(
            messages=[Message.new(role="system").with_content(Text(text=prompt))],
            priority=1,
        ))

    async def play_music(self, query: str) -> str:
        """搜索并播放音乐。query 为歌名、歌手名或关键词组合。

        :param query: 搜索关键词，如"周杰伦 晴天"
        """
        # 1. Check cache
        cached = self._lookup_cache(query)
        if cached and cached.get("local_path") and Path(cached["local_path"]).exists():
            local_path = cached["local_path"]
            info = await asyncio.to_thread(_analyze_audio, local_path)
            await self._sound.play_sound(local_path)
            await self._push_choreography_event(cached["title"], cached["artist"], info)
            return _format_play_result(cached["title"], cached["artist"], info, from_cache=True)

        # 2. Search Bilibili
        results = await asyncio.to_thread(self._search_bilibili, query)
        if not results:
            return f"没有找到 '{query}' 相关的音乐"

        # 3. Get audio URL for first result
        best = results[0]
        audio_url = await asyncio.to_thread(self._get_audio_url, best["bvid"])
        if not audio_url:
            return f"找到了 '{query}' 但无法获取播放链接"

        # 4. Download to cache (Bilibili requires Referer, can't stream directly via PyAV)
        safe_name = re.sub(r'[^\w\-]', '_', query)[:50]
        cache_file = self._cache_dir / f"{safe_name}_{int(time.time())}.m4a"
        ok = await asyncio.to_thread(self._download_audio, audio_url, cache_file)
        if not ok:
            return f"下载 '{query}' 音频失败"

        # 5. Analyze BPM and beat times
        info = await asyncio.to_thread(_analyze_audio, str(cache_file))

        # 6. Play local file
        await self._sound.play_sound(str(cache_file))

        # 7. Update cache index
        title = best.get("title", query)
        artist = best.get("author", "未知")
        self._index[query] = {
            "title": title,
            "artist": artist,
            "source": "bilibili",
            "bvid": best["bvid"],
            "local_path": str(cache_file),
            "cached_at": int(time.time()),
        }
        self._save_index()

        await self._push_choreography_event(title, artist, info)
        return _format_play_result(title, artist, info)

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
        await self._sound.stop_sound()
        return "音乐已停止。"

    async def resume_music(self) -> str:
        """恢复音乐播放，并继续配合动作。"""
        # Get remaining time before resuming
        status = await asyncio.to_thread(self._sound._player.status)
        remaining_s = 0.0
        if status.duration_s and status.position_s:
            remaining_s = max(0, status.duration_s - status.position_s)

        await self._sound.resume_sound()

        # Push new choreography event with remaining time
        if remaining_s > 0:
            await self._push_remaining_choreography_event(remaining_s)

        return f"音乐已恢复播放，剩余约{remaining_s:.0f}秒。"

    async def _push_remaining_choreography_event(self, remaining_s: float) -> None:
        """Push a ReactAgentEvent for the remaining playback time."""
        prompt = (
            f"音乐已恢复播放，剩余约{remaining_s:.0f}秒。"
            f"请立即输出一段动作编排来配合剩余的音乐，总时长尽量接近{remaining_s:.0f}秒。"
            f"编排应该丰富有层次，自由组合dance、head_move、antennas_move、emotion。"
            f"<loop>只用在需要重复的片段，不要把整个编排包在一个大loop里。"
            f"\n示例：\n<reachy_mini:dance name=\"groovy_sway_and_roll\"/>"
            f"<reachy_mini:emotion name=\"cheerful1\"/>"
            f"<loop times=\"2\"><reachy_mini:dance name=\"simple_nod\"/>"
            f"<reachy_mini:antennas_move left=\"0.3\" right=\"-0.3\"/></loop>"
            f"<reachy_mini:dance name=\"side_to_side_sway\"/>"
        )
        await self._eventbus.put(ReactAgentEvent(
            messages=[Message.new(role="system").with_content(Text(text=prompt))],
            priority=1,
        ))


class MusicSearchProvider(Provider[MusicSearch]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        sound = con.force_fetch(Sound)
        ws = con.force_fetch(Workspace)
        eventbus = con.force_fetch(EventBus)
        return MusicSearch(sound, ws, eventbus)
