import asyncio
import logging
from pathlib import Path
from typing import List

from ghoshell_common.contracts import Workspace, Storage, FileStorage
from ghoshell_container import INSTANCE, IoCContainer, Provider
from ghoshell_moss import Channel, PyChannel, Message, Text
from reachy_mini import ReachyMini

from moss_in_reachy_mini.audio.file_player import ReachyMiniAudioFilePlayer
from moss_in_reachy_mini.audio.mixer import AudioMixer

logger = logging.getLogger(__name__)


class Sound:
    def __init__(self, mini: ReachyMini, player: ReachyMiniAudioFilePlayer, mixer: AudioMixer, storage: FileStorage) -> None:
        self._mini = mini
        self._storage = storage
        self._player = player
        self._mixer = mixer

    def _resolve_sound_path(self, sound_file: str) -> Path:
        p = Path(sound_file)
        if p.is_absolute():
            return p

        self._storage.dir("", recursive=False)

        audio_dir = Path(self._storage.abspath()).resolve()
        candidate = (audio_dir / p).resolve()

        # If user specifies a filename with extension, try the exact file first.
        # Only fall back to stem-search when that exact file doesn't exist.
        if p.suffix and candidate.exists():
            return candidate

        # Otherwise: search based on facts from the assets/audio directory.
        search_dir = (audio_dir / p.parent).resolve()
        if not str(search_dir).startswith(str(audio_dir)):
            raise ValueError("Invalid sound_file path")
        if not search_dir.exists():
            return candidate

        # Always treat extension as a hint for asset lookup.
        # e.g. user may pass "song.mp3" but only "song.wav" exists.
        target_name = p.stem if p.suffix else p.name
        files = [f for f in search_dir.iterdir() if f.is_file()]

        exact = [f for f in files if f.stem == target_name]
        if exact:
            # Deterministic pick among *found* candidates (not guessing).
            priority = [
                ".wav",
                ".flac",
                ".ogg",
                ".opus",
                ".m4a",
                ".aac",
                ".mp3",
            ]
            priority_index = {ext: i for i, ext in enumerate(priority)}

            def key_fn(fp: Path) -> tuple[int, str]:
                return priority_index.get(fp.suffix.lower(), 999), fp.name

            return min(exact, key=key_fn)

        contains = [f for f in files if target_name in f.stem]
        if len(contains) == 1:
            return contains[0]
        if len(contains) > 1:
            choices = ", ".join(sorted(f.name for f in contains)[:8])
            raise FileNotFoundError(
                f"Multiple matches for '{target_name}' under assets/audio: {choices}. "
                "Please specify the exact filename."
            )

        return candidate

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    def play_sound_doc(self):
        return ("Play a sound (local file or URL). "
                "If it's a relative path, it is resolved under assets/audio/. "
                "Supports pause/resume/stop via corresponding commands."
                f"@param sound_file: sound filename; The available files are as follows: {list(self._storage.dir("", recursive=False))}")

    async def play_sound(self, sound_file: str) -> None:
        """Play audio (file or URL) by decoding to PCM and streaming to Reachy Mini."""

        # `Sound.play_sound` is the MOSS command entry; keep it async but run
        # playback control in a worker thread.
        source: str
        if self._looks_like_url(sound_file):
            source = sound_file
        else:
            path = self._resolve_sound_path(sound_file)
            logger.info("Playing audio file: %s", path)
            if not path.exists():
                audio_dir = Path(self._storage.abspath())
                raise FileNotFoundError(
                    f"Sound file '{sound_file}' not found under {audio_dir}. Please check the filename in assets/audio."
                )
            source = str(path)

        await asyncio.to_thread(self._player.play, source)

    async def pause_sound(self) -> None:
        await asyncio.to_thread(self._player.pause)

    async def resume_sound(self) -> None:
        await asyncio.to_thread(self._player.resume)

    async def stop_sound(self) -> None:
        await asyncio.to_thread(self._player.stop)

    async def set_volume(self, level: float) -> str:
        """设置绝对音量。用户说"音量调到50"或"音量设为80"时使用。

        :param level: 音量百分比，范围0~100。例如50表示50%音量。
        """
        if self._mixer is None:
            return "音量控制不可用"
        self._mixer.set_volume(level / 100.0)
        pct = int(self._mixer.volume() * 100)
        return f"音量已调整为{pct}%"

    async def volume_up(self) -> str:
        """调大音量。用户说"大声点"、"声音调大"时使用。"""
        if self._mixer is None:
            return "音量控制不可用"
        current = self._mixer.volume()
        step = 0.2 if current >= 0.1 else 0.05
        self._mixer.set_volume(current + step)
        pct = int(self._mixer.volume() * 100)
        return f"音量已调大到{pct}%"

    async def volume_down(self) -> str:
        """调小音量。用户说"小声点"、"声音调小"时使用。"""
        if self._mixer is None:
            return "音量控制不可用"
        current = self._mixer.volume()
        step = 0.2 if current > 0.1 else 0.05
        self._mixer.set_volume(current - step)
        pct = int(self._mixer.volume() * 100)
        return f"音量已调小到{pct}%"

    async def context_messages(self) -> List[Message]:
        msg = Message.new(role="user", name="__sound__")
        pct = int(self._mixer.volume() * 100)
        status = await asyncio.to_thread(self._player.status)
        msg.with_content(
            Text(text=f"当前音量为{pct},"),
            Text(text=f"当前播放器状态为{status}"),
        )

        return [msg]

    def as_channel(self) -> PyChannel:
        chan = PyChannel(name="sound", description="")
        chan.build.command(
            name="play_sound",
            doc=self.play_sound_doc,
        )(self.play_sound)

        chan.build.command(
            name="pause_sound",
            doc="暂停当前音频/音乐播放。用户说暂停音乐时，**优先输出**此命令。",
        )(self.pause_sound)

        chan.build.command(
            name="resume_sound",
            doc="恢复音频/音乐播放。用户说继续播放时使用此命令。",
        )(self.resume_sound)

        chan.build.command(
            name="stop_sound",
            doc="停止音频/音乐播放。用户说停止音乐、关掉音乐时，**优先输出**此命令。",
        )(self.stop_sound)

        chan.build.command(
            name="set_volume",
            doc=(
                "设置绝对音量。用户说'音量调到50'、'音量设为80'时使用。"
                "level为百分比(0~100)。"
                "注意：必须先输出此命令，再说话，否则说话时音量还是旧的。"
            ),
        )(self.set_volume)

        chan.build.command(
            name="volume_up",
            doc=(
                "调大音量。用户说'大声点'、'声音调大'时使用。"
                "注意：必须先输出此命令，再说话，否则说话时音量还是旧的。"
            ),
        )(self.volume_up)

        chan.build.command(
            name="volume_down",
            doc=(
                "调小音量。用户说'小声点'、'声音调小'、'声音太大了'时使用。"
                "注意：必须先输出此命令，再说话，否则说话时音量还是旧的。"
            ),
        )(self.volume_down)

        chan.build.context_messages(self.context_messages)

        return chan


class SoundProvider(Provider[Sound]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        ws = con.force_fetch(Workspace)
        sound_storage: FileStorage|Storage = ws.assets().sub_storage("audio")
        player = ReachyMiniAudioFilePlayer(mini)
        mixer = con.force_fetch(AudioMixer)
        # Route play_sound to mixer to allow mixing with TTS.
        player._mixer = mixer  # type: ignore[attr-defined]
        sound = Sound(mini, player, mixer, sound_storage)
        return sound
