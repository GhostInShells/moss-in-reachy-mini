import asyncio
import logging
from pathlib import Path

from ghoshell_common.contracts import Workspace
from ghoshell_container import INSTANCE, IoCContainer, Provider
from reachy_mini import ReachyMini

from moss_in_reachy_mini.audio.file_player import ReachyMiniAudioFilePlayer
from moss_in_reachy_mini.audio.mixer import AudioMixer

logger = logging.getLogger(__name__)


class Sound:
    def __init__(self, mini: ReachyMini, workspace: Workspace) -> None:
        self._mini = mini
        self._workspace = workspace
        self._player = ReachyMiniAudioFilePlayer(mini)

    def _resolve_sound_path(self, sound_file: str) -> Path:
        p = Path(sound_file)
        if p.is_absolute():
            return p

        audio_dir = Path(self._workspace.assets().sub_storage("audio").abspath()).resolve()
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
                return (priority_index.get(fp.suffix.lower(), 999), fp.name)

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
                audio_dir = Path(self._workspace.assets().sub_storage("audio").abspath())
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

    async def sound_status(self) -> str:
        status = await asyncio.to_thread(self._player.status)
        return status.to_str()


class SoundProvider(Provider[Sound]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        ws = con.force_fetch(Workspace)
        sound = Sound(mini, ws)
        try:
            mixer = con.force_fetch(AudioMixer)
        except Exception:
            mixer = None
        if mixer is not None:
            # Route play_sound to mixer to allow mixing with TTS.
            sound._player._mixer = mixer  # type: ignore[attr-defined]
        return sound
