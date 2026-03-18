import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, Provider
from reachy_mini import ReachyMini

from moss_in_reachy_mini.audio.pcm_utils import ensure_2d_f32, ensure_channels, resample_f32, safe_int

logger = logging.getLogger(__name__)


def _ensure_2d_f32(data: np.ndarray) -> np.ndarray:
    # Keep existing behavior: auto-transpose (channels, frames) when unambiguous.
    return ensure_2d_f32(data, allow_transpose=True)


def _ensure_channels(data: np.ndarray, channels: int) -> np.ndarray:
    # Keep existing behavior: channels<=0 -> clamp to 1.
    return ensure_channels(
        data,
        channels,
        default_channels=1,
        allow_transpose=True,
    )


def _resample_f32(data: np.ndarray, *, origin_rate: int, target_rate: int) -> np.ndarray:
    # Keep existing behavior: accept channels-first input.
    return resample_f32(data, origin_rate=origin_rate, target_rate=target_rate, allow_transpose=True)


@dataclass
class _Source:
    name: str
    buffers: deque[np.ndarray]
    enabled: bool = True
    closed: bool = False


class AudioMixer:
    """Single-output audio mixer.

    All producers (TTS, play_sound, etc.) should push PCM into this mixer.
    The mixer is the only component that pushes to `mini.media.push_audio_sample()`.

    This avoids:
    - interleaving/chunk-fighting when multiple producers push concurrently
    - "Output stream is not open" warnings caused by competing start/stop calls
    """

    def __init__(
        self,
        mini: ReachyMini,
        *,
        logger: LoggerItf | None = None,
        chunk_ms: int = 20,
    ) -> None:
        self._mini = mini
        self._logger = logger or logging.getLogger("AudioMixer")
        self._chunk_ms = int(chunk_ms)

        # Keep the underlying SoundDevice buffer short; otherwise "pause" will
        # appear ineffective because a lot of audio is already queued.
        try:
            self._max_output_buffers = max(1, int(os.getenv("REACHY_MINI_MIXER_MAX_OUTPUT_BUFFERS", "3")))
        except Exception:
            self._max_output_buffers = 3

        self._lock = threading.Lock()
        self._sources: dict[str, _Source] = {}

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._force_output_sr = safe_int(os.getenv("REACHY_MINI_AUDIO_OUTPUT_SR"))
        self._sr: int | None = None
        self._ch: int | None = None

    def sample_rate(self) -> int:
        return int(self._sr or 0)

    def channels(self) -> int:
        return int(self._ch or 0)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ensure_output_open()
        self._thread = threading.Thread(target=self._loop, name="AudioMixer", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=float(join_timeout_s))
        self._thread = None

    def clear(self, source_id: str) -> None:
        """Clear buffered PCM for a specific source.

        Important: we avoid flushing the global device output buffer by default.
        Flushing the device buffer can abruptly cut other sources (e.g. TTS).
        We only flush when it looks safe: no other enabled source has queued PCM.
        """

        should_flush = False
        with self._lock:
            src = self._sources.get(source_id)
            if src is not None:
                src.buffers.clear()

            # Only flush the device buffer if no other enabled source is queued.
            for other_id, other in self._sources.items():
                if other_id == source_id:
                    continue
                if not other.enabled:
                    continue
                if other.buffers:
                    should_flush = False
                    break
            else:
                should_flush = True

        if should_flush:
            audio = self._audio_backend()
            if audio is not None:
                try:
                    audio.clear_output_buffer()
                except Exception:
                    pass

    def set_enabled(self, source_id: str, enabled: bool) -> None:
        with self._lock:
            src = self._sources.get(source_id)
            if src is None:
                src = _Source(name=source_id, buffers=deque(), enabled=bool(enabled))
                self._sources[source_id] = src
            else:
                src.enabled = bool(enabled)

    def push(
        self,
        source_id: str,
        data: np.ndarray,
        *,
        rate: int,
        channels: int,
    ) -> None:
        """Push PCM to a source queue.

        `data` can be 1D interleaved or 2D; we normalize to float32 frames x channels.
        """

        if self._stop.is_set():
            return

        self.start()
        self._ensure_output_open()

        sr_out = self._sr or rate
        ch_out = self._ch or channels

        arr = _ensure_2d_f32(data)
        arr = _ensure_channels(arr, channels)
        if rate != sr_out:
            arr = _resample_f32(arr, origin_rate=int(rate), target_rate=int(sr_out))
        if channels != ch_out:
            arr = _ensure_channels(arr, int(ch_out))
        arr = np.clip(arr, -1.0, 1.0)
        arr = np.ascontiguousarray(arr)

        with self._lock:
            src = self._sources.get(source_id)
            if src is None:
                src = _Source(name=source_id, buffers=deque())
                self._sources[source_id] = src
            src.buffers.append(arr)

    def _audio_backend(self):  # type: ignore[no-untyped-def]
        return getattr(self._mini.media, "audio", None)

    def _output_buffer_len(self) -> int:
        audio = self._audio_backend()
        if audio is None:
            return 0
        buf = getattr(audio, "_output_buffer", None)
        lock = getattr(audio, "_output_lock", None)
        if buf is None:
            return 0
        try:
            if lock is not None:
                with lock:
                    return int(len(buf))
            return int(len(buf))
        except Exception:
            return 0

    def _get_output_stream_samplerate(self) -> int | None:
        audio = self._audio_backend()
        if audio is None:
            return None
        stream = getattr(audio, "_output_stream", None)
        if stream is None:
            return None
        sr = getattr(stream, "samplerate", None)
        try:
            return int(sr) if sr is not None else None
        except Exception:
            return None

    def _open_sounddevice_output_stream(self, audio, *, samplerate: int) -> bool:  # type: ignore[no-untyped-def]
        callback = getattr(audio, "_output_callback", None)
        device_id = getattr(audio, "_output_device_id", None)
        if callback is None:
            return False

        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            return False

        try:
            audio.stop_playing()
        except Exception:
            pass
        try:
            audio.clear_output_buffer()
        except Exception:
            pass

        try:
            audio._output_stream = sd.OutputStream(
                samplerate=int(samplerate),
                device=device_id,
                callback=callback,
            )
            audio._output_stream.start()
            self._logger.info("Audio output stream opened (forced samplerate=%s)", samplerate)
            return True
        except Exception as e:
            self._logger.warning("Failed to open forced output stream samplerate=%s: %s", samplerate, e)
            try:
                audio._output_stream = None
            except Exception:
                pass
            return False

    def _ensure_output_open(self) -> None:
        audio = self._audio_backend()
        if audio is None:
            raise RuntimeError("ReachyMini media audio backend is not initialized")

        if getattr(audio, "_output_stream", None) is None:
            if self._force_output_sr and self._open_sounddevice_output_stream(audio, samplerate=self._force_output_sr):
                pass
            else:
                self._mini.media.start_playing()

        if self._force_output_sr:
            current_sr = self._get_output_stream_samplerate()
            if current_sr is not None and int(current_sr) != int(self._force_output_sr):
                self._open_sounddevice_output_stream(audio, samplerate=self._force_output_sr)

        sr = self._get_output_stream_samplerate() or int(self._mini.media.get_output_audio_samplerate())
        ch = int(self._mini.media.get_output_channels())
        if sr <= 0:
            sr = 16000
        if ch <= 0:
            ch = 2
        self._sr = sr
        self._ch = ch

    def _take_frames(self, src: _Source, frames: int, *, ch_out: int) -> np.ndarray:
        if frames <= 0:
            return np.zeros((0, ch_out), dtype=np.float32)

        out_parts: list[np.ndarray] = []
        remain = frames
        while remain > 0 and src.buffers:
            head = src.buffers[0]
            if head.shape[0] <= remain:
                out_parts.append(head)
                remain -= head.shape[0]
                src.buffers.popleft()
            else:
                out_parts.append(head[:remain])
                src.buffers[0] = head[remain:]
                remain = 0

        if not out_parts:
            return np.zeros((frames, ch_out), dtype=np.float32)

        chunk = np.concatenate(out_parts, axis=0) if len(out_parts) > 1 else out_parts[0]
        if chunk.shape[0] < frames:
            pad = np.zeros((frames - chunk.shape[0], ch_out), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk

    def _loop(self) -> None:
        next_ts = time.monotonic()
        while not self._stop.is_set():
            try:
                self._ensure_output_open()
                sr = int(self._sr or 16000)
                ch = int(self._ch or 2)
                frames = max(1, int(sr * self._chunk_ms / 1000))

                with self._lock:
                    sources = list(self._sources.values())

                if not sources:
                    time.sleep(0.02)
                    continue

                mixed = np.zeros((frames, ch), dtype=np.float32)
                any_data = False
                with self._lock:
                    for src in sources:
                        if not src.enabled:
                            continue
                        if not src.buffers:
                            continue
                        any_data = True
                        mixed += self._take_frames(src, frames, ch_out=ch)

                if any_data:
                    # Don't over-queue: keep output latency small so pause/stop are responsive.
                    if self._output_buffer_len() >= self._max_output_buffers:
                        time.sleep(0.005)
                        continue
                    mixed = np.clip(mixed, -1.0, 1.0)
                    self._mini.media.push_audio_sample(mixed)

                # Pace output.
                next_ts += float(frames) / float(sr)
                sleep_s = next_ts - time.monotonic()
                if sleep_s > 0:
                    time.sleep(min(sleep_s, 0.05))
                else:
                    # If lagging, reset.
                    next_ts = time.monotonic()
            except Exception as e:
                self._logger.warning("AudioMixer loop error: %s", e)
                time.sleep(0.05)


class AudioMixerProvider(Provider[AudioMixer]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> AudioMixer:
        mini = con.force_fetch(ReachyMini)
        mixer = AudioMixer(mini, logger=con.get(LoggerItf))
        # Lazy start: do not open the PortAudio output stream at process boot.
        # On some platforms, keeping an output stream open can prevent other
        # PortAudio clients (e.g. PyAudio microphone capture for ASR/PTT) from
        # starting properly. Mixer starts automatically on first `push()`.
        return mixer
