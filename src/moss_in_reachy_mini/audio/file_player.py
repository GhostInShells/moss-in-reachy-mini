import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from reachy_mini import ReachyMini

from moss_in_reachy_mini.audio.mixer import AudioMixer
from moss_in_reachy_mini.audio.pcm_utils import ensure_2d_f32, ensure_channels, resample_f32, safe_int

logger = logging.getLogger(__name__)


@dataclass
class AudioPlaybackStatus:
    state: str = "idle"  # idle|playing|paused|stopped|error
    source: str = ""
    position_s: float = 0.0
    duration_s: float | None = None
    error: str | None = None

    def to_str(self) -> str:
        dur = "unknown" if self.duration_s is None else f"{self.duration_s:.2f}s"
        err = "" if self.error is None else f" error={self.error}"
        return f"state={self.state} source={self.source} position={self.position_s:.2f}s duration={dur}{err}"


def _ensure_2d_f32(data: np.ndarray) -> np.ndarray:
    # Keep original behavior: do not auto-transpose (channels, frames).
    return ensure_2d_f32(data, allow_transpose=False)


def _ensure_channels(data: np.ndarray, channels: int) -> np.ndarray:
    # Keep original behavior: when channels<=0 default to stereo.
    return ensure_channels(
        data,
        channels,
        default_channels=2,
        allow_transpose=False,
    )


def _resample_f32(data: np.ndarray, *, origin_rate: int, target_rate: int) -> np.ndarray:
    # Keep original behavior: do not auto-transpose (channels, frames).
    return resample_f32(data, origin_rate=origin_rate, target_rate=target_rate, allow_transpose=False)


def _normalize_pyav_audio_frame(raw: np.ndarray, out_frame, *, ch_out: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Normalize PyAV audio frame ndarray to float32 (frames, channels).

    PyAV/FFmpeg can surface audio frames in multiple layouts depending on codec
    and build options.
    """

    arr = raw

    # Metadata from PyAV.
    samples: int | None
    try:
        samples = int(getattr(out_frame, "samples", 0) or 0) or None
    except Exception:
        samples = None

    ch_in: int | None
    try:
        frame_layout = getattr(out_frame, "layout", None)
        ch_in = int(getattr(frame_layout, "channels", 0) or 0) or None
    except Exception:
        ch_in = None
    if not ch_in:
        try:
            ch_in = int(getattr(out_frame, "channels", 0) or 0) or None
        except Exception:
            ch_in = None

    is_planar = None
    try:
        is_planar = getattr(getattr(out_frame, "format", None), "is_planar", None)
    except Exception:
        is_planar = None

    # Normalize shape.
    if arr.ndim == 2:
        # Some builds return packed audio as a 2D row/col vector,
        # e.g. (1, samples * channels) instead of (samples, channels).
        if is_planar is False and samples is not None and samples > 0:
            # Row vector case: (1, samples * channels)
            if arr.shape[0] == 1 and arr.shape[1] % samples == 0 and (arr.shape[1] // samples) <= 8:
                inferred_ch = int(arr.shape[1] // samples)
                if inferred_ch > 0:
                    ch_in = ch_in or inferred_ch
                    arr = arr.reshape((samples, inferred_ch))
            # Column vector case: (samples * channels, 1)
            elif arr.shape[1] == 1 and arr.shape[0] % samples == 0 and (arr.shape[0] // samples) <= 8:
                inferred_ch = int(arr.shape[0] // samples)
                if inferred_ch > 0:
                    ch_in = ch_in or inferred_ch
                    arr = arr.reshape((samples, inferred_ch))

        # Planar: (channels, samples)
        if is_planar is True or (ch_in and arr.shape[0] == ch_in and arr.shape[0] <= 8):
            arr = arr.T

        if arr.shape[0] == 0:
            return np.zeros((0, max(1, ch_out)), dtype=np.float32)
    elif arr.ndim == 1:
        # Packed interleaved: reshape to (samples, channels).
        if ch_in and ch_in > 1:
            if samples and arr.size == samples * ch_in:
                arr = arr.reshape((samples, ch_in))
            elif arr.size % ch_in == 0:
                arr = arr.reshape((arr.size // ch_in, ch_in))
            else:
                arr = arr[:, None]
        else:
            arr = arr[:, None]
    else:
        raise ValueError(f"Unsupported PyAV audio ndarray shape: {arr.shape}")

    arr = _ensure_channels(arr, ch_out)
    arr = np.clip(arr.astype(np.float32, copy=False), -1.0, 1.0)
    return np.ascontiguousarray(arr)


class ReachyMiniAudioFilePlayer:
    """Decode audio and stream PCM to ReachyMini audio output.

    This class is intentionally conservative:
    - A single playback at a time (new `play` stops previous).
    - pause/resume only affects pushing new PCM (buffered audio may still play).
    """

    def __init__(
        self,
        mini: ReachyMini,
        *,
        chunk_ms: int = 40,
        safety_delay_s: float = 0.2,
        mixer: AudioMixer | None = None,
    ) -> None:
        self._mini = mini
        self._mixer = mixer
        self._chunk_ms = int(chunk_ms)
        self._safety_delay_s = float(safety_delay_s)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pause_gate = threading.Event()
        self._pause_gate.set()
        self._thread: threading.Thread | None = None
        self._status = AudioPlaybackStatus()
        self._target_buffer_s = 0.5

        # Timing model for status + pacing.
        # We estimate played time from a monotonic clock, excluding user-triggered pause time.
        # (Buffered audio may still play briefly while paused; we treat pause as "time freezes".)
        self._timing_start_ts: float | None = None
        self._pause_started_ts: float | None = None
        self._paused_accum_s: float = 0.0

        self._debug = os.getenv("REACHY_MINI_AUDIO_DEBUG", "").lower() in {"1", "true", "yes"}

        # Optional override: force output stream sample rate.
        # Useful on some macOS setups where the selected device reports a low
        # default samplerate (e.g. 16000) but playback sounds time-stretched.
        # When a mixer is used, it owns the output stream settings.
        self._force_output_sr = None if mixer is not None else safe_int(os.getenv("REACHY_MINI_AUDIO_OUTPUT_SR"))

    def _audio_backend(self):  # type: ignore[no-untyped-def]
        return getattr(self._mini.media, "audio", None)

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
        """Best-effort open a SoundDevice output stream with a specific samplerate.

        This uses reachy_mini's internal attributes. Only runs when the backend
        exposes `_output_callback` and `_output_device_id`.
        """

        callback = getattr(audio, "_output_callback", None)
        device_id = getattr(audio, "_output_device_id", None)
        if callback is None:
            return False

        import sounddevice as sd  # type: ignore

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
            logger.info("Audio output stream opened (forced samplerate=%s)", samplerate)
            return True
        except Exception as e:
            logger.warning("Failed to open forced output stream samplerate=%s: %s", samplerate, e)
            try:
                audio._output_stream = None
            except Exception:
                pass
            return False

    def _ensure_output_open(self) -> None:
        if self._mixer is not None:
            # Mixer owns the output stream.
            return
        # Best-effort ensure the underlying audio output stream is open.
        # Some environments may close the output stream from other components
        # (e.g. concurrent TTS). In that case, subsequent push calls will be dropped.
        # We try to reopen it proactively.

        audio = self._audio_backend()
        if audio is None:
            raise RuntimeError("ReachyMini media audio backend is not initialized")

        # If we want a forced samplerate, try to open the stream with that rate.
        # This avoids a "start_playing -> stop_playing -> reopen" churn.
        if self._force_output_sr and getattr(audio, "_output_stream", None) is None:
            if self._open_sounddevice_output_stream(audio, samplerate=self._force_output_sr):
                return

        # Default open.
        if getattr(audio, "_output_stream", None) is None:
            self._mini.media.start_playing()

        # If the stream exists but samplerate differs, best-effort reopen.
        if self._force_output_sr:
            current_sr = self._get_output_stream_samplerate()
            if current_sr is not None and int(current_sr) != int(self._force_output_sr):
                self._open_sounddevice_output_stream(audio, samplerate=self._force_output_sr)

        if self._debug:
            actual_sr = self._get_output_stream_samplerate()
            logger.info("Audio output stream samplerate=%s", actual_sr)

    def status(self) -> AudioPlaybackStatus:
        with self._lock:
            return AudioPlaybackStatus(**self._status.__dict__)

    def pause(self) -> None:
        # With a mixer, pausing is a mixer concern (disable the `music` source),
        # and should not depend on the local status machine.
        if self._mixer is not None:
            with self._lock:
                self._status.state = "paused"
                if self._pause_started_ts is None and self._timing_start_ts is not None:
                    self._pause_started_ts = time.monotonic()
            self._pause_gate.clear()
            # Drop already queued output chunks so pause is audible immediately.
            self._mixer.clear("music")
            self._mixer.set_enabled("music", False)
            return

        with self._lock:
            if self._status.state != "playing":
                return
            self._status.state = "paused"
            if self._pause_started_ts is None and self._timing_start_ts is not None:
                self._pause_started_ts = time.monotonic()
        self._pause_gate.clear()

    def resume(self) -> None:
        if self._mixer is not None:
            with self._lock:
                # Resume should be idempotent when mixer is used.
                self._status.state = "playing"
                if self._pause_started_ts is not None:
                    self._paused_accum_s += max(0.0, time.monotonic() - self._pause_started_ts)
                    self._pause_started_ts = None
            self._pause_gate.set()
            self._mixer.set_enabled("music", True)
            return

        with self._lock:
            if self._status.state != "paused":
                return
            self._status.state = "playing"
            if self._pause_started_ts is not None:
                self._paused_accum_s += max(0.0, time.monotonic() - self._pause_started_ts)
                self._pause_started_ts = None
        self._pause_gate.set()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        self._stop.set()
        self._pause_gate.set()

        if self._mixer is not None:
            # Stop means drop queued audio immediately.
            self._mixer.clear("music")
            self._mixer.set_enabled("music", False)

        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=float(join_timeout_s))

        with self._lock:
            if self._status.state in {"playing", "paused"}:
                self._status.state = "stopped"
            self._pause_started_ts = None

    def play(self, source: str) -> None:
        # Stop previous playback first.
        self.stop(join_timeout_s=1.0)

        # Re-enable music on new playback.
        if self._mixer is not None:
            self._mixer.set_enabled("music", True)

        with self._lock:
            self._stop.clear()
            self._pause_gate.set()
            self._status = AudioPlaybackStatus(state="playing", source=source, position_s=0.0)
            self._timing_start_ts = None
            self._pause_started_ts = None
            self._paused_accum_s = 0.0

        self._thread = threading.Thread(target=self._worker, args=(source,), daemon=True, name="ReachyMiniAudioFile")
        self._thread.start()

    def _worker(self, source: str) -> None:
        started = False
        try:
            started = self._stream_source(source)
            with self._lock:
                if self._status.state not in {"stopped", "error"}:
                    self._status.state = "idle"
        except Exception as e:
            logger.exception("Audio playback failed")
            with self._lock:
                self._status.state = "error"
                self._status.error = str(e)
        finally:
            if started:
                pass

    def _stream_source(self, source: str) -> bool:
        # Prefer PyAV when available (supports many codecs and URLs).
        # Can be disabled for troubleshooting dylib conflicts (e.g. macOS cv2 vs av).
        disable_pyav = os.getenv("REACHY_MINI_AUDIO_DISABLE_PYAV", "").lower() in {"1", "true", "yes"}

        if disable_pyav:
            av = None  # type: ignore
        else:
            try:
                import av  # type: ignore
            except Exception:
                av = None  # type: ignore

        # Ensure the output stream is open early so we can use the actual
        # samplerate it runs at.
        self._ensure_output_open()

        sr_out = self._get_output_stream_samplerate() or int(self._mini.media.get_output_audio_samplerate())
        ch_out = int(self._mini.media.get_output_channels())
        if sr_out <= 0:
            sr_out = 16000
        if ch_out <= 0:
            ch_out = 2

        frames_per_chunk = max(1, int(sr_out * self._chunk_ms / 1000))
        start_ts = time.monotonic()
        with self._lock:
            self._timing_start_ts = start_ts
            self._pause_started_ts = None
            self._paused_accum_s = 0.0
        pushed_frames = 0

        if av is not None:
            self._stream_with_pyav(
                av,
                source,
                sr_out=sr_out,
                ch_out=ch_out,
                frames_per_chunk=frames_per_chunk,
                pushed_frames_ref=[pushed_frames],
            )
            return True

        # Fallback: local files only via soundfile.
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Audio source not found and PyAV is unavailable: {source}")

        # Libsndfile (soundfile) does not support m4a/aac in many builds.
        if p.suffix.lower() in {".m4a", ".aac"}:
            raise RuntimeError(
                f"PyAV ('av') is required to decode {p.suffix} files. "
                f"Please install 'av' (and ffmpeg libs) on the robot. source={source}"
            )

        import soundfile as sf

        data, sr_in = sf.read(str(p), dtype="float32", always_2d=True)
        if data.size == 0:
            return False

        logger.info(
            "Audio decode via soundfile: src=%s sr_in=%s shape=%s -> sr_out=%s ch_out=%s",
            source,
            sr_in,
            tuple(data.shape),
            sr_out,
            ch_out,
        )

        data = np.clip(data.astype(np.float32, copy=False), -1.0, 1.0)
        data = _resample_f32(data, origin_rate=int(sr_in), target_rate=sr_out)
        data = np.clip(data, -1.0, 1.0)
        data = _ensure_channels(data, ch_out)

        duration_s = float(data.shape[0]) / float(sr_out)
        with self._lock:
            self._status.duration_s = duration_s

        self._push_in_chunks(
            data,
            sr_out=sr_out,
            frames_per_chunk=frames_per_chunk,
            pushed_frames_ref=[pushed_frames],
        )

        return True

    def _stream_with_pyav(
        self,
        av,
        source: str,
        *,
        sr_out: int,
        ch_out: int,
        frames_per_chunk: int,
        pushed_frames_ref: list[int],
    ) -> None:  # type: ignore[no-untyped-def]
        container = av.open(source)
        try:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return

            logger.info(
                "Audio decode via PyAV: src=%s -> sr_out=%s ch_out=%s",
                source,
                sr_out,
                ch_out,
            )

            # duration is in microseconds for container.duration.
            if container.duration is not None:
                with self._lock:
                    self._status.duration_s = float(container.duration) / 1_000_000.0

            layout = "stereo" if ch_out == 2 else ("mono" if ch_out == 1 else None)
            # IMPORTANT:
            # Do NOT request PyAV/FFmpeg to change the sample rate here.
            #
            # On some environments (notably macOS with multiple ffmpeg dylibs
            # from cv2 + av), we observed cases where the output frame reports
            # `sample_rate == sr_out` (e.g. 16000) while the PCM data still
            # behaves like the input rate (e.g. 48000), resulting in a slowed
            # down playback when pushed into a 16k output stream.
            #
            # We therefore only use the resampler to convert sample format/layout
            # to float, and always do sample-rate conversion ourselves using
            # `_resample_f32` when needed.
            resampler = av.audio.resampler.AudioResampler(format="flt", layout=layout)

            logged_rates = False
            pushed_any = False

            src_sr: int | None = None
            src_chunk_frames: int | None = None

            # Buffer in the *source* sample rate domain (e.g. 48k). We resample
            # in bigger chunks to reduce Python overhead and avoid underruns
            # (which can sound like "slow playback").
            src_buffer: list[np.ndarray] = []
            src_buffered_frames = 0

            for packet in container.demux(stream):
                if self._stop.is_set():
                    break

                # Pause gate: blocks pushing new PCM (buffered audio may still play).
                self._pause_gate.wait()
                if self._stop.is_set():
                    break

                for frame in packet.decode():
                    if self._stop.is_set():
                        break
                    self._pause_gate.wait()

                    if frame is None:
                        continue

                    # Resample to output format/rate/layout.
                    # NOTE: PyAV may return a single frame or a list of frames.
                    res = resampler.resample(frame)
                    if res is None:
                        continue

                    frames = res if isinstance(res, list) else [res]
                    for out_frame in frames:
                        if out_frame is None:
                            continue

                        frame_sr = getattr(out_frame, "sample_rate", None) or getattr(out_frame, "rate", None)
                        if not logged_rates:
                            try:
                                in_sr = getattr(frame, "sample_rate", None) or getattr(frame, "rate", None)
                                logger.info(
                                    "PyAV frame rates: in_sr=%s out_sr=%s target_sr=%s",
                                    in_sr,
                                    frame_sr,
                                    sr_out,
                                )
                            except Exception:
                                pass
                            logged_rates = True

                        if frame_sr is not None and int(frame_sr) > 0:
                            cur_sr = int(frame_sr)
                        else:
                            # Fallback: use stream/sample rate.
                            maybe = getattr(frame, "sample_rate", None) or getattr(frame, "rate", None)
                            cur_sr = int(maybe) if maybe else sr_out

                        if src_sr is None:
                            src_sr = cur_sr
                            # How many source frames correspond to one output chunk.
                            src_chunk_frames = max(1, int(round(frames_per_chunk * float(src_sr) / float(sr_out))))
                            logger.info(
                                "PyAV stream inferred src_sr=%s -> sr_out=%s, src_chunk_frames=%s",
                                src_sr,
                                sr_out,
                                src_chunk_frames,
                            )
                        elif cur_sr != src_sr:
                            # Rare for typical files; ignore mismatch and treat as src_sr.
                            cur_sr = src_sr

                        raw_arr = out_frame.to_ndarray()
                        arr = _normalize_pyav_audio_frame(raw_arr, out_frame, ch_out=ch_out)
                        if self._debug:
                            try:
                                logger.info(
                                    "PyAV frame: raw_shape=%s samples=%s layout=%s planar=%s -> norm_shape=%s",
                                    tuple(getattr(raw_arr, "shape", ())),
                                    getattr(out_frame, "samples", None),
                                    getattr(getattr(out_frame, "layout", None), "name", None),
                                    getattr(getattr(out_frame, "format", None), "is_planar", None),
                                    tuple(arr.shape),
                                )
                            except Exception:
                                pass

                        # Buffer at source sample rate.
                        src_buffer.append(arr)
                        src_buffered_frames += int(arr.shape[0])

                    # Keep latency bounded: push roughly in frames_per_chunk.
                    if src_sr is None or src_chunk_frames is None:
                        continue

                    while src_buffered_frames >= src_chunk_frames and not self._stop.is_set():
                        self._pause_gate.wait()
                        self._ensure_output_open()

                        chunk_src = self._pop_from_buffer(src_buffer, src_chunk_frames)
                        src_buffered_frames -= int(chunk_src.shape[0])

                        if src_sr != sr_out:
                            chunk = _resample_f32(chunk_src, origin_rate=src_sr, target_rate=sr_out)
                        else:
                            chunk = chunk_src

                        if self._mixer is not None:
                            self._mixer.push(
                                "music",
                                chunk,
                                rate=sr_out,
                                channels=ch_out,
                            )
                        else:
                            self._mini.media.push_audio_sample(chunk)
                        pushed_any = True
                        pushed_frames_ref[0] += int(chunk.shape[0])
                        self._pace(sr_out=sr_out, pushed_frames=pushed_frames_ref[0])

            # Flush remaining buffered audio.
            if not self._stop.is_set() and src_buffer:
                tail_src = np.concatenate(src_buffer, axis=0) if len(src_buffer) > 1 else src_buffer[0]
                if src_sr is not None and src_sr != sr_out:
                    tail = _resample_f32(tail_src, origin_rate=src_sr, target_rate=sr_out)
                else:
                    tail = tail_src

                self._push_in_chunks(
                    tail,
                    sr_out=sr_out,
                    frames_per_chunk=frames_per_chunk,
                    pushed_frames_ref=pushed_frames_ref,
                )
                pushed_any = pushed_any or bool(tail.size)
        finally:
            try:
                container.close()
            except Exception:
                pass

        if not pushed_any and not self._stop.is_set():
            raise RuntimeError(f"PyAV decoded no playable audio frames: source={source}")

        # Allow device buffer to finish.
        # We typically keep ~self._target_buffer_s of audio queued.
        time.sleep(max(self._safety_delay_s, float(self._target_buffer_s) + 0.2))

    @staticmethod
    def _pop_from_buffer(buffer: list[np.ndarray], frames: int) -> np.ndarray:
        """Pop exactly `frames` from buffer list."""

        if not buffer:
            return np.zeros((0, 1), dtype=np.float32)

        out_parts: list[np.ndarray] = []
        remain = frames
        while remain > 0 and buffer:
            head = buffer[0]
            if head.shape[0] <= remain:
                out_parts.append(head)
                remain -= head.shape[0]
                buffer.pop(0)
            else:
                out_parts.append(head[:remain])
                buffer[0] = head[remain:]
                remain = 0
        return np.concatenate(out_parts, axis=0) if len(out_parts) > 1 else out_parts[0]

    def _push_in_chunks(
        self,
        data: np.ndarray,
        *,
        sr_out: int,
        frames_per_chunk: int,
        pushed_frames_ref: list[int],
    ) -> None:
        data = _ensure_2d_f32(data)
        i = 0
        total = int(data.shape[0])
        while i < total and not self._stop.is_set():
            self._pause_gate.wait()
            if self._stop.is_set():
                break
            self._ensure_output_open()
            chunk = data[i : i + frames_per_chunk]
            if chunk.size == 0:
                break
            if self._mixer is not None:
                self._mixer.push(
                    "music",
                    chunk,
                    rate=sr_out,
                    channels=int(chunk.shape[1]) if chunk.ndim == 2 else 1,
                )
            else:
                self._mini.media.push_audio_sample(chunk)
            pushed_frames_ref[0] += int(chunk.shape[0])
            i += frames_per_chunk

            self._pace(sr_out=sr_out, pushed_frames=pushed_frames_ref[0])

        time.sleep(self._safety_delay_s)

    def _pace(self, *, sr_out: int, pushed_frames: int) -> None:
        """Keep a small buffer ahead of playback, avoiding underruns."""

        now = time.monotonic()

        with self._lock:
            start_ts = self._timing_start_ts
            paused_accum_s = float(self._paused_accum_s)
            paused_at = self._pause_started_ts

        if start_ts is None:
            return

        # If currently paused, freeze the effective clock at the pause moment.
        effective_now = paused_at if paused_at is not None else now
        elapsed_s = max(0.0, float(effective_now - start_ts) - paused_accum_s)

        played_est = elapsed_s * float(sr_out)
        buffered = max(0.0, float(pushed_frames) - played_est)
        target = float(self._target_buffer_s) * float(sr_out)
        if buffered > target:
            # Sleep just enough to get close to the target buffer.
            sleep_s = (buffered - target) / float(sr_out)
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.1))

        # Update status based on estimated played time (not pushed time), so we
        # don't jump to EOF instantly.
        with self._lock:
            self._status.position_s = max(self._status.position_s, elapsed_s)
