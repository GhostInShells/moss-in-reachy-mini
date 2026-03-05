import datetime
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ghoshell_common.contracts import FileStorage, Workspace
from ghoshell_container import Container, IoCContainer, Provider
from reachy_mini import ReachyMini

from moss_in_reachy_mini.audio.mic_hub import MicHub, MicSubscription
from moss_in_reachy_mini.camera.camera_worker import CameraWorker
from moss_in_reachy_mini.camera.frame_hub import FrameHub
from moss_in_reachy_mini.video.recorder_debug_logger import get_video_recorder_logger
from moss_in_reachy_mini.video.settings import VideoRecordSettings

logger = logging.getLogger(__name__)


@dataclass
class RecordingInfo:
    recording: bool
    file_name: str = ""
    started_at: float = 0.0
    note: str = ""


@dataclass
class RecordingResult:
    saved_path: str
    duration_s: int
    has_audio_out: bool
    has_audio_in: bool
    started_at_ts: int
    stopped_at_ts: int
    meta_path: str = ""


@dataclass
class _StartRequest:
    file_name: str
    include_mic: bool


class VideoRecorderWorker:
    """Background A/V recorder.

    - Video: from FrameHub (single camera capture source)
    - Audio output: tapped from ReachyMiniStreamPlayer (push_output_audio)
    - Audio input: recorded from system microphone via PyAudio (optional)
    - Mux/encode: ffmpeg subprocess
    """

    def __init__(
        self,
        mini: ReachyMini,
        frame_hub: FrameHub,
        camera_worker: CameraWorker | None,
        storage: FileStorage,
        *,
        fps: int = 10,
        mic_enabled: bool = True,
        mic_rate: int = 48000,
        mic_channels: int = 1,
        x264_crf: int = 18,
        x264_preset: str = "fast",
        audio_bitrate_kbps: int = 256,
        frame_source: str = "raw",
        scale: str = "",
        max_width: int = 0,
        max_height: int = 0,
        keep_tmp: bool = False,
        container: IoCContainer | None = None,
    ):
        self._mini = mini
        self._frame_hub = frame_hub
        self._camera_worker = camera_worker
        self._storage = storage
        self._fps = max(1, int(fps))
        self._mic_enabled = mic_enabled
        self._x264_crf = int(x264_crf)
        self._x264_preset = (x264_preset or "fast").strip()
        self._audio_bitrate_kbps = int(audio_bitrate_kbps)
        self._frame_source = (frame_source or "raw").strip().lower()
        self._scale = (scale or "").strip()
        self._max_width = max(0, int(max_width))
        self._max_height = max(0, int(max_height))
        self._keep_tmp = bool(keep_tmp)

        self._container = Container(parent=container)
        # Use a dedicated file logger to avoid being swallowed by Rich/Prompt console UI.
        self._logger = get_video_recorder_logger(storage.abspath())

        self._recording_lock = threading.Lock()
        self._info = RecordingInfo(recording=False)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._ffmpeg: subprocess.Popen | None = None
        self._ffmpeg_err: bytes = b""
        self._ffmpeg_err_thread: threading.Thread | None = None
        self._video_stdin = None

        self._rec_dir: str | None = None
        self._tmp_video_path: str | None = None
        self._audio_out_wav: wave.Wave_write | None = None
        self._audio_in_wav: wave.Wave_write | None = None

        self._last_result: RecordingResult | None = None

        # Output audio is event-based (only pushed when TTS plays). To keep A/V timeline aligned,
        # we store timestamps and fill silence gaps when writing the wav.
        self._audio_out_q: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=400)
        self._audio_in_q: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._mic_thread: threading.Thread | None = None
        self._mic_sub: MicSubscription | None = None

        # Monotonic start time for A/V timeline alignment.
        self._av_start_mono: float | None = None
        self._audio_out_cursor_s: float = 0.0

        # Start requests are executed in the worker loop to avoid blocking the agent thread.
        self._start_q: queue.Queue[_StartRequest] = queue.Queue(maxsize=1)
        self._start_failed: str = ""

        # Resolve output audio format from mini
        self._audio_out_rate = int(self._mini.media.get_output_audio_samplerate())
        self._audio_out_channels = int(self._mini.media.get_output_channels())

        # Mic format (best effort; may fallback on open failure)
        self._audio_in_rate = int(mic_rate)
        self._audio_in_channels = max(1, int(mic_channels))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._logger.info("worker thread started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._logger.info("worker thread stopped")

    def status(self) -> RecordingInfo:
        with self._recording_lock:
            return RecordingInfo(**self._info.__dict__)

    def start_recording(self, note: str = "") -> str:
        self._logger.info("start_recording note=%r", note)
        self.start()
        with self._recording_lock:
            if self._info.recording:
                self._logger.info("already recording: %s", self._info.file_name)
                return self._info.file_name

            now = time.time()
            ts = datetime.datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
            safe_note = "".join(c for c in note.strip().replace(" ", "_") if c.isalnum() or c in "-_")
            suffix = f"_{safe_note}" if safe_note else ""
            file_name = f"reachy_mini_{ts}{suffix}.mp4"
            self._info = RecordingInfo(recording=True, file_name=file_name, started_at=now, note=note)
            self._last_result = None

        # Clear any stale buffered audio from previous sessions
        while not self._audio_out_q.empty():
            try:
                self._audio_out_q.get_nowait()
                self._audio_out_q.task_done()
            except queue.Empty:
                break
        while not self._audio_in_q.empty():
            try:
                self._audio_in_q.get_nowait()
                self._audio_in_q.task_done()
            except queue.Empty:
                break

        include_mic = self._mic_enabled and self._mic_supported()
        self._start_failed = ""
        self._av_start_mono = None
        self._audio_out_cursor_s = 0.0
        self._logger.info("queue start file=%s mic=%s", file_name, include_mic)
        try:
            # Best-effort: if a previous request is still pending, drop it.
            while not self._start_q.empty():
                try:
                    self._start_q.get_nowait()
                except queue.Empty:
                    break
            self._start_q.put_nowait(_StartRequest(file_name=file_name, include_mic=include_mic))
        except queue.Full:
            # Shouldn't happen due to draining above; fail closed.
            self._start_failed = "start queue full"
            with self._recording_lock:
                self._info = RecordingInfo(recording=False)
            raise RuntimeError("failed to start recording: start queue full")

        return file_name

    def stop_recording(self) -> str:
        self._logger.info("stop_recording")
        with self._recording_lock:
            if not self._info.recording:
                self._logger.info("stop_recording: not recording")
                return ""
            file_name = self._info.file_name
            started_at = self._info.started_at
            self._info = RecordingInfo(recording=False)

        # If start hasn't actually completed yet, cancel pending request and exit.
        if self._ffmpeg is None and self._rec_dir is None:
            try:
                while not self._start_q.empty():
                    self._start_q.get_nowait()
            except Exception:
                pass
            self._logger.warning("stop_recording: recording not fully started")
            return ""

        self._stop_mic()
        out_path = os.path.join(self._storage.abspath(), file_name)
        result = self._finalize_recording(out_path, started_at=started_at)
        self._last_result = result
        return out_path

    def last_result(self) -> RecordingResult | None:
        return self._last_result

    def last_error(self) -> str:
        return self._start_failed

    def push_output_audio(self, pcm_bytes: bytes) -> None:
        """Tap point for robot output audio (int16 PCM interleaved)."""
        if not pcm_bytes:
            return
        if not self.status().recording:
            return
        try:
            self._audio_out_q.put_nowait((time.monotonic(), pcm_bytes))
        except queue.Full:
            # drop if overloaded
            pass

    def _mic_supported(self) -> bool:
        try:
            _ = self._container.force_fetch(MicHub)
            return True
        except Exception:
            # Fallback: PyAudio direct capture if MicHub not available
            try:
                import pyaudio  # type: ignore

                _ = pyaudio
                return True
            except Exception:
                return False

    def _is_annotated(self) -> bool:
        return self._frame_source in ("annotated", "anno", "overlay")

    def _get_latest_video_frame(self) -> Optional[np.ndarray]:
        """Return frame in RGB format."""
        if self._is_annotated():
            if self._camera_worker is None:
                return None
            cam_frame = self._camera_worker.get_latest_frame()
            if cam_frame is None or cam_frame.image is None:
                return None
            # CameraWorker stores RGB already
            return cam_frame.image.copy()

        raw = self._frame_hub.get_latest_frame()
        if raw is None:
            return None
        # FrameHub provides BGR from SDK; convert to RGB
        return raw[:, :, ::-1].copy()

    # ---- Internals ----

    def _open_ffmpeg(self, file_name: str, *, include_mic: bool) -> None:
        # Ensure directory exists
        os.makedirs(self._storage.abspath(), exist_ok=True)
        rec_dir = os.path.join(self._storage.abspath(), ".tmp", f".{file_name}.rec")
        os.makedirs(rec_dir, exist_ok=True)
        self._rec_dir = rec_dir
        self._tmp_video_path = os.path.join(rec_dir, "video.mp4")

        # Prepare wav writers
        audio_out_wav_path = os.path.join(rec_dir, "audio_out.wav")
        self._audio_out_wav = wave.open(audio_out_wav_path, "wb")
        self._audio_out_wav.setnchannels(self._audio_out_channels)
        self._audio_out_wav.setsampwidth(2)
        self._audio_out_wav.setframerate(self._audio_out_rate)

        if include_mic:
            audio_in_wav_path = os.path.join(rec_dir, "audio_in.wav")
            self._audio_in_wav = wave.open(audio_in_wav_path, "wb")
            self._audio_in_wav.setnchannels(self._audio_in_channels)
            self._audio_in_wav.setsampwidth(2)
            self._audio_in_wav.setframerate(self._audio_in_rate)
        else:
            self._audio_in_wav = None

        # Wait for first frame to know width/height.
        # IMPORTANT: do NOT call `mini.media.get_frame()` here, as the camera capture
        # loop should be centralized in FrameHub to avoid concurrent camera access.
        start_at = time.time()
        frame = None
        while frame is None and time.time() - start_at < 5.0:
            frame = self._get_latest_video_frame()
            if frame is None:
                time.sleep(0.05)
        if frame is None:
            raise RuntimeError("no camera frame available for recording")

        waited_s = time.time() - start_at
        self._logger.info("got first frame after %.2fs", waited_s)

        height, width = frame.shape[0], frame.shape[1]
        self._logger.info("frame size: %dx%d fps=%s", width, height, self._fps)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
        ]

        cmd.extend(["-map", "0:v"])

        # Video scaling / resizing
        vf = ""
        if self._scale:
            vf = f"scale={self._scale}"
        elif self._max_width > 0 or self._max_height > 0:
            # Preserve aspect ratio; ensure even dimensions for yuv420p with -2
            if self._max_width > 0 and self._max_height > 0:
                vf = (
                    f"scale=w={self._max_width}:h={self._max_height}:force_original_aspect_ratio=decrease,"
                    f"pad={self._max_width}:{self._max_height}:(ow-iw)/2:(oh-ih)/2"
                )
            elif self._max_width > 0:
                vf = f"scale=w={self._max_width}:h=-2"
            else:
                vf = f"scale=w=-2:h={self._max_height}"

        if vf:
            cmd.extend(["-vf", vf])

        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                self._x264_preset,
                "-crf",
                str(self._x264_crf),
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                self._tmp_video_path,
            ]
        )

        self._logger.info("ffmpeg cmd: %s", " ".join(cmd))

        try:
            self._ffmpeg = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            self._close_audio_writers()
            raise RuntimeError("ffmpeg not found in PATH") from e

        self._ffmpeg_err = b""
        self._ffmpeg_err_thread = threading.Thread(target=self._read_ffmpeg_stderr, daemon=True)
        self._ffmpeg_err_thread.start()

        self._video_stdin = self._ffmpeg.stdin
        # Define A/V timebase at the moment the encoder pipeline is ready.
        self._av_start_mono = time.monotonic()
        self._audio_out_cursor_s = 0.0
        self._logger.info("recording started: %s", os.path.join(self._storage.abspath(), file_name))

    def _write_silence_out(self, seconds: float) -> None:
        if self._audio_out_wav is None:
            return
        if seconds <= 0:
            return
        frames = int(seconds * self._audio_out_rate)
        if frames <= 0:
            return
        chunk_frames = min(frames, self._audio_out_rate)  # at most 1s per write
        zero = (b"\x00\x00" * (chunk_frames * self._audio_out_channels))
        remaining = frames
        while remaining > 0:
            n = min(remaining, chunk_frames)
            if n != chunk_frames:
                zero = (b"\x00\x00" * (n * self._audio_out_channels))
            self._audio_out_wav.writeframes(zero)
            remaining -= n

    def _write_output_audio_event(self, ts_mono: float, pcm: bytes) -> None:
        if self._audio_out_wav is None:
            return
        start = self._av_start_mono
        if start is None:
            # Pipeline not ready yet; ignore for alignment.
            return

        # Clamp events to t>=0 on our timeline.
        rel_s = max(0.0, ts_mono - start)
        if rel_s > self._audio_out_cursor_s:
            self._write_silence_out(rel_s - self._audio_out_cursor_s)
            self._audio_out_cursor_s = rel_s

        # Write PCM and advance cursor by audio duration.
        self._audio_out_wav.writeframes(pcm)
        bytes_per_frame = 2 * self._audio_out_channels
        frames = len(pcm) // max(1, bytes_per_frame)
        self._audio_out_cursor_s += frames / float(self._audio_out_rate)

    def _close_ffmpeg(self) -> None:
        try:
            if self._video_stdin:
                self._video_stdin.close()
        except Exception as e:
            self._logger.warning("failed to close video stdin: %s", e)
        self._video_stdin = None

        if self._ffmpeg is None:
            return

        try:
            self._ffmpeg.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._ffmpeg.terminate()
            try:
                self._ffmpeg.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._ffmpeg.kill()
        finally:
            if self._ffmpeg_err_thread is not None:
                self._ffmpeg_err_thread.join(timeout=0.2)
            err = self._ffmpeg_err.decode("utf-8", errors="ignore")
            if err:
                self._logger.error("ffmpeg stderr: %s", err)
            if self._ffmpeg.returncode not in (0, None):
                raise RuntimeError(f"ffmpeg exited with code {self._ffmpeg.returncode}: {err}")
            self._ffmpeg = None
            self._ffmpeg_err_thread = None

    def _close_audio_writers(self) -> None:
        for w in (self._audio_out_wav, self._audio_in_wav):
            try:
                if w is not None:
                    w.close()
            except Exception as e:
                self._logger.warning("failed to close audio writer: %s", e)
        self._audio_out_wav = None
        self._audio_in_wav = None

    def _finalize_recording(self, out_path: str, *, started_at: float) -> RecordingResult:
        started_at_ts = int(started_at)
        stopped_at_ts = int(time.time())
        try:
            self._close_ffmpeg()
        finally:
            self._close_audio_writers()

        if not self._rec_dir or not self._tmp_video_path:
            raise RuntimeError("recording temp directory not initialized")

        tmp_video = self._tmp_video_path
        audio_out_wav_path = os.path.join(self._rec_dir, "audio_out.wav")
        audio_in_wav_path = os.path.join(self._rec_dir, "audio_in.wav")

        has_audio_out = os.path.exists(audio_out_wav_path) and os.path.getsize(audio_out_wav_path) > 44
        has_audio_in = os.path.exists(audio_in_wav_path) and os.path.getsize(audio_in_wav_path) > 44

        if not os.path.exists(tmp_video) or os.path.getsize(tmp_video) < 1024:
            raise RuntimeError("video temp file not created or too small")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if not has_audio_out and not has_audio_in:
            if self._keep_tmp:
                shutil.copyfile(tmp_video, out_path)
            else:
                os.replace(tmp_video, out_path)
                shutil.rmtree(self._rec_dir, ignore_errors=True)
            self._rec_dir = None
            self._tmp_video_path = None
            result = RecordingResult(
                saved_path=out_path,
                duration_s=max(0, int(time.time() - started_at)),
                has_audio_out=False,
                has_audio_in=False,
                started_at_ts=started_at_ts,
                stopped_at_ts=stopped_at_ts,
            )
            self._write_sidecar_json(result)
            return result

        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_video]
        if has_audio_out:
            cmd.extend(["-i", audio_out_wav_path])
        if has_audio_in:
            cmd.extend(["-i", audio_in_wav_path])

        if has_audio_out and has_audio_in:
            cmd.extend(
                [
                    "-filter_complex",
                    "[1:a][2:a]amix=inputs=2:duration=longest[a]",
                    "-map",
                    "0:v",
                    "-map",
                    "[a]",
                ]
            )
        else:
            cmd.extend(["-map", "0:v", "-map", "1:a"])

        cmd.extend(
            [
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                f"{self._audio_bitrate_kbps}k",
                "-movflags",
                "+faststart",
                out_path,
            ]
        )

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg mux failed: {err}")

        if not self._keep_tmp:
            shutil.rmtree(self._rec_dir, ignore_errors=True)
        self._rec_dir = None
        self._tmp_video_path = None
        result = RecordingResult(
            saved_path=out_path,
            duration_s=max(0, int(time.time() - started_at)),
            has_audio_out=has_audio_out,
            has_audio_in=has_audio_in,
            started_at_ts=started_at_ts,
            stopped_at_ts=stopped_at_ts,
        )
        self._write_sidecar_json(result)
        return result

    def _write_sidecar_json(self, result: RecordingResult) -> None:
        """Write metadata next to the final mp4 as `<name>.mp4.json`."""
        meta_path = f"{result.saved_path}.json"
        payload = {
            "saved_path": result.saved_path,
            "started_at_ts": result.started_at_ts,
            "stopped_at_ts": result.stopped_at_ts,
            "duration_s": result.duration_s,
            "has_audio_out": result.has_audio_out,
            "has_audio_in": result.has_audio_in,
            "keep_tmp": self._keep_tmp,
            "frame_source": self._frame_source,
            "fps": self._fps,
            "scale": self._scale,
            "max_width": self._max_width,
            "max_height": self._max_height,
            "x264_crf": self._x264_crf,
            "x264_preset": self._x264_preset,
            "audio_bitrate_kbps": self._audio_bitrate_kbps,
            "mic_enabled": self._mic_enabled,
            "mic_rate": self._audio_in_rate,
            "mic_channels": self._audio_in_channels,
            "audio_out_rate": self._audio_out_rate,
            "audio_out_channels": self._audio_out_channels,
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            result.meta_path = meta_path
        except Exception as e:
            self._logger.warning("failed to write sidecar json: %s", e)

    def _read_ffmpeg_stderr(self) -> None:
        if self._ffmpeg is None or self._ffmpeg.stderr is None:
            return
        try:
            self._ffmpeg_err = self._ffmpeg.stderr.read() or b""
        except Exception:
            self._ffmpeg_err = b""

    def _start_mic(self) -> None:
        if self._mic_thread is not None and self._mic_thread.is_alive():
            return
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._mic_thread.start()

    def _stop_mic(self) -> None:
        if self._mic_thread is None:
            return
        self._mic_thread.join(timeout=1.0)
        self._mic_thread = None
        if self._mic_sub is not None:
            try:
                self._mic_sub.close()
            except Exception:
                pass
        self._mic_sub = None

    def _mic_loop(self) -> None:
        # Prefer shared MicHub to avoid multi-stream conflicts with ASR/PTT.
        try:
            hub = self._container.force_fetch(MicHub)
        except Exception:
            hub = None

        if hub is not None:
            try:
                # Recorder subscription can drop oldest frames to avoid blocking capture.
                sub = hub.subscribe(name="video_recorder", max_queue=1200, drop_policy="drop_oldest")
                self._mic_sub = sub
                # Align recorder wav metadata to hub format.
                self._audio_in_rate = sub.rate
                self._audio_in_channels = sub.channels
                sub.drain()
            except Exception as e:
                self._logger.warning("failed to subscribe MicHub: %s", e)
                self._mic_sub = None

        if self._mic_sub is not None:
            while self.status().recording and not self._stop_event.is_set():
                try:
                    data = self._mic_sub.get(timeout=0.2)
                    self._mic_sub.task_done()
                except queue.Empty:
                    continue
                except Exception:
                    break
                try:
                    self._audio_in_q.put_nowait(data)
                except queue.Full:
                    pass
            return

        # Fallback: direct PyAudio capture (may conflict with other users of mic).
        try:
            import pyaudio  # type: ignore
        except Exception as e:
            self._logger.warning("PyAudio not available for mic recording: %s", e)
            return

        pa = pyaudio.PyAudio()
        stream = None
        for rate in (self._audio_in_rate, 44100, 16000):
            try:
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=self._audio_in_channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=1024,
                )
                self._audio_in_rate = rate
                break
            except Exception:
                stream = None
        if stream is None:
            self._logger.warning("failed to open microphone input")
            pa.terminate()
            return

        try:
            while self.status().recording and not self._stop_event.is_set():
                data = stream.read(1024, exception_on_overflow=False)
                try:
                    self._audio_in_q.put_nowait(data)
                except queue.Full:
                    pass
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                self._logger.warning("failed to close mic stream: %s", e)
            pa.terminate()

    def _loop(self) -> None:
        self._frame_hub.start()

        last_frame_at = 0.0
        frame_interval = 1.0 / self._fps
        last_heartbeat_at = 0.0

        while not self._stop_event.is_set():
            if not self.status().recording:
                time.sleep(0.05)
                continue

            # Lazily start ffmpeg inside worker thread so agent command won't block.
            if self._ffmpeg is None and self._rec_dir is None:
                try:
                    req = self._start_q.get_nowait()
                except queue.Empty:
                    req = None
                if req is not None:
                    try:
                        self._logger.info("background starting ffmpeg: %s", req.file_name)
                        self._open_ffmpeg(req.file_name, include_mic=req.include_mic)
                        # Fail fast if ffmpeg exits immediately (common when pipes are misconfigured).
                        time.sleep(0.05)
                        if self._ffmpeg is not None and self._ffmpeg.poll() is not None:
                            self._close_ffmpeg()
                        if req.include_mic:
                            self._start_mic()
                    except Exception as e:
                        self._start_failed = str(e)
                        self._logger.exception("failed to start")
                        with self._recording_lock:
                            self._info = RecordingInfo(recording=False)
                        # Best-effort cleanup
                        try:
                            self._close_audio_writers()
                        except Exception:
                            pass
                        try:
                            if self._rec_dir and not self._keep_tmp:
                                shutil.rmtree(self._rec_dir, ignore_errors=True)
                        except Exception:
                            pass
                        self._rec_dir = None
                        self._tmp_video_path = None
                        time.sleep(0.1)
                        continue

            now = time.time()
            if now - last_heartbeat_at >= 2.0:
                try:
                    info = self.status()
                    self._logger.debug(
                        "heartbeat file=%s out_q=%s in_q=%s",
                        info.file_name,
                        self._audio_out_q.qsize(),
                        self._audio_in_q.qsize(),
                    )
                except Exception:
                    pass
                last_heartbeat_at = now

            # ---- Video ----
            if now - last_frame_at >= frame_interval:
                frame = self._get_latest_video_frame()
                if frame is not None and self._video_stdin is not None:
                    try:
                        self._video_stdin.write(frame.tobytes())
                    except BlockingIOError:
                        pass
                    except Exception as e:
                        self._logger.warning("video write failed: %s", e)
                last_frame_at = now

            # ---- Audio output (TTS tap) ----
            if self._audio_out_wav is not None:
                try:
                    ts_mono, pcm = self._audio_out_q.get_nowait()
                    self._audio_out_q.task_done()
                    self._write_output_audio_event(ts_mono, pcm)
                except queue.Empty:
                    pass
                except Exception as e:
                    self._logger.warning("audio-out write failed: %s", e)

            # ---- Audio input (mic) ----
            if self._audio_in_wav is not None:
                try:
                    pcm = self._audio_in_q.get_nowait()
                    self._audio_in_q.task_done()
                    self._audio_in_wav.writeframes(pcm)
                except queue.Empty:
                    pass
                except Exception as e:
                    self._logger.warning("audio-in write failed: %s", e)

            time.sleep(0.005)


class VideoRecorderWorkerProvider(Provider[VideoRecorderWorker]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> VideoRecorderWorker:
        mini = con.force_fetch(ReachyMini)
        frame_hub = con.force_fetch(FrameHub)
        camera_worker = con.get(CameraWorker)
        ws = con.force_fetch(Workspace)
        storage = ws.runtime().sub_storage("video_records")
        if not hasattr(storage, "abspath"):
            raise TypeError("workspace runtime storage does not support abspath()")

        cfg = VideoRecordSettings()

        return VideoRecorderWorker(
            mini=mini,
            frame_hub=frame_hub,
            camera_worker=camera_worker,
            storage=storage,
            fps=cfg.video_record_fps,
            mic_enabled=cfg.video_record_mic_enabled,
            mic_rate=cfg.video_record_mic_rate,
            mic_channels=cfg.video_record_mic_channels,
            x264_crf=cfg.video_record_x264_crf,
            x264_preset=cfg.video_record_x264_preset,
            audio_bitrate_kbps=cfg.video_record_audio_bitrate_kbps,
            frame_source=cfg.video_record_frame_source,
            scale=cfg.video_record_scale,
            max_width=cfg.video_record_max_width,
            max_height=cfg.video_record_max_height,
            keep_tmp=cfg.video_record_keep_tmp,
            container=con,
        )
