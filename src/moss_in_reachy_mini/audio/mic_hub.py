import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import scipy.signal as signal
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, Provider

from framework.listener.concepts import AudioInput
from framework.listener.configs import ListenerConfig

logger = logging.getLogger(__name__)


DropPolicy = Literal["drop_new", "drop_oldest"]


@dataclass
class _Subscriber:
    name: str
    q: "queue.Queue[bytes]"
    drop_policy: DropPolicy
    closed: threading.Event


class MicHub:
    """Single microphone capture fan-out.

    The hub opens the physical microphone exactly once (PyAudio/PortAudio), then
    distributes captured PCM frames (int16 interleaved) to multiple subscribers.

    Design goals:
    - Avoid multi-stream conflicts when both ASR (PTT) and video recording need mic.
    - Keep ASR responsive: subscriber queues never backpressure the capture thread.
    """

    def __init__(
        self,
        *,
        device_index: Optional[int],
        rate: int,
        channels: int,
        frames_per_buffer: int,
        logger: LoggerItf | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger("MicHub")
        self._device_index = device_index
        self._rate = int(rate)
        self._channels = int(channels)
        self._frames_per_buffer = int(frames_per_buffer)

        self._lock = threading.Lock()
        self._subs: dict[str, _Subscriber] = {}

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Lazy init: allow importing the module without PyAudio installed.
        self._pa = None
        self._stream = None

    @property
    def rate(self) -> int:
        return self._rate

    @property
    def channels(self) -> int:
        return self._channels

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        try:
            import pyaudio  # type: ignore
        except Exception as e:
            raise RuntimeError("PyAudio is required for microphone capture") from e

        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        if self._stream is None:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=self._frames_per_buffer,
            )

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="MicHub", daemon=True)
        self._thread.start()
        self._logger.info(
            "MicHub started device_index=%s rate=%s channels=%s fpb=%s",
            self._device_index,
            self._rate,
            self._channels,
            self._frames_per_buffer,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

        try:
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            pass
        self._pa = None

        self._logger.info("MicHub stopped")

    def subscribe(
        self,
        name: str,
        *,
        max_queue: int = 400,
        drop_policy: DropPolicy = "drop_oldest",
    ) -> "MicSubscription":
        self.start()
        sub = _Subscriber(
            name=name,
            q=queue.Queue(maxsize=max(1, int(max_queue))),
            drop_policy=drop_policy,
            closed=threading.Event(),
        )
        with self._lock:
            # Replace existing subscription with the same name.
            old = self._subs.pop(name, None)
            if old is not None:
                old.closed.set()
            self._subs[name] = sub
        return MicSubscription(self, sub)

    def new_audio_input(
        self,
        name: str,
        *,
        max_queue: int = 400,
        drop_policy: DropPolicy = "drop_oldest",
    ) -> AudioInput:
        return MicHubAudioInput(
            hub=self,
            name=name,
            max_queue=max_queue,
            drop_policy=drop_policy,
        )

    def _unsubscribe(self, sub: _Subscriber) -> None:
        with self._lock:
            cur = self._subs.get(sub.name)
            if cur is sub:
                self._subs.pop(sub.name, None)
        sub.closed.set()

    def _loop(self) -> None:
        if self._stream is None:
            return

        while not self._stop.is_set():
            try:
                data = self._stream.read(self._frames_per_buffer, exception_on_overflow=False)
            except Exception as e:
                self._logger.warning("MicHub read failed: %s", e)
                time.sleep(0.01)
                continue

            with self._lock:
                subs = list(self._subs.values())

            for sub in subs:
                if sub.closed.is_set():
                    continue
                try:
                    sub.q.put_nowait(data)
                except queue.Full:
                    if sub.drop_policy == "drop_new":
                        continue
                    # drop_oldest: free one slot then put.
                    try:
                        sub.q.get_nowait()
                        sub.q.task_done()
                    except Exception:
                        continue
                    try:
                        sub.q.put_nowait(data)
                    except Exception:
                        continue


class MicSubscription:
    def __init__(self, hub: MicHub, sub: _Subscriber) -> None:
        self._hub = hub
        self._sub = sub

    @property
    def rate(self) -> int:
        return self._hub.rate

    @property
    def channels(self) -> int:
        return self._hub.channels

    def get(self, timeout: float | None = None) -> bytes:
        return self._sub.q.get(timeout=timeout)

    def get_nowait(self) -> bytes:
        return self._sub.q.get_nowait()

    def task_done(self) -> None:
        self._sub.q.task_done()

    def drain(self) -> None:
        while not self._sub.q.empty():
            try:
                self._sub.q.get_nowait()
                self._sub.q.task_done()
            except Exception:
                break

    def close(self) -> None:
        self._hub._unsubscribe(self._sub)


class MicHubAudioInput(AudioInput):
    """AudioInput adapter backed by a MicHub subscription."""

    def __init__(
        self,
        *,
        hub: MicHub,
        name: str,
        max_queue: int,
        drop_policy: DropPolicy,
        dtype: np.dtype = np.int16,
    ) -> None:
        self._hub = hub
        self._name = name
        self._dtype = dtype
        self._sub = hub.subscribe(name=name, max_queue=max_queue, drop_policy=drop_policy)
        self._started = False
        self._closed = False
        self._buffer = bytearray()

    @property
    def input_id(self) -> str:
        return self._name

    @property
    def rate(self) -> int:
        return self._hub.rate

    @property
    def channels(self) -> int:
        return self._hub.channels

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def start(self) -> None:
        if self._closed:
            raise OSError("MicHubAudioInput already closed")
        self._started = True
        self._sub.drain()
        self._buffer.clear()

    def stop(self) -> None:
        self._started = False
        self._sub.drain()
        self._buffer.clear()

    def closed(self) -> bool:
        return self._closed

    def close(self, error: Optional[Exception] = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._started = False
        self._buffer.clear()
        try:
            self._sub.close()
        except Exception:
            pass

    def _return_zero(self, duration: float) -> np.ndarray:
        frames = int(self.rate * duration)
        return np.zeros(frames * self.channels, dtype=self.dtype)

    def _resample(self, audio_data: np.ndarray, rate: Optional[int]) -> np.ndarray:
        if rate is None or rate == self.rate:
            return audio_data
        number_of_samples = int(len(audio_data) * float(rate) / self.rate)
        resampled = signal.resample(audio_data, number_of_samples)
        return resampled.astype(self.dtype)

    def read(self, *, rate: Optional[int] = None, duration: Optional[float] = None) -> np.ndarray:
        if not self._started:
            raise RuntimeError("MicHubAudioInput is not running")

        if duration is None:
            duration = 0.128

        frames_needed = max(1, int(self.rate * duration))
        bytes_per_frame = 2 * self.channels
        bytes_needed = frames_needed * bytes_per_frame

        deadline = time.time() + max(0.2, float(duration) * 2.0)
        while len(self._buffer) < bytes_needed and not self._closed:
            timeout = max(0.0, deadline - time.time())
            if timeout == 0.0:
                break
            try:
                chunk = self._sub.get(timeout=timeout)
                self._sub.task_done()
                self._buffer.extend(chunk)
            except queue.Empty:
                break
            except Exception:
                break

        if len(self._buffer) < bytes_needed:
            return self._return_zero(duration)

        chunk = bytes(self._buffer[:bytes_needed])
        del self._buffer[:bytes_needed]
        np_data = np.frombuffer(chunk, dtype=self.dtype)
        return self._resample(np_data, rate)


class MicHubProvider(Provider[MicHub]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> MicHub:
        cfg = ListenerConfig().resolve_env()
        audio_cfg = cfg.get_audio_input_config().resolve_env()
        device_index = audio_cfg.get_device_index()
        rate = int(audio_cfg.rate)
        channels = int(audio_cfg.channels)
        frames_per_buffer = int(audio_cfg.chunk_size)

        hub = MicHub(
            device_index=device_index,
            rate=rate,
            channels=channels,
            frames_per_buffer=frames_per_buffer,
            logger=con.get(LoggerItf),
        )
        # Start early so both PTT and recorder can reuse without extra latency.
        hub.start()
        return hub
