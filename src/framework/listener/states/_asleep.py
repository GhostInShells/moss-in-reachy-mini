from queue import Queue, Empty
from threading import Event
from threading import Thread
from typing import Optional, Callable

import numpy as np
from ghoshell_common.contracts import LoggerItf

from framework.listener.concepts import (
    ListenerState, ListenerStateName, AudioInput,
    AudioInputLoop, ListenerCallback,
)

WakenDetector = Callable[[np.ndarray], bool]


class AsleepState(ListenerState):
    """
    等待唤醒的状态.
    """

    def __init__(
            self,
            *,
            audio_input: AudioInput,
            callback: ListenerCallback,
            logger: LoggerItf,
            waken_detector: Callable[[np.ndarray], bool],
            frame_duration: Optional[float] = None,
            resample_rate: Optional[int] = None,
            next_state_name: str = ListenerStateName.listening,
    ):
        self._audio_input = audio_input
        self._logger = logger
        self._waken_detector = waken_detector
        self._callback = callback
        self._closed_event = Event()
        self._audio_queue: Queue[np.ndarray] = Queue()
        self._frame_duration = frame_duration
        self._resample_rate = resample_rate
        self._next_state: Optional[ListenerState] = None
        self._next_state_name = next_state_name

    def name(self) -> ListenerStateName:
        return ListenerStateName.asleep

    def clear_buffer(self) -> None:
        """
        不需要做任何事情.
        """
        pass

    def commit(self) -> None:
        """
        不需要做任何事情.
        """
        pass

    def set_vad(self, vad_time: int) -> None:
        pass

    def next(self) -> Optional[ListenerState.NextState]:
        return self._next_state

    def _awaken_detecting_loop(self) -> None:
        while not self._closed_event.is_set():
            try:
                audio_data = self._audio_queue.get(timeout=0.5)
                if self._waken_detector(audio_data):
                    self._logger.info(f"Waken signal detected")
                    self._next_state = ListenerState.NextState(
                        state_name=self._next_state_name,
                        audio_buffer=None,
                    )
                    self._callback.on_waken(),
                    # 退出循环.
                    break
            except Empty:
                continue
            except Exception as e:
                self._logger.exception(e)
                # 要求重启.
                self._next_state = ListenerState.NextState(state_name=self.name(), audio_buffer=None)
        self._closed_event.set()

    def start(self) -> None:
        audio_loop = AudioInputLoop(
            self._audio_queue.put,
            self._audio_input,
            stop_event=self._closed_event,
            resample_rate=self._resample_rate,
            frame=self._frame_duration,
        )
        audio_loop.start()
        thread = Thread(target=self._awaken_detecting_loop, daemon=True)
        thread.start()

    def close(self) -> None:
        self._closed_event.set()
