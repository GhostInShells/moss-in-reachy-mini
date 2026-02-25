from abc import ABC, abstractmethod
from threading import Thread, Event
from typing import List, Optional

import numpy as np
import pyaudio
from ghoshell_common.contracts import LoggerItf

from moss_in_reachy_mini.listener.concepts.listener import (
    ListenerService,
    ListenerState, ListenerStateName,
    ListenerCallback, RecognitionCallback,
    Recognizer, AudioInput,

)
from moss_in_reachy_mini.listener.callbacks import LoggerCallback
from moss_in_reachy_mini.listener.configs import ListenerConfig
from moss_in_reachy_mini.listener.states import (
    ListeningState, VAD,
    AsleepState, WakenDetector,
    DeafState,
    PdtWaitingState,
    PdtListeningState,
)
from moss_in_reachy_mini.listener.volcengine_bm import VocEngineBigModelASR

__all__ = ['BasicListenerService', 'ListenerServiceImpl']


class BasicListenerService(ListenerService, ABC):

    def __init__(
            self,
            default_state_name: str,
            *,
            logger: LoggerItf,
            callback: Optional[ListenerCallback] = None,
            state_loop_interval: float = 0.1,
    ):
        self._logger = logger
        self._state_loop_interval = state_loop_interval
        self._callback = callback or LoggerCallback(logger)
        self._current_state = self._make_state(default_state_name, None)
        self._next_state: Optional[ListenerState.NextState] = None
        self._bootstrapped: bool = False
        self._shutdown_event = Event()
        self._main_state_loop_thread = Thread(target=self._main_state_loop, daemon=True)

    @abstractmethod
    def audio_input(self) -> AudioInput:
        pass

    @abstractmethod
    def recognizer(self) -> Recognizer:
        pass

    @abstractmethod
    def _make_state(self, state_name: str, buffer: Optional[np.ndarray]) -> ListenerState:
        """
        实例化 state 对象的函数.
        """
        pass

    def set_callback(self, callback: ListenerCallback) -> None:
        self._callback = callback

    def clear_buffer(self) -> None:
        self._current_state.clear_buffer()

    def commit(self) -> None:
        self._current_state.commit()

    def set_vad(self, vad_time: int) -> None:
        self._current_state.set_vad(vad_time)

    def all_states(self) -> List[str]:
        return [
            ListenerStateName.listening.value,
            ListenerStateName.deaf.value,
            ListenerStateName.asleep.value,
            ListenerStateName.pdt_listening.value,
            ListenerStateName.pdt_waiting.value
        ]

    def _set_state(self, state: str, buffer: Optional[np.ndarray], force: bool = False) -> None:
        if force or self._next_state is None:
            # 标记切换的 sate.
            self._next_state = ListenerState.NextState(state, buffer)
            self._logger.info("set next state %s", self._next_state)

    def current_state(self) -> ListenerState:
        return self._current_state

    def _main_state_loop(self) -> None:
        while not self._shutdown_event.wait(0.05):
            self._logger.debug("status check loop start")
            # 判断要不要进行状态变更.
            if self._check_change_state():
                self._logger.info("listener state changed to %s", self._current_state.name())
                continue
            # 判断是否有下一个状态.
            next_state = self._current_state.next()
            if next_state is not None:
                # 不是最优先, 如果用户已经设置了变更, 那个更优先.
                self._set_state(next_state.state_name, next_state.audio_buffer, force=False)

    def _check_change_state(self) -> bool:
        if self._next_state is None:
            return False
        next_state_name = self._next_state.state_name
        buffer = self._next_state.audio_buffer
        try:
            self._logger.info("change listener to next state %s", next_state_name)
            current_state = self._current_state
            # 关闭当前状态. 可能有阻塞时间.
            current_state.close()
            # 创建新的 state.
            new_state = self._make_state(state_name=next_state_name, buffer=buffer)
            new_state.start()
            self._current_state = new_state
            # 成功切换后, 才设置 next state 为空. 同时通知.
            self._callback.on_state_change(new_state.name())
            return True
        except Exception as e:
            self._logger.exception(e)
            self._callback.on_error(f"change state to {next_state_name} failed: {e}")
            return False
        finally:
            self._next_state = None

    def bootstrap(self):
        if self._bootstrapped:
            return
        self._bootstrapped = True
        # 启动默认的 state.
        self._current_state.start()
        self._main_state_loop_thread.start()

    def shutdown(self) -> None:
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        self._main_state_loop_thread.join()

    def set_state(self, state: str) -> None:
        state = str(state)
        if state not in self.all_states():
            raise NotImplementedError(f"state {state} is not supported")
        self._set_state(state, None, True)


class ListenerServiceImpl(BasicListenerService):

    def __init__(
            self,
            config: ListenerConfig,
            *,
            logger: LoggerItf,
            default_state_name: str = "",
            callback: Optional[ListenerCallback] = None,
            audio_input: AudioInput=None,
    ):
        self._config = config.resolve_env()
        self._pa = pyaudio.PyAudio()
        self._audio_input = audio_input or self._config.get_audio_input_config().new_audio_input(pa=self._pa, logger=logger,
                                                                                  dtype=np.int16)
        super().__init__(
            default_state_name or config.default_state_name,
            logger=logger,
            callback=callback,
            state_loop_interval=config.state_loop_interval,
        )

    def audio_input(self) -> AudioInput:
        return self._audio_input

    def recognizer(self) -> Recognizer:
        return self.make_recognizer(self._config, self._logger, self._callback)

    @staticmethod
    def make_recognizer(config: ListenerConfig, logger: LoggerItf, callback: RecognitionCallback) -> Recognizer:
        if config.use_asr == "volcengine_bm_asr":
            return VocEngineBigModelASR(
                config=config.volcengine_bm_asr,
                logger=logger,
                callback=callback,
            )
        else:
            raise NotImplementedError(f"{config.use_asr} is not supported")

    def _get_vad(self) -> Optional[VAD]:
        """
        没有本地的实现. 未来可以重写这个函数.
        """
        return None

    def _get_waken_detector(self) -> Optional[WakenDetector]:
        return None

    def _make_state(self, state_name: str, buffer: Optional[np.ndarray]) -> ListenerState:
        if state_name == ListenerStateName.listening.value:
            return ListeningState(
                recognizer=self.recognizer(),
                audio_input=self.audio_input(),
                callback=self._callback,
                logger=self._logger,
                vad=self._get_vad(),
                on_complete_state=self._config.asr_on_vad_state,
                max_idle_time=self._config.asr_max_idle_time,
                on_max_idle_state=self._config.asr_on_idle_state,
                # todo: max retry 要不要配呢? 没想明白.
            )
        elif state_name == ListenerStateName.asleep.value:
            waken = self._get_waken_detector()
            if waken is None:
                # 等价于聋子.
                return DeafState()
            return AsleepState(
                audio_input=self.audio_input(),
                callback=self._callback,
                logger=self._logger,
                waken_detector=waken,
                frame_duration=None,
                resample_rate=None,
                # 通常都是唤醒后进入聆听.
                next_state_name=ListenerStateName.listening.value,
            )
        elif state_name == ListenerStateName.pdt_listening.value:
            return PdtListeningState(
                recognizer=self.recognizer(),
                audio_input=self.audio_input(),
                callback=self._callback,
                logger=self._logger,
            )
        elif state_name == ListenerStateName.pdt_waiting.value:
            return PdtWaitingState()
        # 剩下的都聋得了.
        return DeafState()
