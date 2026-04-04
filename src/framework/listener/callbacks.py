from typing import TYPE_CHECKING

import numpy as np
from ghoshell_common.contracts import LoggerItf

from framework.listener.concepts import ListenerCallback, Recognition

if TYPE_CHECKING:
    from rich.console import Console

__all__ = ['LoggerCallback', 'ConsoleCallback', 'AsyncLoggerCallback', 'AsyncConsoleCallback']


class LoggerCallback(ListenerCallback):

    def __init__(self, logger: LoggerItf):
        self.logger = logger

    def on_waken(self) -> None:
        self.logger.info('[LoggerCallback] Waken!')

    def on_vad(self, recognition: Recognition) -> None:
        self.logger.info('[LoggerCallback] VAD on %s', recognition)

    def on_recognition(self, result: Recognition) -> None:
        self.logger.info('[LoggerCallback] recognition result: %s', result)

    def on_error(self, error: str) -> None:
        self.logger.error('[LoggerCallback] error: %s', error)

    def on_state_change(self, state: str) -> None:
        self.logger.info('[LoggerCallback] on change state %s', state)


class ConsoleCallback(ListenerCallback):

    def __init__(self, console: "Console"):
        self.console = console

    def on_waken(self) -> None:
        self.console.print("[Console] Waken!")

    def on_vad(self, recognition: Recognition) -> None:
        self.console.print("[Console] VAD!")

    def on_state_change(self, state: str) -> None:
        self.console.print("[Console] change state to %s" % state)

    def on_recognition(self, result: Recognition) -> None:
        self.console.print("[asr] id: %s  | seq: %s | is_last: %s" % (result.batch_id, result.seq, result.is_last))
        self.console.print("[asr] text: %s" % result.text)

    def on_error(self, error: str) -> None:
        self.console.print("[Console] error %s" % error)


# 异步回调类
class AsyncLoggerCallback:
    """
    异步日志回调。
    实现 AsyncRecognitionCallback 和 AsyncListenerCallback 接口。
    """

    def __init__(self, logger: LoggerItf):
        self.logger = logger

    async def on_recognition(self, result: Recognition) -> None:
        self.logger.info('[AsyncLoggerCallback] recognition result: %s', result)

    async def on_error(self, error: str) -> None:
        self.logger.error('[AsyncLoggerCallback] error: %s', error)

    async def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
        # 默认先不实现
        return

    async def on_waken(self) -> None:
        self.logger.info('[AsyncLoggerCallback] Waken!')

    async def on_state_change(self, state: str) -> None:
        self.logger.info('[AsyncLoggerCallback] on change state %s', state)


class AsyncConsoleCallback:
    """
    异步控制台回调。
    """

    def __init__(self, console: "Console"):
        self.console = console

    async def on_recognition(self, result: Recognition) -> None:
        self.console.print("[async asr] id: %s  | seq: %s | is_last: %s" % (result.batch_id, result.seq, result.is_last))
        self.console.print("[async asr] text: %s" % result.text)

    async def on_error(self, error: str) -> None:
        self.console.print("[AsyncConsole] error %s" % error)

    async def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
        # 默认先不实现
        return

    async def on_waken(self) -> None:
        self.console.print("[AsyncConsole] Waken!")

    async def on_state_change(self, state: str) -> None:
        self.console.print("[AsyncConsole] change state to %s" % state)
