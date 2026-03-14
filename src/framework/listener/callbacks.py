from typing import TYPE_CHECKING

from ghoshell_common.contracts import LoggerItf

from framework.listener.concepts import ListenerCallback, Recognition

if TYPE_CHECKING:
    from rich.console import Console

__all__ = ['LoggerCallback', 'ConsoleCallback']


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
