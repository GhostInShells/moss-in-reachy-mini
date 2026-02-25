from typing import Union

from moss_in_reachy_mini.listener.concepts.listener import (
    ListenerState, ListenerStateName
)


class PdtWaitingState(ListenerState):
    """
    啥都不干.
    """

    def __init__(self):
        self.next_state = None

    def name(self) -> ListenerStateName:
        return ListenerStateName.pdt_waiting

    def clear_buffer(self) -> None:
        return

    def commit(self) -> None:
        self.next_state = ListenerState.NextState(
            state_name=ListenerStateName.pdt_listening.value,
            audio_buffer=None,
        )
        return

    def set_vad(self, vad_time: int) -> None:
        return

    def next(self) -> Union[ListenerState.NextState, None]:
        return self.next_state

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None
