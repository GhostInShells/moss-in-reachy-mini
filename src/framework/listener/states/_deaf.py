from typing import Union

from framework.listener.concepts import (
    ListenerState, ListenerStateName
)


class DeafState(ListenerState):
    """
    啥都不干.
    """

    def name(self) -> ListenerStateName:
        return ListenerStateName.deaf

    def clear_buffer(self) -> None:
        return

    def commit(self) -> None:
        return

    def set_vad(self, vad_time: int) -> None:
        return

    def next(self) -> Union[ListenerState.NextState, None]:
        return None

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None
