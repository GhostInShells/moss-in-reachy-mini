import abc
from typing import List

from ghoshell_moss import Message


class Memory(abc.ABC):

    @abc.abstractmethod
    async def save_turn(self, session_id: str, inputs: List[Message], outputs: List[Message]) -> str:
        pass

    @abc.abstractmethod
    async def get_session_history(self, session_id: str = "") -> List[Message]:
        pass
