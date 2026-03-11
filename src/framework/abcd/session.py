import abc
from typing import List

from ghoshell_moss import Message


class Session(abc.ABC):

    @abc.abstractmethod
    async def save_turn(self, inputs: List[Message], outputs: List[Message]):
        pass

    @abc.abstractmethod
    async def get_session_history(self) -> List[Message]:
        pass

    @abc.abstractmethod
    async def start(self):
        pass

    @abc.abstractmethod
    async def close(self):
        pass