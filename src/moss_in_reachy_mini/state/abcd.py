import abc
import asyncio
import time
from typing import Optional

from ghoshell_moss import PyChannel

from framework.abcd.agent_hook import AgentHook


class QuitIdleMove(Exception):
    pass


class MiniStateHook(AgentHook, abc.ABC):
    NAME = ""

    in_switchable = True
    out_switchable = True

    def __init__(self):
        self._run_idle_move_task: Optional[asyncio.Task] = None
        self._idle_move_duration = 0
        self._idle_move_elapsed = 0.1

    @abc.abstractmethod
    async def on_self_enter(self):
        pass

    @abc.abstractmethod
    async def on_self_exit(self):
        pass

    @abc.abstractmethod
    async def _run_idle_move(self):
        pass

    async def run_idle_move(self):
        start = int(time.time())
        try:
            while True:
                await asyncio.sleep(self._idle_move_elapsed)
                now = int(time.time())
                self._idle_move_duration = now - start

                await self._run_idle_move()

        except asyncio.CancelledError:
            raise
        except QuitIdleMove:
            pass
        finally:
            self._idle_move_duration = 0

    async def start_idle_move(self):
        await self.cancel_idle_move()
        self._run_idle_move_task = asyncio.create_task(self.run_idle_move())

    async def cancel_idle_move(self):
        if self._run_idle_move_task is not None and not self._run_idle_move_task.done():
            self._run_idle_move_task.cancel()
            try:
                await self._run_idle_move_task
            except asyncio.CancelledError:
                pass

    async def on_idle(self):
        await self.start_idle_move()

    async def on_responding(self):
        await self.cancel_idle_move()


class InitialState(MiniStateHook):
    NAME = "initial"

    def __init__(self):
        super().__init__()

    async def on_self_enter(self):
        pass

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        pass

    def as_channel(self) -> None:
        return None

