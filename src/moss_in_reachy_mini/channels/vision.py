import asyncio
from collections import deque

from reachy_mini import ReachyMini


class Vision:

    def __init__(self, mini: ReachyMini):
        self.mini = mini
        self.mini.media.get_frame()

        self._vision_max = 5
        self._vision_elapsed = 3
        self._dq = deque(maxlen=self._vision_max)
        self._dq_lock = asyncio.Lock()

        self._closed = asyncio.Event()

    async def context_messages(self):
        pass

    async def _run_loop(self):
        if not self._closed.is_set():
            pass

    async def start(self):
        pass

    async def close(self):
        pass