from typing import Optional

from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer
from ghoshell_moss import Channel, ChannelRuntime, PyChannel


class VolcWebSearch(Channel):
    def __init__(self, name: str, description: str, api_key: str):
        self._name = name
        self._description = description
        self._api_key = api_key

        self._runtime: Optional[ChannelRuntime] = None

    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return uuid()

    def description(self) -> str:
        return self._description

    async def websearch(self, query: str, count: int=10):
        pass

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime is not None and self._runtime.is_available():
            return self._runtime

        chan = PyChannel(name=self.name(), description=self.description())
        chan.build.command()(self.websearch)

        self._runtime = chan.bootstrap(container=container)
        return self._runtime

