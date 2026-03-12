import asyncio

from ghoshell_common.contracts import Storage, Workspace
from ghoshell_container import Provider, IoCContainer, INSTANCE
from ghoshell_moss import CommandToken, ChannelCtx, MOSSShell, CommandStackResult, PyCommand, CommandTask
from ghoshell_moss.core.concepts.command import CommandTaskResult


class CtmlRepo:
    """
    当Shell注册如下两个函数作为command时，可以支持对话示教保存学习的ctml作为既有技能，并且支持执行已保存的ctml
    """
    def __init__(self, storage: Storage):
        self._storage = storage

    async def save_ctml(self, ctml__, name: str):
        """
        Save CTML commands as text to storage.
        :param ctml__: Nested CTML commands to be executed as a synchronized group.
               The commands will be parsed as sub-tasks and managed by the wait primitive.
        :param name: ctml name
        """
        ctml_text = ""

        async for c in ctml__:
            if not isinstance(c, CommandToken):
                continue

            ctml_text += c.content

        self._storage.put(f"{name}.ctml", ctml_text.encode())

    def list_ctml_names(self):
        ctml_filenames = self._storage.dir("", recursive=False, patten="*.ctml")
        ctml_names = [f.split(".")[0] for f in ctml_filenames]
        return ctml_names

    # 使用方式
    # chan.build.command(doc=ctml_repo.execute_ctml_docstring)(ctml_repo.execute_ctml)
    def execute_ctml_docstring(self):
        return f"""
Execute saved CTML by name.

name list: {self.list_ctml_names()}

:param name: The name of the CTML command to execute.
"""

    async def execute_ctml(self, name: str):
        ctml_text = self._storage.get(f"{name}.ctml").decode()
        shell = ChannelCtx.get_contract(MOSSShell)

        return CommandStackResult(
            shell.parse_text_to_tasks(text=ctml_text),
        )


class CtmlRepoProvider(Provider[CtmlRepo]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        ws = con.force_fetch(Workspace)
        storage = ws.runtime().sub_storage("ctml_repo")
        return CtmlRepo(storage)
