from framework.agent.agent_hook import BaseAgentHook


class InitialState(BaseAgentHook):
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

