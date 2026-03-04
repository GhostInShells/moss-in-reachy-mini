from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import Text, Message, PyChannel
from reachy_mini import ReachyMini

from moss_in_reachy_mini.state.abcd import MiniStateHook


class AsleepState(MiniStateHook):

    NAME = "asleep"

    def __init__(self, mini: ReachyMini):
        super().__init__()
        self.mini = mini

    async def on_self_enter(self):
        self.mini.set_target_body_yaw(0.0)
        self.mini.goto_sleep()
        self.mini.disable_motors()

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        pass

    async def context_messages(self):
        msg = Message.new(role="system").with_content(
            Text(text="你现在处于asleep状态，你必须得先切换到其他状态才能继续和用户进行交互，比如waken"),
        )
        return [msg]

    def as_channel(self):
        chan = PyChannel(name=AsleepState.NAME, description=f"current state is asleep", block=True)
        chan.build.with_context_messages(self.context_messages)
        return chan


class AsleepStateProvider(Provider[AsleepState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> AsleepState:
        mini = con.force_fetch(ReachyMini)
        return AsleepState(mini=mini)
