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


class AsleepStateProvider(Provider[AsleepState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> AsleepState:
        mini = con.force_fetch(ReachyMini)
        return AsleepState(mini=mini)
