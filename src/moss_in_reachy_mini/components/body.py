from typing import List

from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss import Command, PyCommand
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove
from reachy_mini.utils import create_head_pose
from reachy_mini_dances_library import DanceMove
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES

from moss_in_reachy_mini.components.head_tracker import HeadTracker
from moss_in_reachy_mini.moves.head_move import HeadMove
from moss_in_reachy_mini.utils import load_emotions


class Body:
    def __init__(self, mini: ReachyMini, container: IoCContainer):
        self.mini = mini

        self._emotions_storage, self._emotions = load_emotions(container)

    async def dance(self, name: str):
        if not AVAILABLE_MOVES.get(name):
            raise ValueError(f'{name} is not a valid dance')
        await self.mini.async_play_move(DanceMove(name))
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
        ))

    def dance_docstring(self):
        dance_docstrings = []
        for name, move in AVAILABLE_MOVES.items():
            func, params, meta = move
            dance_docstrings.append(f"name: {name} description: {meta.get("description", "")}")
        return f"Dance can be chosen in \n{"\n".join(dance_docstrings)}"

    async def emotion(self, name: str):
        params = self._emotions.get(name)
        if not params:
            raise ValueError(f"{name} is not a valid emotion")

        sound_path = None
        # if play_sound:
        #     sound_path = f"{self._emotions_storage.abspath()}/{name}.wav"

        await self.mini.async_play_move(RecordedMove(move=params, sound_path=sound_path))
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
        ))

    def emotion_docstring(self):
        emotion_docstrings = []
        for name, params in self._emotions.items():
            emotion_docstrings.append(f"name: {name} description: {params.get('description', '')}")
        return f"Emotion can be only chosen in \n{"\n".join(emotion_docstrings)}\nDo not use unknown emotion name"

    def as_commands(self) -> List[Command]:
        return [
            PyCommand(
                func=self.dance,
                name="dance",
                doc=self.dance_docstring(),
            ),
            PyCommand(
                func=self.emotion,
                name="emotion",
                doc=self.emotion_docstring(),
            )
        ]

class BodyProvider(Provider[Body]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        return Body(mini, con)
