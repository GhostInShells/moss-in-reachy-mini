import numpy as np
from numpy import typing as npt
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import time_trajectory, linear_pose_interpolation


class HeadMove(Move):
    def __init__(
            self,
            start_pose: npt.NDArray[np.float64],
            target_pose: npt.NDArray[np.float64],
            duration: float = 0.5,
    ):
        self.start_pose = start_pose
        self.target_pose = target_pose
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None
    ]:
        interp_time = time_trajectory(t / self.duration)
        interp_head_pose = linear_pose_interpolation(
            self.start_pose, self.target_pose, interp_time
        )
        return interp_head_pose, None, None