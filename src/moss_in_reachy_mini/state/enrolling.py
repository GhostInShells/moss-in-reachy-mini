import asyncio
import logging
from pathlib import Path
from typing import Optional

import cv2
from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import IoCContainer, Provider
from reachy_mini import ReachyMini

from framework.abcd.agent_event import CTMLAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.camera.camera_worker import CameraWorker
from moss_in_reachy_mini.camera.face_recognizer import FaceRecognizer
from moss_in_reachy_mini.components.head import Head
from moss_in_reachy_mini.state.abcd import MiniStateHook


class EnrollingState(MiniStateHook):
    """人脸注册状态"""

    NAME = "enrolling"
    in_switchable = False  # 只能通过 start_face_registration 进入，不能直接 switch_state
    out_switchable = True

    def __init__(
        self,
        mini: ReachyMini,
        camera_worker: CameraWorker,
        face_recognizer: FaceRecognizer,
        head: Head,
        eventbus: EventBus,
        workspace: Workspace,
        logger: LoggerItf = None,
    ):
        super().__init__()
        self.mini = mini
        self.camera_worker = camera_worker
        self.face_recognizer = face_recognizer
        self.head = head
        self.logger = logger or logging.getLogger("EnrollingState")
        self.eventbus = eventbus
        self.workspace = workspace
        self.target_name: Optional[str] = None

        self._max_retries = 2
        self._pose_wait_seconds = 8  # 拍照前等待用户调整姿势（需覆盖流式 TTS 播报时间 + 用户反应时间）
        self._face_detect_attempts = 5  # 拍照时等待检测到人脸的最大尝试次数
        self._verify_attempts = 3  # 验证识别时的最大尝试次数
        self._registration_task: Optional[asyncio.Task] = None
        self._cancelled = False

    async def on_self_enter(self):
        self._cancelled = False
        self.mini.enable_motors()
        self.head._idle_suppressed = True
        await self.head.stop_tracking_face()
        await self.head.reset()
        self._registration_task = asyncio.create_task(self._start_registration_process())

    async def on_self_exit(self):
        self._cancelled = True
        if self._registration_task and not self._registration_task.done():
            self._registration_task.cancel()
            try:
                await self._registration_task
            except asyncio.CancelledError:
                pass
        self._registration_task = None
        self.target_name = None
        self.head._idle_suppressed = False

    async def _run_idle_move(self):
        pass

    async def _start_registration_process(self):
        """启动人脸注册流程"""
        try:
            await self._do_registration()
        except asyncio.CancelledError:
            self.logger.info("Registration process cancelled")
            raise
        except Exception:
            self.logger.exception("Registration process failed with unexpected error")
            await self._speak("录入过程出现异常，返回唤醒模式。")
            await asyncio.sleep(3)
            await self._return_to_waken_state()

    async def _do_registration(self):
        if not self.target_name:
            await self._speak("抱歉，未获取到用户名称，返回唤醒模式。")
            await self._return_to_waken_state()
            return

        is_update = self.target_name in self.face_recognizer.known_faces
        if is_update:
            await self._speak(f"好的，{self.target_name}，开始更新你的人脸数据。")
        else:
            await self._speak(f"好的，开始为{self.target_name}录入人脸。")
        await asyncio.sleep(3)

        for attempt in range(1, self._max_retries + 1):
            success = await self._capture_user_images()
            if not success:
                if attempt < self._max_retries:
                    await self._speak("拍照失败，我们重新来一次。")
                    await asyncio.sleep(3)
                    continue
                await self._speak("多次拍照失败，返回唤醒模式。")
                await self._return_to_waken_state()
                return

            await self._speak("照片拍好了，正在学习你的样子。")
            await asyncio.sleep(2)

            success = await self._train_face()
            if not success:
                if attempt < self._max_retries:
                    await self._speak("学习失败，我们重新拍一次。")
                    await asyncio.sleep(3)
                    continue
                await self._speak("多次学习失败，返回唤醒模式。")
                await self._return_to_waken_state()
                return

            await self._speak("学习完成，让我看看能不能认出你。请面向我。")
            await asyncio.sleep(4)

            recognized = await self._verify_recognition()
            if recognized:
                if is_update:
                    await self._speak(f"好的，{self.target_name}，人脸数据已更新。")
                else:
                    await self._speak(f"太好了，{self.target_name}，我记住你了！")
                await self._return_to_waken_and_track()
                return

            if attempt < self._max_retries:
                await self._speak("还没认出来，我们重新录入一次。")
                await asyncio.sleep(3)
            else:
                await self._speak("多次验证未通过，返回唤醒模式。")

        await self._return_to_waken_state()

    async def _capture_user_images(self) -> bool:
        """捕获用户正面、左侧、右侧图像，每张拍照前校验画面中有人脸"""
        try:
            faces_dir = self.workspace.runtime().sub_storage("vision").sub_storage("faces")
            user_dir = Path(faces_dir.abspath()) / self.target_name
            user_dir.mkdir(parents=True, exist_ok=True)

            steps = [
                ("请正面看着我。", "front.jpg"),
                ("请把头稍微转向左边。", "left_profile.jpg"),
                ("请把头稍微转向右边。", "right_profile.jpg"),
            ]

            for prompt, image_name in steps:
                await self._speak(prompt)
                await asyncio.sleep(self._pose_wait_seconds)

                frame = await self._wait_for_face()
                if frame is None:
                    await self._speak("没有检测到人脸，请确保面部在我的视野中。")
                    await asyncio.sleep(3)
                    # 再尝试一次
                    frame = await self._wait_for_face()
                    if frame is None:
                        return False

                image_path = user_dir / image_name
                cv2.imwrite(str(image_path), frame)  # raw_image 已是 BGR，直接保存
                self.logger.info("Saved image: %s", image_path)

            return True
        except Exception:
            self.logger.exception("Failed to capture user images")
            return False

    async def _wait_for_face(self):
        """多次尝试获取包含人脸的帧，返回原始图像（BGR，无标注）或 None"""
        for _ in range(self._face_detect_attempts):
            frame = self.camera_worker.get_latest_frame()
            if frame.raw_image is not None and len(frame.face_positons) > 0:
                return frame.raw_image
            await asyncio.sleep(0.5)
        return None

    async def _train_face(self) -> bool:
        """训练人脸"""
        try:
            from moss_in_reachy_mini.scripts.train_face import from_storage

            faces_storage = self.workspace.runtime().sub_storage("vision").sub_storage("faces")
            from_storage(self.face_recognizer, faces_storage)
            self.logger.info("Face training completed successfully")
            return True
        except Exception:
            self.logger.exception("Face training failed")
            return False

    async def _verify_recognition(self) -> bool:
        """验证人脸识别，多帧尝试提高鲁棒性"""
        for _ in range(self._verify_attempts):
            await asyncio.sleep(1)
            frame = self.camera_worker.get_latest_frame()
            if frame.raw_image is not None:
                positions = self.face_recognizer.get_face_positions(frame.raw_image)
                for position in positions:
                    if position.name == self.target_name:
                        return True
        return False

    async def _speak(self, text: str):
        """通过 <say> CTML 标签直接触发 TTS，不经过 LLM"""
        if self._cancelled:
            return
        await self.eventbus.put(CTMLAgentEvent(ctml=f"<say>{text}</say>"))

    async def _return_to_waken_state(self):
        """返回唤醒状态"""
        if self._cancelled:
            return
        await self.eventbus.put(CTMLAgentEvent(ctml='<reachy_mini:switch_state state_name="waken" />'))

    async def _return_to_waken_and_track(self):
        """返回唤醒状态并自动开始追踪刚录入的用户"""
        if self._cancelled:
            return
        await self.eventbus.put(CTMLAgentEvent(ctml=(
            '<reachy_mini:switch_state state_name="waken" />'
            f'<reachy_mini:start_tracking_face name="{self.target_name}"/>'
        )))


class EnrollingStateProvider(Provider[EnrollingState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> EnrollingState:
        mini = con.force_fetch(ReachyMini)
        camera_worker = con.force_fetch(CameraWorker)
        face_recognizer = camera_worker.face_recognizer
        head = con.force_fetch(Head)
        eventbus = con.force_fetch(EventBus)
        workspace = con.force_fetch(Workspace)
        logger = con.get(logging.Logger)

        return EnrollingState(
            mini=mini,
            camera_worker=camera_worker,
            face_recognizer=face_recognizer,
            head=head,
            eventbus=eventbus,
            workspace=workspace,
            logger=logger,
        )
