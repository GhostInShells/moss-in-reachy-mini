import asyncio
import logging
import time
from typing import List

from PIL import Image
from ghoshell_common.contracts import Storage, Workspace, LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE, Container
from ghoshell_moss import Message, Base64Image, Text, PyChannel

from framework.abcd.agent_event import VisionAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.camera.camera_worker import CameraWorker
from moss_in_reachy_mini.camera.model import get_closest_position


class Vision:

    def __init__(
            self,
            camera_worker: CameraWorker,
            storage: Storage,
            eventbus: EventBus,
            logger: LoggerItf=None,
    ):
        self.camera_worker = camera_worker
        self.face_recognizer = camera_worker.face_recognizer
        self.vision_storage = storage
        self.eventbus = eventbus
        self.logger = logger or logging.getLogger("Vision")

    async def look(self, about: str = '', fps: int = 1, n: int = 1):
        """
        主动获取机器人摄像头的视觉信息。

        支持连续拍摄多帧图片(建议fps*n不要超过3)，并将图片交给Agent处理

        :param about: 本次 look 操作的原因或备注信息，默认为空字符串
        :param fps: 每秒采集的帧数，默认为 1, 最大为 3
        :param n: 采集的总帧数，默认为 1, 最大为 3
        """
        fps = min(fps, 3)
        n = min(n, 3)

        # 校验fps*n不超过3，超过时优先减fps（fps>1时），否则减n
        while fps * n > 3:
            if fps > 1:
                fps -= 1
            else:
                n -= 1

        # 计算间隔
        interval = 1 / fps
        # 计算总帧数
        total_frames = n

        frames: List[Base64Image] = []
        for i in range(total_frames):
            frame = self.camera_worker.get_latest_frame()
            frames.append(frame.to_base64_image())
            if total_frames == 1:  # 只有一帧,无需等待
                break
            await asyncio.sleep(interval)

        if frames:
            event = VisionAgentEvent(content=about + ' 本次look成功,你只需要说你看到的视觉就可以了,不需要再次调用look', images=frames)
        else:
            event = VisionAgentEvent(content=about + ' 本次look失败,没有获取到视觉信息', images=[])

        event.priority = -1  # 降低优先级
        await self.eventbus.put(event)

    async def enroll(self, name: str):
        """
        用途：通过机器人摄像头主动获取一帧视觉图片，基于该帧获取到的人脸信息，添加到人脸数据库。
        场景：1. 认识新朋友 2. 加强对老朋友的识别
        要求：当需要录入新人但是不知道名字时，需要先向用户询问。

        :param name: 录入的人名，仅支持中文名拼音或英文名
        """
        if not name.isascii():
            raise ValueError("Enrollment name must be Pinyin name or English name")

        frame = self.camera_worker.get_latest_frame()
        if name in self.face_recognizer.get_known_faces():
            # 已认识的用户
            face_positions = [p for p in frame.face_positons if p.name == name]
        else:
            # 未认识的用户
            face_positions = [p for p in frame.face_positons if not p.is_recognized]

        if len(face_positions) == 0:
            raise ValueError("No face found in the frame")

        target = get_closest_position(face_positions)
        img_np_data = self.face_recognizer.crop_face_from_bbox(frame.image, target.bbox)

        storage = self.vision_storage.sub_storage("faces").sub_storage(name)
        img = Image.fromarray(img_np_data)
        storage.put(f"enroll_face_{str(int(time.time()))}", img.tobytes())

        self.face_recognizer.add_known_face(name, target.embedding)
        return f"Successfully enrolled {name}"

    async def unenroll(self, name: str):
        """
        用途：删除人脸数据库中的一个用户。
        场景：仅用户主动要求删除.
        :param name: 录入的人名
        """
        if not name.isascii():
            raise ValueError("Enrollment name must be Pinyin name or English name")

        if name not in self.face_recognizer.get_known_faces():
            raise ValueError(f"User {name} is not known")

        self.face_recognizer.remove_known_face(name, save=True)
        return f"Successfully unenrolled {name}"

    async def context_messages(self):
        msg = Message.new(role="system", name="__reachy_mini_vision__")
        frame = self.camera_worker.get_latest_frame()
        if frame.image is not None:
            msg.with_content(
                Text(text="This image is what you see")
            ).with_content(
                frame.to_base64_image()
            )
        else:
            msg.with_content(
                Text(text="No vision available")
            )

        face_known_user_msg = Message.new(role="system", name="__reachy_mini_face_known_user__").with_content(
            Text(text=f"These are face known users: {', '.join(self.face_recognizer.get_known_faces())}"),
        )

        return [msg, face_known_user_msg]

    def as_channel(self):
        chan = PyChannel(name="vision", description="use camera to look", blocking=True)
        chan.build.command()(self.look)
        chan.build.context_messages(self.context_messages)
        chan.build.command()(self.enroll)
        chan.build.command()(self.unenroll)
        return chan


class VisionProvider(Provider[Vision]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        camera_worker = con.force_fetch(CameraWorker)
        ws = con.force_fetch(Workspace)
        vision_storage = ws.runtime().sub_storage("vision")
        eventbus = con.force_fetch(EventBus)
        logger = con.get(LoggerItf)
        return Vision(camera_worker, vision_storage, eventbus=eventbus, logger=logger)
