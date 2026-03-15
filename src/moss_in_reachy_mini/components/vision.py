import asyncio
import logging
from typing import List

from PIL import Image
from ghoshell_common.contracts import Storage, Workspace, LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE, Container
from ghoshell_moss import Message, Base64Image, Text, PyChannel

from framework.abcd.agent import Agent
from framework.abcd.agent_event import VisionAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.camera.camera_worker import CameraWorker


class Vision:

    def __init__(self, camera_worker: CameraWorker, storage: Storage, container: IoCContainer=None):
        self.camera_worker = camera_worker
        self.face_recognizer = camera_worker.face_recognizer
        self.vision_storage = storage
        self._container = Container(parent=container)
        self.logger = self._container.get(LoggerItf) or logging.getLogger("Vision")

    async def look(self, about: str = '', fps: int = 1, n: int = 1):
        """
        主动获取机器人摄像头的视觉信息。

        支持连续拍摄多帧图片(建议fps*n不要超过3)，并将图片交给

        :param about: 本次 look 操作的原因或备注信息，默认为空字符串
        :param fps: 每秒采集的帧数，默认为 1, 最大为 3
        :param n: 采集的总帧数，默认为 1, 最大为 3
        """
        eventbus = self._container.force_fetch(EventBus)

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
        await eventbus.put(event)

    async def context_messages(self):
        msg = Message.new(role="system", name="__reachy_mini_vision__")
        frame = self.camera_worker.get_latest_frame()
        if frame.image is not None:
            # 告诉 LLM 图上标注了哪些已识别的人
            recognized_names = [
                pos.name for pos in frame.face_positons
                if pos.is_recognized and pos.name
            ]
            if recognized_names:
                text = (
                    f"This image is what you see. "
                    f"已识别的用户（图中已标注绿框和名字）: {', '.join(recognized_names)}。"
                    f"请用他们的名字来称呼他们。"
                )
            else:
                text = "This image is what you see"
            msg.with_content(
                Text(text=text)
            ).with_content(
                frame.to_base64_image()
            )

        else:
            msg.with_content(
                Text(text="No vision available")
            )
        return [msg]

    def as_channel(self):
        chan = PyChannel(name="vision", description="use camera to look", blocking=True)
        chan.build.command()(self.look)
        chan.build.context_messages(self.context_messages)
        return chan


class VisionProvider(Provider[Vision]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        camera_worker = con.force_fetch(CameraWorker)
        ws = con.force_fetch(Workspace)
        vision_storage = ws.runtime().sub_storage("vision")
        return Vision(camera_worker, vision_storage, container=con)
