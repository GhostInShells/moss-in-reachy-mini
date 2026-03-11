import asyncio

from ghoshell_common.contracts import Storage, YamlConfig, WorkspaceConfigs, FileStorage
from ghoshell_moss import Message, Text, PyChannel
from pydantic import Field


class MetaConfig(YamlConfig):
    relative_path = ".memory_meta.yaml"

    # 人格
    personality_md: str = Field(default="personality.md", description="personality file")
    behavior_preference_md: str = Field(default="behavior_preference.md", description="behavior preference file")
    mood_base_md: str = Field(default="mood_base.md", description="mood base file")

    # 记忆
    autobiographical_memory_md: str = Field(default="autobiographical_memory.md", description="autobiographical memory file")
    summary_memory_md: str = Field(default="summary_memory.md", description="summary memory file")


class StorageMemory:
    def __init__(self, storage: Storage | FileStorage):
        self.storage = storage

        self._configs = WorkspaceConfigs(storage)
        self._meta_config = self._configs.get_or_create(MetaConfig())

        # lifecycle
        self._started = asyncio.Event()

    @property
    def meta_config(self) -> MetaConfig:
        return self._meta_config

    # ==== Model Context ====
    async def refresh_personality(self, text__: str):
        """
        当用户产生诉求或者你觉得必要的时候，基于现在的人格来完善并赋予你自己新的人格

        Markdown格式写入
        一级标题：基础人格
        二级标题：
            身份设定：名字、角色、年龄感、职业 / 身份等
            性格特质：...
            说话风格：...
            互动风格：...
            价值观倾向：重视效率、重视情感、重视事实、重视关系等
        """
        self.storage.put(self._meta_config.personality_md, text__.encode())

    async def refresh_behavior_preference(self, text__: str):
        """
        你可以自主要求来更新你自己的行为偏好

        Markdown格式写入
        一级标题：行为偏好
        二级标题：
            对话启动偏好：主动 / 被动
            回应长度偏好：超短 / 正常 / 详细
            情绪表达强度：弱 / 中 / 强
            冲突处理风格：回避 / 安抚 / 讲道理 / 直接
        """
        self.storage.put(self._meta_config.behavior_preference_md, text__.encode())

    async def refresh_mood_base(self, text__: str):
        """
        你可以自主要求来更新你自己的情绪基底

        Markdown格式写入
        一级标题：情绪基底
        二级标题：
            基础情绪：平静、开心、温和、疲惫等
            情绪稳定性：易波动 / 很稳定
            情绪恢复速度：快 / 中 / 慢
        """
        self.storage.put(self._meta_config.mood_base_md, text__.encode())

    async def refresh_autobiographical_memory(self, text__: str):
        """
        当发生一些“经历”后，由用户要求或你觉得必要的时候更新你自己的自传记忆

        Markdown格式写入
        一级标题：自传记忆
        二级标题：
            自己的 “经历”：重要事件、时间、地点、感受
            关键决策与结果
            成长 / 变化轨迹
        """
        self.storage.put(self._meta_config.autobiographical_memory_md, text__.encode())

    async def refresh_summary_memory(self, text__: str):
        """
        更新你自己的交互历史摘要，需要和之前的交互历史摘要融合提炼，避免完全覆盖掉之前的摘要

        Markdown格式写入
        一级标题：交互历史摘要
        二级标题：
            重要对话摘要（非原始聊天记录，允许上千token写入）
            共同经历事件（允许上千token写入）
            对方对 Agent 的评价 / 态度（允许上千token写入）
        """
        self.storage.put(self._meta_config.summary_memory_md, text__.encode())

    async def read_md(self, md: str) -> str:
        if not self.storage.exists(md):
            return ""
        res = self.storage.get(md)
        return res.decode()

    async def context_messages(self):
        msgs = []
        personality = await self.read_md(self._meta_config.personality_md)
        behavior_preference = await self.read_md(self._meta_config.behavior_preference_md)
        mood_base = await self.read_md(self._meta_config.mood_base_md)
        msgs.append(Message.new(role="system", name="__personality__").with_content(
            Text(text=personality or "Empty"),
            Text(text=behavior_preference or "Empty"),
            Text(text=mood_base or "Empty"),
        ))

        autobiographical_memory = await self.read_md(self._meta_config.autobiographical_memory_md)
        summary_memory = await self.read_md(self._meta_config.summary_memory_md)
        msgs.append(Message.new(role="system", name="__memory__").with_content(
            Text(text=autobiographical_memory or "Empty"),
            Text(text=summary_memory or "Empty"),
        ))

        msgs.append(Message.new(role="system", name="__memory_settings__").with_content(
            Text(text=f"Config: {self._meta_config.model_dump_json()}"),
        ))
        return msgs

    def as_channel(self) -> PyChannel:
        memory = PyChannel(
            name="memory",
            description="refresh相关的command请放置在你的回答结束末尾调用，并且请提前告知用户你要更新你自己的人格和记忆了",
        )

        memory.build.command()(self.refresh_personality)
        memory.build.command()(self.refresh_behavior_preference)
        memory.build.command()(self.refresh_mood_base)
        memory.build.command()(self.refresh_autobiographical_memory)
        memory.build.command()(self.refresh_summary_memory)
        memory.build.context_messages(self.context_messages)

        return memory


