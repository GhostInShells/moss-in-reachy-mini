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
    consciousness_memory_md: str = Field(default="consciousness_memory.md", description="conscious memory file")


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
    async def refresh(self, text__: str, part: str):
        """
        更新指定part相关记忆
        :param text__: 内容
        :param part: 可选项有`personality`,`behavior_preference`,`mood_base`,`autobiographical_memory`,`summary_memory`,`consciousness_memory`
        """
        self.storage.put(f"{part}.md", text__.encode())

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
        personality_contents = []
        if personality:
            personality_contents.append(Text(text=f"personality: {personality}"))
        if behavior_preference:
            personality_contents.append(Text(text=f"behavior_preference: {behavior_preference}"))
        if mood_base:
            personality_contents.append(Text(text=f"mood_base: {mood_base}"))
        msgs.append(Message.new(role="system", name="__personality__").with_content(
            *personality_contents
        ))

        autobiographical_memory = await self.read_md(self._meta_config.autobiographical_memory_md)
        summary_memory = await self.read_md(self._meta_config.summary_memory_md)
        consciousness_memory = await self.read_md(self._meta_config.consciousness_memory_md)
        memory_contents = []
        if autobiographical_memory:
            memory_contents.append(Text(text=f"autobiographical_memory: {autobiographical_memory}"))
        if summary_memory:
            memory_contents.append(Text(text=f"summary_memory: {summary_memory}"))
        if consciousness_memory:
            memory_contents.append(Text(text=f"consciousness_memory: {consciousness_memory}"))
        msgs.append(Message.new(role="system", name="__memory__").with_content(
            *memory_contents
        ))

        msgs.append(Message.new(role="system", name="__memory_settings__").with_content(
            Text(text=f"Config: {self._meta_config.model_dump_json()}"),
        ))
        return msgs

    def as_channel(self, read_only: bool = False) -> PyChannel:
        memory = PyChannel(
            name="memory",
            description="你的记忆存储和读取通道",
        )

        if not read_only:
            memory.build.command()(self.refresh)
        memory.build.context_messages(self.context_messages)

        return memory


