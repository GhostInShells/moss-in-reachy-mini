import asyncio
import json
import time
import uuid
from typing import List, Optional

from ghoshell_common.contracts import Storage, Workspace, YamlConfig, WorkspaceConfigs, FileStorage
from ghoshell_container import Container, IoCContainer
from ghoshell_moss import Message, Text, Addition, PyChannel
from pydantic import BaseModel, Field

from framework.abcd.memory import Memory


class Session(BaseModel):
    messages: List[Message] = Field(...)


class Turn(BaseModel):
    turn_id: str = Field(default="", description="turn id")  # 用来关联同一轮对话的input和output


class TurnAddition(Addition, Turn):
    @classmethod
    def keyword(cls) -> str:
        return "session_turn"


class MetaConfig(YamlConfig):
    relative_path = ".meta.yaml"

    current_session_id: str = Field(default="", description="current session id")
    turn_rounds: int = Field(default=10, description="turn rounds")
    max_tokens: int = Field(default=-1, description="max tokens")

    # 人格
    personality_md: str = Field(default=".personality.md", description="personality file")
    behavior_preference_md: str = Field(default=".behavior_preference.md", description="behavior preference file")
    mood_base_md: str = Field(default=".mood_base.md", description="mood base file")

    # 记忆
    autobiographical_memory_md: str = Field(default=".autobiographical_memory.md", description="autobiographical memory file")
    summary_memory_md: str = Field(default=".summary_memory.md", description="summary memory file")


class StorageMemory(Memory):
    def __init__(self, storage: Storage | FileStorage):
        self.storage = storage

        self._configs = WorkspaceConfigs(storage)
        self._meta_config = self._configs.get_or_create(MetaConfig())

        self._current_session: Optional[Session] = None

        # lifecycle
        self._started = asyncio.Event()

    def session_file_name(self) -> str:
        return f"thread_{self._meta_config.current_session_id}.json"

    # ==== Agent Memory Interface ====
    async def _save_current_session(self):
        file_bytes = self._current_session.model_dump_json(indent=4, ensure_ascii=False, exclude_none=True).encode()
        self.storage.put(self.session_file_name(), file_bytes)

    async def save_turn(self, inputs: List[Message], outputs: List[Message]):
        if not self._has_current_session():
            await self.new_session()

        if self.storage.exists(self.session_file_name()):
            file_data = self.storage.get(self.session_file_name())
            self._current_session = Session.model_validate_json(file_data)

        saved_outputs = [o for o in outputs if o.is_completed()]  # 只保存完整的消息包

        turn_id = str(uuid.uuid4())
        for msg in inputs + saved_outputs:
            msg.with_additions(TurnAddition(turn_id=turn_id))

        self._current_session.messages.extend(inputs + saved_outputs)
        await self._save_current_session()

    async def get_session_history(self) -> List[Message]:
        if not self._started.is_set():
            await self.start()

        session_history = self._current_session.model_copy(deep=True)
        session_history.messages.reverse()  # 倒转顺序方便处理

        res = []
        current_turn = ""
        count = 0
        for msg in session_history.messages:
            turn = TurnAddition.read(msg)
            if not turn:
                continue
            if turn.turn_id != current_turn:
                current_turn = turn.turn_id
                count += 1
            # 超过的轮次就不要了
            if count > self._meta_config.turn_rounds:
                break
            res.append(msg)

        res.reverse()  # 再把顺序调整回来
        return res

    def _has_current_session(self) -> bool:
        if not self._meta_config.current_session_id:
            return False
        if not self.storage.exists(self.session_file_name()):
            return False
        return True

    async def start(self):
        if not self._has_current_session():
            await self.new_session()
        else:
            file_data = self.storage.get(self.session_file_name())
            self._current_session = Session.model_validate_json(file_data)
        self._started.set()

    async def close(self):
        pass

    # ==== Model Context ====
    async def new_session(self):
        self._meta_config.current_session_id = str(uuid.uuid4())
        self._configs.save(self._meta_config)
        self._current_session = Session(messages=[])
        await self._save_current_session()

    async def set_limitation(self, turn_rounds: int=10, max_tokens: int=-1) -> str:
        """Configures conversation context visibility boundaries for the agent.

        This tool controls how much historical conversation the agent can access by setting two key parameters:
        maximum conversation rounds and maximum token limit for historical content. Execution is restricted
        to explicit user requests only (no arbitrary/automatic execution).

        Args:
            turn_rounds: Integer defining the maximum number of conversation rounds (user-agent exchanges)
                the agent can access. Default: 10. Valid range: ≥ 0 (0 = no historical rounds accessible).
            max_tokens: Integer defining the maximum number of tokens of historical conversation content
                the agent can access. Default: -1 (special value = unlimited tokens). Valid range: -1 or ≥ 0
                (0 = no token-based content accessible).

        Execution Rules (STRICTLY ENFORCE):
            1. Invocation Restriction: Do NOT execute this tool automatically or arbitrarily. Execute ONLY if:
               - The user explicitly requests adjustment of context visibility (e.g., "Limit to 5 conversation rounds",
                 "Set max tokens to 2000 for history access")
               - The user provides explicit values for turn_rounds, max_tokens, or both
        """
        self._meta_config.turn_rounds = turn_rounds
        self._meta_config.max_tokens = max_tokens
        self._configs.save(self._meta_config)
        return "set memory session limitation done"

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
        if not self.storage.exists(self._meta_config.summary_memory_md):
            return ""
        res = self.storage.get(self._meta_config.summary_memory_md)
        return res.decode()

    async def context_messages(self):
        msgs = await self.get_session_history()

        msgs.append(Message.new(role="system", name="__personality__").with_content(
            Text(text=await self.read_md(self._meta_config.personality_md)),
            Text(text=await self.read_md(self._meta_config.behavior_preference_md)),
            Text(text=await self.read_md(self._meta_config.mood_base_md)),
        ))

        msgs.append(Message.new(role="system", name="__memory__").with_content(
            Text(text=await self.read_md(self._meta_config.autobiographical_memory_md)),
            Text(text=await self.read_md(self._meta_config.summary_memory_md)),
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

        memory.build.command()(self.new_session)
        memory.build.command()(self.set_limitation)
        memory.build.command()(self.refresh_personality)
        memory.build.command()(self.refresh_behavior_preference)
        memory.build.command()(self.refresh_mood_base)
        memory.build.command()(self.refresh_autobiographical_memory)
        memory.build.command()(self.refresh_summary_memory)
        memory.build.with_context_messages(self.context_messages)

        return memory


def new_ws_storage_memory(container: Container=None) -> StorageMemory:
    ws = container.force_fetch(Workspace)
    storage = ws.runtime().sub_storage("memory")
    return StorageMemory(storage)
