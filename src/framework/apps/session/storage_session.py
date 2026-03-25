import asyncio
import uuid
from typing import List

from ghoshell_common.contracts import Storage, YamlConfig, WorkspaceConfigs, FileStorage
from ghoshell_moss import Message, Text, Addition, PyChannel
from pydantic import BaseModel, Field

from framework.abcd.session import Session


class SessionData(BaseModel):
    messages: List[Message] = Field(...)


class TurnAddition(Addition):
    turn_id: str = Field(default="", description="turn id")  # 用来关联同一轮对话的input和output

    @classmethod
    def keyword(cls) -> str:
        return "session_turn"


class MetaConfig(YamlConfig):
    relative_path = ".session_meta.yaml"

    current_session_id: str = Field(default="", description="current session id")
    turn_rounds: int = Field(default=10, description="turn rounds")
    max_tokens: int = Field(default=-1, description="max tokens")


class StorageSession(Session):
    def __init__(self, storage: Storage | FileStorage):
        self.storage = storage

        self._configs = WorkspaceConfigs(storage)
        self._meta_config = self._configs.get_or_create(MetaConfig())

        # lifecycle
        self._started = asyncio.Event()

    @property
    def meta_config(self) -> MetaConfig:
        return self._meta_config

    def session_file_name(self) -> str:
        return f"thread_{self._meta_config.current_session_id}.json"

    # ==== model context ====
    async def _get_session(self) -> SessionData:
        if not self._has_current_session():
            await self.new_session()
        file_data = self.storage.get(self.session_file_name())
        session = SessionData.model_validate_json(file_data)
        return session

    async def _save_session(self, session: SessionData):
        file_bytes = session.model_dump_json(indent=4, ensure_ascii=False, exclude_none=True).encode()
        self.storage.put(self.session_file_name(), file_bytes)

    async def save_turn(self, inputs: List[Message], outputs: List[Message]):
        session = await self._get_session()

        saved_outputs = [o for o in outputs if o.is_completed() and not o.is_empty() and o.contents]  # 只保存完整的消息包

        turn_id = str(uuid.uuid4())
        for msg in inputs + saved_outputs:
            msg.with_additions(TurnAddition(turn_id=turn_id))

        session.messages.extend(inputs + saved_outputs)
        await self._save_session(session)

    async def get_session_history(self) -> List[Message]:
        if not self._started.is_set():
            await self.start()

        session_history = await self._get_session()
        session_history.messages.reverse()

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
        self._started.set()

    async def close(self):
        pass

    # ==== Model Context ====
    async def new_session(self):
        self._meta_config.current_session_id = str(uuid.uuid4())
        self._configs.save(self._meta_config)
        new_session = SessionData(messages=[])
        await self._save_session(new_session)

    async def set_limitation(self, turn_rounds: int=10) -> str:
        """
        设置Session最大可见轮次
        :param turn_rounds: 最大轮次
        """
        self._meta_config.turn_rounds = turn_rounds
        # self._meta_config.max_tokens = max_tokens
        self._configs.save(self._meta_config)
        return "set session session limitation done"

    async def context_messages(self):
        # 记忆放到了MainAgent里组织顺序
        # msgs = await self.get_session_history()

        return [Message.new(role="system", name="__session_settings__").with_content(
            Text(text=f"Config: {self._meta_config.model_dump_json()}"),
        )]

    def as_channel(self) -> PyChannel:
        session = PyChannel(
            name="session",
            description="聊天会话通道"
        )
        session.build.command()(self.new_session)
        session.build.command()(self.set_limitation)
        session.build.context_messages(self.context_messages)

        return session


