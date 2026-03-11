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
    relative_path = ".conversation_meta.yaml"

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

        saved_outputs = [o for o in outputs if o.is_completed() and not o.is_empty()]  # 只保存完整的消息包

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
        return "set conversation session limitation done"

    async def context_messages(self):
        msgs = await self.get_session_history()

        msgs.append(Message.new(role="system", name="__session_settings__").with_content(
            Text(text=f"Config: {self._meta_config.model_dump_json()}"),
        ))
        return msgs

    def as_channel(self) -> PyChannel:
        conversation = PyChannel(
            name="conversation",
            description="refresh相关的command请放置在你的回答结束末尾调用，并且请提前告知用户你要更新你自己的人格和记忆了",
        )

        conversation.build.command()(self.new_session)
        conversation.build.command()(self.set_limitation)
        conversation.build.context_messages(self.context_messages)

        return conversation


