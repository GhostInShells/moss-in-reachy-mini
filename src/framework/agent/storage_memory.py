import json
import time
import uuid
from typing import List

from ghoshell_common.contracts import Storage, Workspace
from ghoshell_container import Container
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


class StorageMemory(Memory):
    def __init__(self, storage: Storage):
        self.storage = storage
        self.turn_rounds = 10
        self.max_tokens = -1

        self._current_session_id = ""
        self._summary_file_name = ".summary.md"

    def session_file_name(self, session_id: str="") -> str:
        if not session_id:
            session_id = self._current_session_id
        return f"thread_{session_id}.json"

    # ==== Agent Memory Interface ====
    async def save_turn(self, session_id: str, inputs: List[Message], outputs: List[Message]) -> str:
        if not session_id:
            self._current_session_id = str(int(time.time()))
        else:
            self._current_session_id = session_id

        session_history: Session = Session(messages=[])
        if self.storage.exists(self.session_file_name()):
            file_data = self.storage.get(self.session_file_name())
            session_history = Session.model_validate_json(file_data)

        saved_outputs = [o for o in outputs if o.is_completed()]  # 只保存完整的消息包

        turn_id = str(uuid.uuid4())
        for msg in inputs + saved_outputs:
            msg.with_additions(TurnAddition(turn_id=turn_id))

        session_history.messages.extend(inputs + saved_outputs)
        file_bytes = session_history.model_dump_json(ensure_ascii=False, exclude_none=True).encode()

        self.storage.put(self.session_file_name(), file_bytes)
        return self._current_session_id

    async def get_session_history(self, session_id: str="") -> List[Message]:
        if not session_id:
            session_id = self._current_session_id
        if not self.storage.exists(self.session_file_name(session_id)):
            return []

        file_data = self.storage.get(self.session_file_name(session_id))
        session_history = Session.model_validate_json(file_data)
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
            if count > self.turn_rounds:
                break
            res.append(msg)

        res.reverse()  # 再把顺序调整回来
        return res

    # ==== Model Context ====
    async def set_limitation(self, turn_rounds: int=10, max_tokens: int=-1) -> str:
        self.turn_rounds = turn_rounds
        self.max_tokens = max_tokens
        return "set memory session limitation done"

    async def write_summary(self, text__: str):
        """
        refresh your memory summary with new summary, you can write a very long summary in Markdown format.
        """
        self.storage.put(self._summary_file_name, text__.encode())

    async def read_summary(self) -> str:
        if not self.storage.exists(self._summary_file_name):
            return ""
        res = self.storage.get(self._summary_file_name)
        return res.decode()

    async def context_messages(self):
        msgs = await self.get_session_history()
        read_summary = await self.read_summary()
        msgs.append(Message.new(role="assistant", name="__history_summary__").with_content(
            Text(text=read_summary),
        ))
        return msgs

    def as_channel(self) -> PyChannel:
        memory = PyChannel(name="memory")

        memory.build.command()(self.set_limitation)
        memory.build.command()(self.write_summary)
        memory.build.with_context_messages(self.context_messages)

        return memory


def new_ws_storage_memory(container: Container=None) -> StorageMemory:
    ws = container.force_fetch(Workspace)
    storage = ws.runtime().sub_storage("memory")
    return StorageMemory(storage)
