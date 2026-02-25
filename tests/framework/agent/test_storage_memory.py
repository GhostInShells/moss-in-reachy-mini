import pytest
from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_moss import Message, Text

from framework.agent.storage_memory import StorageMemory


@pytest.mark.asyncio
async def test_storage_memory():
    # StorageMemory是基于Storage抽象设计的记忆模块 Memory = 记忆
    # MemoryStorage是基于内存设计的Storage抽象的具体实现 Memory = 内存
    memory = StorageMemory(storage=MemoryStorage(dir_=""))

    session_id = "mock-session-id"
    history = await memory.get_session_history(session_id)
    assert len(history) == 0

    # 验证第一次存取数据逻辑
    await memory.save_turn(
        session_id,
        inputs=[
            Message.new(role="user").with_content(Text(text="hello"))
        ],
        outputs=[
            Message.new(role="assistant").with_content(Text(text="world"))
        ]
    )
    history = await memory.get_session_history(session_id)
    assert len(history) == 2

    # 验证后续存取数据逻辑
    await memory.save_turn(
        session_id,
        inputs=[
            Message.new(role="user").with_content(Text(text="hello2"))
        ],
        outputs=[
            Message.new(role="assistant").with_content(Text(text="world2"))
        ]
    )
    history = await memory.get_session_history(session_id)
    assert len(history) == 4

    # 验证summary逻辑
    summary = await memory.read_summary()
    assert summary == ""
    await memory.write_summary(text__="hello3")
    summary = await memory.read_summary()
    assert summary == "hello3"

    # 限制只拿最新的一轮。
    await memory.set_limitation(turn_rounds=1)
    messages = await memory.context_messages()
    assert len(messages) == 3
    assert Text.from_content(messages[0].contents[0]).text == "hello2"
    assert Text.from_content(messages[1].contents[0]).text == "world2"
    assert Text.from_content(messages[2].contents[0]).text == "hello3"
