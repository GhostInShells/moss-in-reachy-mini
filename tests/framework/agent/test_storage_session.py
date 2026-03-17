import pytest
from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_moss import Message, Text

from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.session.storage_session import StorageSession


@pytest.mark.asyncio
async def test_storage_session():
    # StorageMemory是基于Storage抽象设计的记忆模块 Memory = 记忆
    # MemoryStorage是基于内存设计的Storage抽象的具体实现 Memory = 内存
    memory = StorageSession(storage=MemoryStorage(dir_=""))

    history = await memory.get_session_history()
    assert len(history) == 0

    # 验证第一次存取数据逻辑
    await memory.save_turn(
        inputs=[
            Message.new(role="user").with_content(Text(text="hello"))
        ],
        outputs=[
            Message.new(role="assistant").with_content(Text(text="world"))
        ]
    )
    history = await memory.get_session_history()
    assert len(history) == 2

    # 验证后续存取数据逻辑
    await memory.save_turn(
        inputs=[
            Message.new(role="user").with_content(Text(text="hello2"))
        ],
        outputs=[
            Message.new(role="assistant").with_content(Text(text="world2"))
        ]
    )
    history = await memory.get_session_history()
    assert len(history) == 4

    # 限制只拿最新的一轮。
    await memory.set_limitation(turn_rounds=1)
    history = await memory.get_session_history()
    assert len(history) == 2
