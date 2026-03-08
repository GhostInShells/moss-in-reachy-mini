import pytest
from ghoshell_common.contracts.storage import MemoryStorage

from framework.agent.eventbus import QueueEventBus
from framework.apps.todolist import TodoList, TodoTreeGenerator, TodoStatus


@pytest.mark.asyncio
async def test_todolist():
    todolist = TodoList(eventbus=QueueEventBus(), storage=MemoryStorage(dir_=""))
    assert todolist.todos == []

    await todolist.append_todo(key="key1", title="title1", description="description1")
    todos = todolist.todos
    assert len(todos) == 1
    assert todos[0].key == "key1"

    await todolist.append_todo(key="key1.1", title="title1.1", description="description1.1", parent_key="key1")
    todos = todolist.todos
    assert len(todos) == 2
    gen = TodoTreeGenerator(todos)
    todo = gen.get_todo("key1.1")
    assert todo.key == "key1.1"

    await todolist.append_todo(key="key2", title="title2", description="description2")
    todos = todolist.todos
    assert len(todos) == 3

    # 状态测试
    # 开始状态
    assert gen.get_todo("key1.1").status == TodoStatus.TODO
    assert gen.get_todo("key1").status == TodoStatus.TODO
    await todolist.start_todo(key="key1.1")
    todos = todolist.todos
    gen = TodoTreeGenerator(todos)
    assert gen.get_todo("key1.1").status == TodoStatus.DOING
    assert gen.get_todo("key1").status == TodoStatus.DOING

    # 部分子完成状态
    await todolist.append_todo(key="key1.2", title="title1.2", description="description1.2", parent_key="key1")
    await todolist.finish_todo(key="key1.1")
    todos = todolist.todos
    gen = TodoTreeGenerator(todos)
    assert gen.get_todo("key1.1").status == TodoStatus.DONE
    assert gen.get_todo("key1").status == TodoStatus.DOING
    assert gen.get_todo("key1.2").status == TodoStatus.TODO

    # 全部子完成状态
    await todolist.finish_todo(key="key1.2")
    todos = todolist.todos
    gen = TodoTreeGenerator(todos)
    assert gen.get_todo("key1.1").status == TodoStatus.DONE
    assert gen.get_todo("key1").status == TodoStatus.DONE
    assert gen.get_todo("key1.2").status == TodoStatus.DONE