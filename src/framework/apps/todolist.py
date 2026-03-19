import asyncio
import enum
import json
import logging
from collections import defaultdict
from typing import List, Dict

from ghoshell_common.contracts import LoggerItf, FileStorage, Workspace, Storage
from ghoshell_container import Provider, IoCContainer, INSTANCE
from ghoshell_moss import PyChannel, Message, Text
from pydantic import BaseModel, Field

from framework.abcd.agent_event import ProgramInputAgentEvent
from framework.abcd.agent_hub import EventBus
from framework.apps.utils import EnumEncoder


class TodoStatus(enum.Enum):
    TODO = "todo"
    DONE = "done"
    DOING = "doing"


class Todo(BaseModel):
    key: str = Field(description="key")
    title: str = Field(default="", description="title")
    description: str = Field(default="", description="description")
    status: TodoStatus = Field(default=TodoStatus.TODO, description="is_done")
    parent_key: str = Field(default="", description="parent_key")
    result: str = Field(default="", description="result")


# 树形结构生成工具类
class TodoTreeGenerator:
    def __init__(self, todos: List[Todo]):
        self.todos = todos
        # 构建 key 到 To.do. 对象的映射，方便快速查找
        self.todo_map: Dict[str, Todo] = {todo.key: todo for todo in todos}
        # 构建父节点到子节点的映射
        self.parent_children_map: Dict[str, List[Todo]] = self._build_parent_children_map()

    def get_todo(self, key: str, raise_error: bool = True):
        for todo in self.todos:
            if todo.key == key:
                return todo
        if raise_error:
            raise ValueError(f"key {key} not found")
        return None

    def get_children(self, key: str):
        res = defaultdict(list)
        for todo in self.todos:
            res[todo.parent_key].append(todo)
        return res.get(key, [])

    def _build_parent_children_map(self) -> Dict[str, List[Todo]]:
        """构建父节点key到子节点列表的映射"""
        parent_map = {}
        for todo in self.todos:
            parent_key = todo.parent_key or "root"  # 根节点的父key统一为"root"
            if parent_key not in parent_map:
                parent_map[parent_key] = []
            parent_map[parent_key].append(todo)
        return parent_map

    def generate_tree_text(self, parent_key: str = "root", level: int = 0) -> str:
        """
        递归生成树形结构文本
        :param parent_key: 父节点key，默认root表示根节点
        :param level: 当前层级，用于控制缩进和前缀符号
        :return: 树形结构文本
        """
        # 定义层级前缀符号
        prefix_symbols = ["├── ", "└── "]
        indent = "    " * (level - 1) if level > 0 else ""

        tree_text = ""
        children = self.parent_children_map.get(parent_key, [])

        for index, child in enumerate(children[:10]):
            # 判断是最后一个子节点（用└──）还是中间节点（用├──）
            is_last = index == len(children) - 1
            prefix = indent + (prefix_symbols[1] if is_last else prefix_symbols[0]) if level > 0 else ""

            # 构建当前节点的文本行（包含状态、标题、key）
            status_icon = {
                TodoStatus.TODO: "[ ]",
                TodoStatus.DOING: "[~]",
                TodoStatus.DONE: "[✓]"
            }.get(child.status, "[ ]")

            node_line = f"{prefix}{status_icon} {child.title} (key: {child.key}, description: {child.description})"
            tree_text += node_line + "\n"

            # 递归处理子节点
            tree_text += self.generate_tree_text(child.key, level + 1)

        return tree_text


class TodoList:

    def __init__(self, eventbus: EventBus, storage: Storage, logger: LoggerItf=None):
        self.eventbus = eventbus
        self.storage = storage
        self._locker: asyncio.Lock = asyncio.Lock()
        self.logger = logger or logging.getLogger("TodoList")

    @property
    def todos(self) -> List[Todo]:
        if not self.storage.exists("todolist.json"):
            return []
        file_bytes = self.storage.get("todolist.json")

        ll = json.loads(file_bytes)
        res = []
        for item in ll:
            res.append(Todo.model_validate(item))

        return res

    @property
    def todo_todos(self):
        todos = self.todos
        res = []
        for todo in todos:
            if todo.status not in [TodoStatus.DONE, TodoStatus.DOING]:
                res.append(todo)
        return res

    def _save(self, todos: List[Todo]):
        prepare = []
        for todo in todos:
            prepare.append(todo.model_dump())

        file_bytes = json.dumps(prepare, ensure_ascii=False, indent=4, cls=EnumEncoder).encode("utf-8")
        self.storage.put("todolist.json", file_bytes)

    async def clear_todo(self):
        """
        清空当前任务列表
        """
        self._save([])
        return "already clear todolist"

    async def append_todo(self, key: str, title: str, description: str, parent_key: str=""):
        """
        给任务列表追加一个to.do.任务
        :param key: 任务的唯一标识
        :param title: 任务的标题
        :param description: 任务的详细描述
        :param parent_key: 任务的父节点，默认为空字符串表示为根节点
        """
        todos = self.todos
        generator = TodoTreeGenerator(todos)
        todo = generator.get_todo(key, raise_error=False)
        if todo:
            raise ValueError(f"key {key} already exists")

        if parent_key:
            generator.get_todo(parent_key, raise_error=True)

        async with self._locker:
            todos.append(
                Todo(
                    key=key,
                    title=title,
                    description=description,
                    parent_key=parent_key,
                )
            )
            self._save(todos)

    async def mark_as_doing(self, key: str):
        """
        标记一个to.do.任务开始，且只能标记叶子节点的to.do.任务，自身和所有的父节点都会被标记为DOING状态
        :param key: 叶子节点的to.do任务
        """
        todos = self.todos
        generator = TodoTreeGenerator(todos)
        children = generator.get_children(key)
        if len(children) > 0:
            raise ValueError(f"Task {key} has children, start it's children first")
        async with self._locker:
            todo = generator.get_todo(key)
            # if todo.status == TodoStatus.DOING:
            #     raise ValueError(f"Task {key} already mark as doing, you need to execute this task now and when you finish this task, mark it as done")
            if todo.status == TodoStatus.DONE:
                raise ValueError(f"Task {key} already mark as done, you need to start another task")
            todo.status = TodoStatus.DOING

            # 递归所有的父节点
            parent = todo
            while True:
                parent = generator.get_todo(parent.parent_key, raise_error=False)
                if not parent:
                    break
                parent.status = TodoStatus.DOING
            self._save(todos)
            return f"Task {key} marked as done, you need to execute this task with {todo.description}"

    async def mark_as_done(self, key: str, text__):
        """
        标记一个to.do.任务已完成，且只能标记叶子节点的to.do.任务，自身会被标记为DONE状态，所有的父节点都会被检查是否所有子任务都完成
        :param key: 叶子节点的to.do任务
        :param text__: 任务的详细执行情况，使用文本传递
        """
        todos = self.todos
        generator = TodoTreeGenerator(todos)
        children = generator.get_children(key)
        if len(children) > 0:
            raise ValueError(f"key {key} has children, finish it's children first")
        async with self._locker:
            todo = generator.get_todo(key)
            todo.status = TodoStatus.DONE
            todo.result = text__

            parent = todo
            while parent:
                parent = generator.get_todo(parent.parent_key, raise_error=False)
                if not parent:
                    break
                children = generator.get_children(parent.key)
                all_done = True
                for child in children:
                    if child.status != TodoStatus.DONE:
                        all_done = False
                        break
                if all_done:
                    parent.status = TodoStatus.DONE
            self._save(todos)

            await self.eventbus.put(ProgramInputAgentEvent(
                message=Message.new(role="user", name="__todolist__").with_content(
                    Text(text=f"任务 {key} 已完成，当前的任务结果是你的旁路大脑完成的，用户并不知道任务结果，你需要自然衔接之前的对话，将任务的结果告诉用户"),
                    Text(text=f"任务名为{todo.title} 描述为{todo.description}"),
                    Text(text=f"任务结果为{todo.result}")
                ),
                agent_id="", # 默认是主agent
            ))

    async def context_messages(self):
        text = TodoTreeGenerator(self.todos).generate_tree_text()
        self.logger.debug(f"current todo list=\n{text}")
        msg = Message.new(role="system", name="__todolist__")
        if text:
            msg.with_content(
                Text(text="todolist如下所示，任务状态说明：[✓]已完成 [~]执行中 [ ]未开始"),
                Text(text=text),
            )
        else:
            msg.with_content(
                Text(text="todolist当前是空的")
            )
        return [msg]

    def as_channel(self, is_main_agent=True):
        chan = PyChannel(
            name="todolist",
            description="规划任务，标记进度",
            blocking=True,
        )
        chan.build.command()(self.append_todo)
        if not is_main_agent:
            chan.build.command()(self.mark_as_doing)
            chan.build.command()(self.mark_as_done)
            # chan.build.command()(self.clear_todo)
        chan.build.context_messages(self.context_messages)
        return chan


class TodoListProvider(Provider[TodoList]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        ws = con.force_fetch(Workspace)
        eventbus = con.force_fetch(EventBus)
        storage: FileStorage|Storage = ws.runtime().sub_storage("todolist")
        logger = con.get(LoggerItf)
        return TodoList(eventbus=eventbus, storage=storage, logger=logger)


if __name__ == "__main__":
    # 创建测试数据
    test_todos = [
        Todo(key="1", title="项目规划", description="xxx", status=TodoStatus.DOING),
        Todo(key="2", title="需求分析", parent_key="1"),
        Todo(key="3", title="用户调研", parent_key="2"),
        Todo(key="4", title="竞品分析", description="xxx", parent_key="2"),
        Todo(key="5", title="技术选型", parent_key="1", status=TodoStatus.DOING),
        Todo(key="6", title="前端框架选择", parent_key="5", status=TodoStatus.DONE),
        Todo(key="7", title="后端架构设计", parent_key="5", status=TodoStatus.DOING),
        Todo(key="8", title="项目验收", status=TodoStatus.TODO)
    ]

    # 打印结果
    print("===== Todo 树形结构 =====")
    print(TodoTreeGenerator(test_todos).generate_tree_text())