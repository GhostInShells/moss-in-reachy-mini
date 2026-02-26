import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, List, Any, Optional, ClassVar, AsyncIterator, AsyncIterable

from ghoshell_common.identifier import Identifier
from ghoshell_container import Container
from ghoshell_moss import Message, Addition
from pydantic import BaseModel, Field
from typing_extensions import Self

from framework.abcd.agent_event import AgentEventModel, AgentEvent


class Response(ABC):
    thread_id: str
    response_id: str
    inputs: List[Message]
    interrupted: bool  # 被打断了
    """输入的消息体, 用于处理保存逻辑"""


    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        结束 response.
        """
        await self.close()

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def close(self):
        pass

    @abstractmethod
    def stream_messages(self) -> AsyncIterable[Message]:
        pass

    @abstractmethod
    def inputted(self) -> List[Message]:
        pass

    @abstractmethod
    def buffered(self) -> List[Message]:
        """
        历史消息.
        """
        pass

    @abstractmethod
    async def interrupt(self) -> None:
        """
        强行中断输出. 提供一个明确的中断信号. 要求是非同步阻塞的.
        """
        pass


class Broadcaster(ABC):

    @abstractmethod
    async def broadcast(self, agent_id: str, message: Union[Message, None]) -> None:
        """
        接受事件, 要足够快!!
        :param agent_id: agent id
        :param message: 需要广播的消息. 为 None 没有任何作用.
        """
        pass

    @abstractmethod
    def bootstrap(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        关闭广播.
        """
        pass


class AgentStateName(str, Enum):
    """
    Agent 状态机的基本状态.
    """
    CRATED = "created"
    """被创建出来, 没有初始化"""

    BOOTSTRAP = "bootstrap"
    """初始化过程中"""

    SHUTDOWN = "shutdown"
    """已经运行结束"""

    RESPONDING = "responding"
    """正在回复中"""

    HALT = "halt"
    """暂停运行, 也不接受事件输入"""

    IDLE = "idle"
    """空闲中, 空闲久了可能自动关闭"""

    SWITCHING = "switching"
    """正在模式切换中, 事件都会被丢弃"""

    @classmethod
    def is_available(cls, state: Self) -> bool:
        """
        用来判断一个 Agent 是否可以接受消息.
        """
        return state != cls.SHUTDOWN and state != cls.HALT and state != cls.CRATED


class EventBus(ABC):
    @abstractmethod
    async def put(self, event: AgentEvent) -> None:
        """
        将 event 入队.
        """
        pass

    @abstractmethod
    async def get(self, timeout: Union[float, None] = None) -> Union[AgentEvent, None]:
        """
        吐出 event.
        :param timeout: 超时时间, 并不会抛出异常, 如果拿不到会返回 None. 如果超时时间为 None, 则是 get_nowait
        """
        pass


class ModelConf(BaseModel):
    default_env: ClassVar[dict[str, None | str]] = {
        "base_url": None,
        "model": "gpt-3.5-turbo",
        "api_key": None,
        "custom_llm_provider": None,
    }

    base_url: Optional[str] = Field(
        default="$MOSS_LLM_BASE_URL",
        description="base url for chat completion",
    )
    model: str = Field(
        default="$MOSS_LLM_MODEL",
        description="llm model name that server provided",
    )
    api_key: Optional[str] = Field(
        default="$MOSS_LLM_API_KEY",
        description="api key",
    )
    custom_llm_provider: Optional[str] = Field(
        default="$MOSS_LLM_PROVIDER",
        description="custom LLM provider name",
    )
    temperature: float = Field(default=0.7, description="temperature")
    n: int = Field(default=1, description="number of iterations")
    max_tokens: int = Field(default=4000, description="max tokens")
    timeout: float = Field(default=30, description="timeout")
    request_timeout: float = Field(default=40, description="request timeout")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="kwargs")
    top_p: Optional[float] = Field(
        default=None,
        description="""
An alternative to sampling with temperature, called nucleus sampling, where the
model considers the results of the tokens with top_p probability mass. So 0.1
means only the tokens comprising the top 10% probability mass are considered.
""",
    )

    def generate_litellm_params(self) -> dict[str, Any]:
        params = self.model_dump(exclude_none=True, exclude={"kwargs"})
        params.update(self.kwargs)
        real_params = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                default_value = self.default_env.get(key, "")
                real_value = os.environ.get(value[1:], default_value)
                if real_value is not None:
                    real_params[key] = real_value
            else:
                real_params[key] = value
        return real_params


class AgentConfig(BaseModel):
    """
    agent 通用配置.
    """
    id: str = Field(description="agent 的 id")
    name: str = Field(description="agent 的名称. 给人看的. ")
    description: str = Field(description="agent 的描述. ")
    instructions: str = Field(default="", description="agent instruction. ")

    max_idle_time: float = Field(
        default=0.0,
        description="最大的闲置时间, 如果一直闲置会导致 agent 自我关闭. 如果为 0 则永不关闭. ",
    )
    event_interval: float = Field(
        default=0.005,
        description="事件为空时的等待时间, 保持一个最快反应时间. ",
    )
    max_error_time: int = Field(
        default=10,
        description="最大连续出错时间, 如果连续出错, 则会进入 halting 状态",
    )

    model: ModelConf = Field(
        default=ModelConf(),
        description="大模型配置"
    )


class Agent(ABC):

    @classmethod
    @abstractmethod
    def new(cls, container: Container, config: AgentConfig) -> Self:
        """
        创建一个 Agent.
        """
        pass

    @abstractmethod
    def info(self) -> Identifier:
        """
        Agent 的基础讯息.
        """
        pass

    @abstractmethod
    def state(self) -> AgentStateName:
        """
        同步获取 Agent 状态, 用于一些展示.
        """
        pass

    @abstractmethod
    def broadcaster(self) -> Broadcaster:
        pass

    @abstractmethod
    def eventbus(self) -> EventBus:
        pass

    @abstractmethod
    def halt(self, toggle: bool) -> None:
        pass

    @abstractmethod
    async def start(self, auto_shutdown: bool) -> None:
        """
        具身智能的 Agent 都是全异步运行的.
        它同时接受多端的输入, 但一次只运行一个事件.
        高优的事件, 会强行打断低优的事件.
        :param auto_shutdown: 如果为 True, 则 Agent 闲置一段时间后会自行 shutdown.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        发送关闭指令. 发送后, Agent 会进入关闭过程.
        """
        pass

    @abstractmethod
    async def wait_until_close(self, shutdown: bool) -> None:
        """
        阻塞等待 Agent 彻底运行结束.
        提供一种阻塞的方法, 方便上一层的线程或者进程理解 Agent 运行结束.
        效果类似 join.
        通常在子进程或者脚本中运行 agent, 可以用这个函数等待它彻底退出.

        :param shutdown: 如果为 True, 则这个函数会主动触发关闭.
        """
        pass


class AgentId(Addition, BaseModel):

    agent_id: str = Field(description="agent id")

    @classmethod
    def keyword(cls) -> str:
        return "agent_id"