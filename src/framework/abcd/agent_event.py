import time
from typing import TypedDict, Optional, Any, Self, ClassVar, List

from ghoshell_common.helpers import uuid
from ghoshell_moss import Base64Image, Message
from pydantic import BaseModel, Field


class AgentEvent(TypedDict):
    event_id: str
    event_type: str
    agent_id: str
    priority: Optional[int]
    issuer: Optional[str]
    overdue: Optional[float]
    created: float

    data: Optional[dict[str, Any]]


class AgentEventModel(BaseModel):
    event_type: ClassVar[str] = ""

    event_id: str = Field(default_factory=uuid, description="事件的id, 也可以用于链路 trace.")
    agent_id: str = Field(default="", description="事件处理的Agent")
    priority: int = Field(
        default=1,
        description=(
            "事件的优先级, 分成三种级别. "
            "<0 表示低优事件, 只要 agent 运行就会入队. "
            "0 表示普通事件, 优先级高于 <0, 但不会中断任何事件运行, 会被高优事件彻底 drop 掉. "
            ">0 表示高优事件, 抢占式调度, 如果优先级低于进行中事件, 会被忽视. 高于进行中事件, 则会中断进行中事件. "
        ),
    )
    issuer: str = Field(default="", description="事件的发起方服务.")
    overdue: float = Field(
        default=0,
        description="以秒为单位, 精确到毫秒级别的时间. 如果一个事件过期, 它会被直接丢弃. 为 0 表示永不过期."
    )
    created: float = Field(
        default_factory=lambda: round(time.time(), 4),
        description="事件创建的事件."
    )

    def is_overdue(self, now: float = None) -> bool:
        """
        事件是否已经过期.
        """
        if now is None:
            now = round(time.time(), 4)
        return self.overdue > 0 and ((now - self.created) > self.overdue)

    def to_agent_event(self) -> AgentEvent:
        data = self.model_dump(exclude_none=True, exclude={"event_id", "event_type", "agent_id", "priority", "issuer", "overdue", "created"})
        return AgentEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            agent_id=self.agent_id,
            priority=self.priority,
            issuer=self.issuer,
            overdue=self.overdue,
            created=self.created,
            data=data
        )

    @classmethod
    def from_agent_event(cls, agent_event: AgentEvent) -> Optional[Self]:
        if cls.event_type != agent_event["event_type"]:
            return None
        data = agent_event.get("data", {})
        data["event_id"] = agent_event["event_id"]
        data["agent_id"] = agent_event["agent_id"]
        data["priority"] = agent_event["priority"]
        data["issuer"] = agent_event["issuer"]
        data["overdue"] = agent_event["overdue"]
        data["created"] = agent_event["created"]
        return cls(**data)

    @classmethod
    def from_agent_event_model(cls, agent_event: "AgentEventModel") -> Optional[Self]:
        if cls.event_type != agent_event.event_type:
            return None
        return cls.from_agent_event(
            agent_event=agent_event.to_agent_event(),
        )


class UserInputAgentEvent(AgentEventModel):
    event_type = "user_input"

    message: Message = Field(description="input message")


class InterruptAgentEvent(AgentEventModel):
    event_type = "interrupt"


class AsrInvokeAgentEvent(AgentEventModel):
    event_type = "asr_invoke"


class VisionAgentEvent(AgentEventModel):
    event_type = "vision"

    content: str = Field(default="", description="视觉的提示文本讯息.")
    images: List[Base64Image] = Field(default_factory=list, description="base64 encoded image.")


class ReactAgentEvent(AgentEventModel):
    event_type = "react"

    messages: List[Message] = Field(description="react messages")


class ShutdownAgentEvent(AgentEventModel):
    event_type = "shutdown"


class UnexpectedAgentEvent(AgentEventModel):
    """
    只是用来边界条件适配的
    """
    event_type = "__unexpected__"

class CTMLAgentEvent(AgentEventModel):
    event_type = "ctml"

    ctml: str = Field(description="ctml 文本")

class ResumeAgentEvent(AgentEventModel):
    event_type = "resume"

    message: Message = Field(description="resume message")
    event: AgentEventModel = Field(description="被中断的事件")
