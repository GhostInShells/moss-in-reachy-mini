from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import Message, Text

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import UserInputAgentEvent, InterruptAgentEvent, CTMLAgentEvent


class AgentFastAPI:
    def __init__(self, eventbus: EventBus, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self._eventbus = eventbus

        # 设置 CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self):
        """设置路由"""

        @self.app.post("/api/events/user_input")
        async def user_input_event(data: Dict[str, Any]):
            """处理用户输入事件"""
            try:
                text = data.get("text", "")
                enqueue = []
                if text:
                    message = Message.new(role="user", name="__user__").with_content(
                        Text(text=text)
                    )
                    event = UserInputAgentEvent(message=message, priority=0)
                    await self._eventbus.put(event.to_agent_event())
                    enqueue.append("Text")

                ctml = data.get("ctml", "")
                if ctml:
                    await self._eventbus.put(
                        CTMLAgentEvent(
                            ctml=ctml,
                            priority=0,
                        ).to_agent_event()
                    )
                    enqueue.append("CTML")

                return {"status": "success", "message": f"Enqueued: {', '.join(enqueue)}"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/api/events/interrupt")
        async def interrupt_event():
            """处理中断事件"""
            try:
                event = InterruptAgentEvent()
                await self._eventbus.put(event.to_agent_event())
                return {"status": "success", "message": "Interrupt event queued"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/")
        async def root():
            return {"message": "ReachyMini Agent API", "status": "running"}

    async def run(self):
        """启动 FastAPI 服务"""
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


class AgentFastAPIProvider(Provider[AgentFastAPI]):
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> AgentFastAPI:
        eventbus = con.force_fetch(EventBus)
        return AgentFastAPI(eventbus, host=self.host, port=self.port)
