from typing import Optional
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import uvicorn


from framework.memory.storage_memory import StorageMemory


class StorageMemoryAPIServer:
    """
    StorageMemory的FastAPI服务封装类
    接收memory实例，提供异步启动/管理API服务的能力
    """

    def __init__(
            self,
            memory: StorageMemory,  # 接收外部传入的memory对象
            host: str = "0.0.0.0",
            port: int = 8088,
    ):
        # 核心参数
        self.memory = memory
        self.host = host
        self.port = port

        # 初始化FastAPI app
        self.app = FastAPI()
        self.server: Optional[uvicorn.Server] = None

        # 注册所有API路由
        self._register_routes()

    def _register_routes(self):
        """注册所有API路由（替代装饰器，更适合类封装）"""
        # 1. 新建会话
        self.app.add_api_route(
            "/memory/new_session",
            self.api_new_session,
            methods=["POST"],
            response_class=JSONResponse
        )

        # 2. 设置记忆限制
        self.app.add_api_route(
            "/memory/set_limitation",
            self.api_set_limitation,
            methods=["POST"],
            response_class=JSONResponse
        )

        # 3. 刷新指定记忆模块
        self.app.add_api_route(
            "/memory/refresh/{module}",
            self.api_refresh_memory,
            methods=["POST"],
            response_class=JSONResponse
        )

        # 4. 读取指定记忆模块
        self.app.add_api_route(
            "/memory/read/{module}",
            self.api_read_memory,
            methods=["GET"],
            response_class=JSONResponse
        )

        # 5. 获取会话信息
        self.app.add_api_route(
            "/memory/session_info",
            self.api_get_session_info,
            methods=["GET"],
            response_class=JSONResponse
        )

        # 在storage_memory_server.py的_register_routes方法中新增：
        self.app.add_api_route(
            "/memory/session_history",
            self.api_get_session_history,
            methods=["GET"],
            response_class=JSONResponse
        )

    # ========== API核心处理方法 ==========
    async def api_new_session(self):
        """新建会话API"""
        try:
            await self.memory.new_session()
            return {
                "code": 0,
                "msg": "新会话创建成功",
                "session_id": self.memory.meta_config.current_session_id
            }
        except Exception as e:
            return {"code": 1, "msg": f"创建会话失败：{str(e)}"}

    async def api_set_limitation(
            self,
            turn_rounds: int = Body(10),
            max_tokens: int = Body(-1)
    ):
        """设置记忆限制API"""
        try:
            res = await self.memory.set_limitation(turn_rounds, max_tokens)
            return {"code": 0, "msg": res}
        except Exception as e:
            return {"code": 1, "msg": f"设置限制失败：{str(e)}"}

    async def api_refresh_memory(
            self,
            module: str,
            content: str = Body(...)
    ):
        """刷新指定记忆模块API"""
        # 模块映射（和StorageMemory的方法对应）
        module_map = {
            "personality": self.memory.refresh_personality,
            "behavior_preference": self.memory.refresh_behavior_preference,
            "mood_base": self.memory.refresh_mood_base,
            "autobiographical": self.memory.refresh_autobiographical_memory,
            "summary": self.memory.refresh_summary_memory
        }

        if module not in module_map:
            return {"code": 1, "msg": f"不支持的模块：{module}"}

        try:
            await module_map[module](content)
            return {"code": 0, "msg": f"{module}模块刷新成功"}
        except Exception as e:
            return {"code": 1, "msg": f"刷新{module}失败：{str(e)}"}

    async def api_read_memory(self, module: str):
        """读取指定记忆模块API"""
        # 模块路径映射
        module_path_map = {
            "personality": self.memory.meta_config.personality_md,
            "behavior_preference": self.memory.meta_config.behavior_preference_md,
            "mood_base": self.memory.meta_config.mood_base_md,
            "autobiographical": self.memory.meta_config.autobiographical_memory_md,
            "summary": self.memory.meta_config.summary_memory_md
        }

        if module not in module_path_map:
            return {"code": 1, "msg": f"不支持的模块：{module}"}

        try:
            content = await self.memory.read_md(module_path_map[module])
            return {"code": 0, "content": content}
        except Exception as e:
            return {"code": 1, "msg": f"读取{module}失败：{str(e)}"}

    async def api_get_session_info(self):
        """获取当前会话信息API"""
        try:
            return {
                "code": 0,
                "session_id": self.memory.meta_config.current_session_id,
                "turn_rounds": self.memory.meta_config.turn_rounds,
                "max_tokens": self.memory.meta_config.max_tokens
            }
        except Exception as e:
            return {"code": 1, "msg": f"获取会话信息失败：{str(e)}"}

    async def api_get_session_history(self):
        """获取会话历史消息API（基于Message原生序列化）"""
        try:
            history = await self.memory.get_session_history()
            # 直接使用Message的dump方法生成可序列化字典
            serialized_history = [msg.dump() for msg in history]
            return {
                "code": 0,
                "history": serialized_history
            }
        except Exception as e:
            return {"code": 1, "msg": f"获取会话历史失败：{str(e)}"}

    async def run(self, reload: bool = False):
        """
        异步启动API服务
        :param reload: 是否开启热重载（开发环境用，生产环境禁用）
        """
        # 配置uvicorn参数
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            access_log=False  # 不打印访问日志
        )

        # 异步启动服务器（替代同步的uvicorn.run）
        self.server = uvicorn.Server(config)
        await self.server.serve()

    async def stop(self):
        """停止API服务（需在事件循环中调用）"""
        await self.server.shutdown()
