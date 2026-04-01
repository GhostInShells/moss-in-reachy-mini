#!/usr/bin/env python3
"""
中国象棋 Agent
通过命令行控制游戏
"""

import asyncio
import json
import logging
import os
from typing import Optional, List, Dict

import websockets
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss import PyChannel, Message, Text, Base64Image, Channel, ChannelRuntime, CommandError
from pydantic import BaseModel, Field

from framework.abcd.agent_event import ProgramInputAgentEvent
from framework.abcd.agent_hub import EventBus
from framework.apps.chinese_chess.pikafish import AsyncPikafish

IS_ENABLE_CHINESE_CHESS = os.getenv("ENABLE_CHINESE_CHESS") == "1"


class BoardState(BaseModel):
    pieces: List[Dict] = Field(default_factory=list)
    current_turn: Optional[str] = Field(default="", description="Current turn")
    move_history: List[str] = Field(default_factory=list)


class GameState(BaseModel):
    game_id: Optional[str] = Field(default="",description="Game identifier")
    invite_code: Optional[str] = Field(default="", description="Invite code")
    game_state: Optional[str] = Field(default="", description="Game state")
    current_turn: Optional[str] = Field(default="", description="Current turn")
    red_player: Optional[str] = Field(default="", description="Red player")
    black_player: Optional[str] = Field(default="", description="Black player")
    winner: Optional[str] = Field(default="", description="Winner")
    board_state: Optional[BoardState] = Field(default=None, description="Board state")

    def is_finished(self) -> bool:
        return self.game_state == "finished"

    def is_playing(self) -> bool:
        return self.game_state == "playing"

    def is_red_player(self) -> bool:
        return self.red_player == "小灵"

    def is_your_turn(self) -> bool:
        if self.game_state == "finished":
            return False
        if self.is_red_player() and self.current_turn == "red":
            return True
        if not self.is_red_player() and self.current_turn == "black":
            return True
        return False


class ChessActionError(Exception):

    def __init__(self, action: str, message: str) -> None:
        self.action = action
        self.message = message
        super().__init__(f"Chess action {action} failed with {message}")


class ChineseChessChannel(Channel):
    """象棋 Agent 类"""

    def __init__(
            self,
            server_url: str = "ws://localhost:8765/ws",
            logger: LoggerItf = None,
    ):
        self._id = uuid()
        self._runtime: Optional[ChannelRuntime] = None
        self.logger = logger or logging.getLogger("chinese_chess")

        self.server_url = server_url
        self.agent_token = "default-agent-token-123456"  # 默认 Agent Token
        self.current_game_state = GameState()
        self.best_move_depth = 3

        self.can_move = asyncio.Event()

        self.ws: Optional[websockets.ClientConnection] = None
        self.max_reconnect_attempts = 5  # 最大重连次数
        self.reconnect_delay = 3  # 重连间隔（秒）
        self._reconnect_attempts = 0  # 当前重连次数
        self._pikafish = AsyncPikafish()

    def name(self) -> str:
        return "chinese_chess"

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return "中国象棋通道"

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime and self._runtime.is_running():
            return self._runtime

        chan = PyChannel(name="chess", description="chinese chess channel")

        chan.build.command(available=lambda: self.current_game_state.game_state not in ["playing", "waiting"])(self.start_game)
        chan.build.command(available=lambda: self.current_game_state.game_state == "playing")(self.move)
        chan.build.command(available=lambda: self.current_game_state.game_state != "finished")(self.end_game)
        chan.build.command()(self.set_best_move_depth)

        chan.build.context_messages(self.context_messages)
        chan.build.start_up(self.on_start_up)
        chan.build.close(self.disconnect)
        self._runtime = chan.bootstrap(container=container)
        return self._runtime

    async def connect(self, auto_reconnect=True):
        """连接到服务器"""
        try:
            # 配置 WebSocket 连接
            # open_timeout: 连接超时（5 分钟 = 300 秒）
            # ping_interval: 每 20 秒发送 ping
            # ping_timeout: 20 秒内无 pong 则关闭
            self.ws = await websockets.connect(
                self.server_url,
                open_timeout=300,  # 5 分钟连接超时
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_size=10 * 1024 * 1024,  # 最大消息 10MB
                max_queue=32
            )
            self.logger.info(f"✓ 已连接到服务器：{self.server_url}")
            self.logger.info(f"  - 连接超时：300 秒（5 分钟）")
            self.logger.info(f"  - Ping 间隔：20 秒")
            self.logger.info(f"  - Ping 超时：20 秒")
            self._reconnect_attempts = 0  # 重置重连计数

            # 连接成功后，查询是否有正在进行的游戏
            await self.sync_game_state()

            # 监听 WebSocket 消息
            asyncio.create_task(self.lister_ws())
            return True
        except Exception as e:
            self.logger.info(f"✗ 连接失败：{e}")
            if auto_reconnect and self._reconnect_attempts < self.max_reconnect_attempts:
                self._reconnect_attempts += 1
                self.logger.info(f"正在重连... ({self._reconnect_attempts}/{self.max_reconnect_attempts})")
                await asyncio.sleep(self.reconnect_delay)
                await self.connect(auto_reconnect)
            return False

    async def sync_game_state(self):
        """同步游戏状态，获取正在进行的游戏"""
        self.logger.info("\n=== 同步游戏状态 ===")

        # 直接发送查询请求，不通过 send_command（避免响应处理冲突）
        message = {
            "action": "game_state",
            "agent_token": self.agent_token
        }

        try:
            await self.ws.send(json.dumps(message, ensure_ascii=False))
        except asyncio.TimeoutError:
            self.logger.info("✗ 查询超时")
        except Exception as e:
            self.logger.info(f"✗ 同步游戏状态失败：{e}")

    async def disconnect(self):
        """断开连接"""
        if self.ws:
            await self.ws.close()
            self.logger.info("✓ 已断开连接")

    async def ensure_connected(self):
        """确保连接有效，如果断开则重连"""
        if not self.ws or self.ws.state != websockets.State.OPEN:
            self.logger.info("检测到连接断开，正在重连...")
            return await self.connect(auto_reconnect=True)
        return True

    async def lister_ws(self):
        # 等待响应（可能需要接收多个消息，跳过广播消息）
        while True:
            try:
                response = await self.ws.recv()
                data = json.loads(response)
                self.logger.info(f"recv {data}")

                if data.get("type") == "game_state":
                    self.current_game_state = GameState.model_validate(data)

                    # 如果是自己的回合，设置事件
                    if self.current_game_state.is_your_turn():
                        self.can_move.set()
                    else:
                        self.can_move.clear()
                # else:
                #     await self.sync_game_state()

                if data.get("type") == "queue_state":
                    self.logger.info(f"  [广播] 收到队列状态更新: {data}")
                    continue

                if data.get("type") == "start_game":
                    self.can_move.clear()
                    await self.sync_game_state()
                    pass

            except asyncio.TimeoutError:
                self.logger.info("✗ 响应超时")
            except websockets.exceptions.ConnectionClosed:
                self.logger.info("✗ 连接已关闭")
                # 尝试重连
                await self.ensure_connected()

    async def send_command(self, action: str, **kwargs):
        """发送命令到服务器"""
        # 确保连接有效
        if not await self.ensure_connected():
            self.logger.info("✗ 无法连接到服务器")
            return None

        message = {
            "action": action,
            "agent_token": self.agent_token,
            **kwargs
        }

        self.logger.info(f"→ 发送命令：{action}")

        try:
            await self.ws.send(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            self.logger.info(f"✗ 发送失败：{e}")
            # 尝试重连
            await self.ensure_connected()
            return None

    async def start_game(self, nickname: str = "小灵"):
        """开始一局新游戏

        Args:
            nickname: 你的昵称，默认为"小灵"
        """
        self.logger.info("\n=== 启动游戏 ===")
        # 使用传入的昵称（默认为"小灵"）
        self.logger.info(f"使用昵称：{nickname}")
        # 构建参数
        params = {"nickname": nickname}
        await self.send_command("start_game", **params)

    async def move(self, move_uci: str):
        """走一步棋

        Args:
            move_uci: UCI 格式的走棋，如 "b2b5"（表示从b2移动到b5）
        """
        game_id = self.current_game_state.game_id

        if not game_id:
            raise ValueError("请先启动游戏")

        # 解析 UCI 格式用于显示
        if len(move_uci) == 4:
            from_uci = move_uci[0:2]
            to_uci = move_uci[2:4]
            from_pos = self._uci_to_pos(from_uci)
            to_pos = self._uci_to_pos(to_uci)
            self.logger.info(f"\n=== 走棋 ===")
            self.logger.info(f"UCI: {move_uci}")
            self.logger.info(f"从：{from_pos} ({from_uci}) → 到：{to_pos} ({to_uci})")
        else:
            self.logger.info(f"\n=== 走棋 ===")
            self.logger.info(f"UCI: {move_uci}")

        params = {
            "game_id": game_id,
            "move": move_uci  # 直接发送完整的 UCI 格式
        }

        await self.send_command("move", **params)

    def _uci_to_pos(self, uci: str) -> list:
        """将 UCI 格式转换为棋盘坐标"""
        col_char = uci[0].lower()
        row_char = uci[1]

        col = ord(col_char) - ord('a')
        row = 9 - int(row_char)

        return [row, col]

    async def end_game(self):
        """结束当前游戏"""
        game_id = self.current_game_state.game_id

        self.logger.info(f"\n=== 结束游戏 (游戏：{game_id}) ===")
        await self.send_command("end_game", game_id=game_id)

    async def set_best_move_depth(self, depth: int):
        """
        设置思考深度（1-10），深度越高越聪明但速度越慢
        :param depth: 思考深度参数
        """
        self.best_move_depth = depth

    async def get_best_move_and_board(self):
        game_state = self.current_game_state
        uci_history = []
        if game_state.board_state:
            uci_history = game_state.board_state.move_history
        best_move, board_str = await self._pikafish.get_best_move(uci_history, self.best_move_depth)
        return best_move, board_str

    async def context_messages(self) -> List[Message]:
        msg = Message.new(role="user", name="__chess__")
        game_state = self.current_game_state
        if game_state and game_state.is_playing():
            # 红黑方信息 - 用更自然的语气
            if game_state.is_red_player():
                msg.with_content(
                    Text(text=f"这局我是红方，执先手。对手是{game_state.black_player}。"),
                )
            else:
                msg.with_content(
                    Text(text=f"这局我是黑方，后手应对。对手是{game_state.red_player}。"),
                )

        else:
            msg.with_content(
                Text(text="现在没有正在进行的棋局。如果你想和我下棋，请用start_game命令开始新游戏～")
            )

        return [msg]

    async def on_start_up(self):
        await self.connect()
        await self._pikafish.start()
