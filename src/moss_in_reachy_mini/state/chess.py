import asyncio
import logging
from typing import Optional, Dict, Any

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import PyChannel, Message, Text
from ghoshell_moss.transports.zmq_channel import ZMQChannelProxy
from reachy_mini import ReachyMini

from framework.abcd.agent_event import ProgramInputAgentEvent
from framework.abcd.agent_hub import EventBus
from framework.apps.chinese_chess.channel import ChineseChessChannel
from framework.apps.chinese_chess.utils import parse_chinese_board, uci_to_chinese_notation
from framework.apps.live.douyin_live import DouyinLive
from moss_in_reachy_mini.state.abcd import BaseAgentHook


class ChessPlayingState(BaseAgentHook):
    """
    象棋对弈状态
    进入此状态时，小灵专注于与当前对手下棋
    同时保持与直播间观众的互动
    """

    NAME = "chess_playing"

    def __init__(
        self,
        mini: ReachyMini,
        eventbus: EventBus,
        douyin_live: DouyinLive,
        chess_channel: ChineseChessChannel,  # 象棋通道
        logger: LoggerItf = None,
    ):
        super().__init__()
        self.mini = mini
        self.eventbus = eventbus
        self.douyin_live = douyin_live
        self.chess_channel = chess_channel
        self.logger = logger or logging.getLogger("ChessPlayingState")
        self._move_task: Optional[asyncio.Task] = None
        self.lastest_board = ""

    async def on_self_enter(self):
        """进入对弈状态"""
        self.mini.enable_motors()
        self.mini.wake_up()
        self.logger.info("进入象棋对弈状态")
        await self.start_idle_move()

    async def on_self_exit(self):
        self.logger.info("退出象棋对弈状态")

    async def _run_idle_move(self):
        """创建直播间互动消息"""

        # 留出3秒buffer
        if self._idle_move_duration < 3:
            return

        if not self._move_task or self._move_task.done():
            self._move_task = asyncio.create_task(self.wait_for_move())

        message_content = []

        unprocessed_events = await self.douyin_live.get_unprocessed_events()
        if unprocessed_events:
            message_content.append(Text(text="直播间的小伙伴们发来了一些消息："))
            for event in unprocessed_events:
                message_content.append(Text(text=f"{event.user_name}：{event.content}"))

            # 发送到事件总线，低优先级
            await self.eventbus.put(ProgramInputAgentEvent(
                prompt=Message.new(role="user", name="__chess_interaction__").with_content(
                    Text(text="对手正在思考，我们趁这个空档看看直播间的小伙伴们说了什么。用轻松的语气和观众互动一下，可以回应他们的留言或者调侃几句。")
                ),
                message=Message.new(role="user", name="__chess_interaction__").with_content(
                    *message_content
                ),
                agent_id="",  # 主脑
                priority=0,   # 普通队列
                overdue=15,
            ))

    async def wait_for_move(self):
        if not self.lastest_board:
            _, self.lastest_board = await self.chess_channel.get_best_move_and_board()

        await self.chess_channel.can_move.wait()

        best_move, board_str = await self.chess_channel.get_best_move_and_board()

        pieces = parse_chinese_board(board_str)
        # pieces_str_list = []
        # for coord in sorted(pieces.keys(), key=lambda x: (int(x[1:]), x[0])):
        #     pieces_str_list.append(f"{coord}: {pieces[coord]}")
        # pieces_str = ", ".join(pieces_str_list)

        message = Message.new(role="user", name="__chess_move__")
        prompt = Message.new(role="user", name="__chess_move__")

        if self.chess_channel.current_game_state.board_state.move_history:
            player_last_move = self.chess_channel.current_game_state.board_state.move_history[-1]
            if self.lastest_board:
                player_last_move_notation = uci_to_chinese_notation(self.lastest_board, player_last_move)
                message.with_content(Text(text=f"对手已走棋{player_last_move_notation}"))
                lastest_pieces = parse_chinese_board(self.lastest_board)
                # lastest_pieces_str_list = []
                # for coord in sorted(lastest_pieces.keys(), key=lambda x: (int(x[1:]), x[0])):
                #     lastest_pieces_str_list.append(f"{coord}: {lastest_pieces[coord]}")
                # lastest_pieces_str = ", ".join(lastest_pieces_str_list)
                prompt.with_content(
                    Text(text=f"之前回合的棋盘是：\n{self.lastest_board}"),
                )
                eaten_piece = lastest_pieces.get(player_last_move[2:4])
                if eaten_piece:
                    message.with_content(Text(text=f"并吃掉你的{eaten_piece}"))

        best_move_notation = uci_to_chinese_notation(board_str, best_move)
        message.with_content(
            Text(text=f"你当前的最佳走法是{best_move_notation}，最佳走棋不一定是为了应对对方的走棋，可能是更长远的布局，UCI格式为{best_move}")
        )
        eaten_piece = pieces.get(best_move[2:4])
        if eaten_piece:
            message.with_content(Text(text=f"并吃掉对手的{eaten_piece}"))

        prompt.with_content(
            Text(text=f"当前回合的棋盘是：\n{board_str}"),
            Text(text="你需要回应简洁、自信、带有挑衅的锋芒。可以是冷笑、嘲讽、直接的警告或冰冷的宣告")
        )

        await self.eventbus.put(ProgramInputAgentEvent(
            prompt=prompt,
            message=message,
            priority=99,
        ))
        self.lastest_board = board_str


class ChessPlayingStateProvider(Provider[ChessPlayingState]):
    """ChessPlayingState的Provider"""

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> ChessPlayingState:
        mini = con.force_fetch(ReachyMini)
        eventbus = con.force_fetch(EventBus)
        douyin_live = con.force_fetch(DouyinLive)
        logger = con.get(LoggerItf)
        chess_channel = con.force_fetch(ChineseChessChannel)

        return ChessPlayingState(
            mini=mini,
            eventbus=eventbus,
            douyin_live=douyin_live,
            chess_channel=chess_channel,
            logger=logger,
        )