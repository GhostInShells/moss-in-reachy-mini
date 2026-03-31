import asyncio
import os
from typing import Tuple


class AsyncPikafish:
    def __init__(self, engine_path: str=None):
        if not engine_path:
            file_abs_path = os.path.abspath(__file__)
            engine_dir = os.path.dirname(file_abs_path)
            engine_path = os.path.join(engine_dir, "pikafish")

        self.engine_path = engine_path
        # 中国象棋固定初始局面（永远不用改）
        self.INIT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        self.process = None
        self.reader = None
        self.writer = None

    async def start(self):
        """异步启动引擎"""
        self.process = await asyncio.create_subprocess_exec(
            self.engine_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.reader = self.process.stdout
        self.writer = self.process.stdin
        # 初始化 UCI 协议
        await self._init_uci()

    async def _init_uci(self):
        """标准UCI初始化流程"""
        await self._send("uci")
        # 等待 uciok
        while True:
            line = await self._read_line()
            if "uciok" in line:
                break

        await self._send("isready")
        while True:
            line = await self._read_line()
            if "readyok" in line:
                break

        await self._send("ucinewgame")

    async def _send(self, cmd: str):
        """异步发送命令"""
        self.writer.write(f"{cmd}\n".encode("utf-8"))
        await self.writer.drain()

    async def _read_line(self) -> str:
        """异步读取一行输出"""
        line = await self.reader.readline()
        return line.decode("utf-8")

    # ===================== 核心异步函数：传入走法列表 → 获取最佳走法 =====================
    async def get_best_move(self, moves_history: list[str], depth: int = 10) -> Tuple[str, str]:
        """
        :param moves_history: 历史走法列表，如 ["c3c4", "g6g5"]
        :param depth: 计算深度，默认10
        :return: 引擎返回的 bestmove
        """
        # 拼接：初始局面 + 全部历史走法 → 引擎自动推导当前棋盘
        moves_str = " ".join(moves_history)
        await self._send(f"position fen {self.INIT_FEN} moves {moves_str}")

        # 异步计算
        await self._send(f"go depth {depth}")

        # 获取当前棋盘字符画
        await self._send("d")

        lines = []
        best_move = ""

        # 循环读取直到获取最佳走法
        while True:
            try:
                line = await asyncio.wait_for(self._read_line(), timeout=0.2)
                line_strip = line.strip()
                if line_strip.startswith("+") or line_strip.startswith("|") :
                    line = line.replace(" K ", "红帅")
                    line = line.replace(" A ", "红仕")
                    line = line.replace(" B ", "红相")
                    line = line.replace(" N ", "红马")
                    line = line.replace(" R ", "红车")
                    line = line.replace(" C ", "红炮")
                    line = line.replace(" P ", "红兵")
                    line = line.replace(" k ", "黑将")
                    line = line.replace(" a ", "黑士")
                    line = line.replace(" b ", "黑象")
                    line = line.replace(" n ", "黑马")
                    line = line.replace(" r ", "黑车")
                    line = line.replace(" c ", "黑炮")
                    line = line.replace(" p ", "黑卒")
                    lines.append(line)
                if line_strip.startswith("a"):
                    lines.append(line)
                if line.startswith("bestmove"):
                    best_move = line.split()[1]
            except asyncio.TimeoutError:
                break

        return best_move, "".join(lines)

    async def close(self):
        """关闭引擎"""
        if self.writer:
            await self._send("quit")
            self.writer.close()
            await self.writer.wait_closed()
        if self.process:
            await self.process.wait()

# ====================== 异步使用演示 ======================
async def main():
    # 1. 替换成你的 Pikafish 路径
    file_abs_path = os.path.abspath(__file__)
    engine_dir = os.path.dirname(file_abs_path)

    ENGINE_PATH = os.path.join(engine_dir, "pikafish")

    # 启动异步引擎
    engine = AsyncPikafish(ENGINE_PATH)
    await engine.start()

    # ✅ 唯一需要维护的：空列表记录走法（不用管棋盘！）
    moves = []

    # 第1步：红方走棋（异步调用，非阻塞）
    best, board = await engine.get_best_move(moves)
    print("红方最佳走法:", best, "\n", board)
    moves.append(best)

    # 第2步：黑方走棋
    best, board = await engine.get_best_move(moves)
    print("黑方最佳走法:", best, "\n", board)
    moves.append(best)

    # 第3步：继续对弈
    best, board = await engine.get_best_move(moves)
    print("红方下一步:", best, "\n", board)
    moves.append(best)

    # 关闭引擎
    await engine.close()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())