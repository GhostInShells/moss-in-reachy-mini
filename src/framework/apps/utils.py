from datetime import datetime
import enum
import json

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss_contrib.agent import ConsoleChat
from rich.layout import Layout


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        # 如果是Enum实例，返回其value
        if isinstance(obj, enum.Enum):
            return obj.value
        # 其他类型按默认逻辑处理
        return super().default(obj)


class AgentConsoleChat(ConsoleChat):
    def __init__(self, agent_id, logger=LoggerItf):
        super().__init__(logger)
        self.agent_id = agent_id

    def start_ai_response(self):
        """开始AI回复"""
        self.ai_response_done.clear()
        self.current_ai_response = ""
        self.is_streaming = True
        self.interrupted = False
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"\n\n[white][{timestamp}] {self.agent_id} AI: [/white]")

    def finalize_ai_response(self):
        """完成AI回复"""
        if self.current_ai_response:
            timestamp = datetime.now().strftime("%H:%M:%S")

            # 添加换行
            self.console.print()

            # 保存到历史记录
            self.conversation_history.append(
                {"role": "assistant", "content": self.current_ai_response, "timestamp": timestamp}
            )

            # 如果rich可用且没有被中断，添加Markdown渲染
            if not self.interrupted:
                # self._add_markdown_rendering(self.current_ai_response)
                pass

        self.console.print("\n")
        self.current_ai_response = None
        self.is_streaming = False
        self.interrupted = False
        self.ai_response_done.set()
