import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import traceback
import threading

from ghoshell_moss_contrib.agent.depends import check_agent
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from ghoshell_common.contracts import LoggerItf
from reachy_mini import ReachyMini

if check_agent():
    from rich.markdown import Markdown
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from moss_in_reachy_mini.listener.concepts.listener import ListenerService, ListenerStateName


class ConsolePTTChat(BaseChat):
    def __init__(self, debug: bool = False, logger: LoggerItf | None = None, mini: ReachyMini = None):
        super().__init__()
        # 存储完整的对话历史
        self.conversation_history: List[Dict] = []

        # 当前正在处理的AI回复
        self.current_ai_response: Optional[str] = None

        # 标记是否正在流式输出
        self.is_streaming = False

        # 标记是否被用户中断
        self.interrupted = False

        # Rich控制台
        self.console = Console()

        # 工作模式
        self.debug = debug
        self.logger = logger

        # Listener相关属性
        self.listener_service: Optional[ListenerService] = None
        self._quit_event = threading.Event()
        self._input_thread: Optional[threading.Thread] = None

        # 打印启动信息
        self._print_startup_info()

        self._setup_enter_to_talk_mode(mini)
        self.ai_response_done = asyncio.Event()

    def _setup_enter_to_talk_mode(self, mini: ReachyMini=None):
        """设置Enter to Talk模式"""
        try:
            # 初始化Listener服务
            self._initialize_listener_service(mini)
            # 启动输入监听线程
            self._start_input_listener()
        except Exception as e:
            self.console.print(f"[red]Error initializing Enter to Talk mode: {e}[/red]")
            self.console.print("[yellow]Falling back to text input mode.[/yellow]")

    def _initialize_listener_service(self, mini: ReachyMini=None):
        """初始化Listener服务"""
        # 这里需要根据实际的依赖注入系统来获取ListenerService
        # 暂时使用一个简化的实现
        from moss_in_reachy_mini.listener.lisenter_impl import ListenerServiceImpl
        from moss_in_reachy_mini.listener.configs import ListenerConfig

        # 创建简单的logger实现

        config = ListenerConfig()

        self.listener_service = ListenerServiceImpl(
            config=config,
            logger=self.logger,
            # audio_input=ReachyMiniInput(mini=mini),
        )

        # 设置回调
        self.listener_service.set_callback(self._create_listener_callback())
        self.listener_service.bootstrap()

    def _create_listener_callback(self):
        """创建Listener回调"""
        from moss_in_reachy_mini.listener.concepts.listener import ListenerCallback, Recognition

        class ConsoleListenerCallback(ListenerCallback):
            def __init__(self, console_chat):
                self.console_chat = console_chat

            def on_recognition(self, result: Recognition) -> None:
                """处理语音识别结果"""
                if result.text and result.text.strip():
                    self.console_chat.console.print(f"[green]Recognizing: {result.text}[/green]")
                    # 将识别结果作为用户输入处理
                    if result.is_last:
                        self.console_chat.console.print(f"[cyan]Recognized: {result.text}[/cyan]")
                        self.console_chat.handle_voice_input(result.text)

            def on_waken(self) -> None:
                """唤醒事件"""
                self.console_chat.console.print("[yellow]Wake word detected[/yellow]")

            def on_state_change(self, state: str) -> None:
                """状态变化事件"""
                if self.console_chat.debug:
                    self.console_chat.console.print(f"[blue]State changed to: {state}[/blue]")

            def on_error(self, error: str) -> None:
                """错误处理"""
                self.console_chat.console.print(f"[red]Listener error: {error}[/red]")

        return ConsoleListenerCallback(self)

    def _start_input_listener(self):
        """启动输入监听线程"""

        def input_loop():
            while not self._quit_event.is_set():
                try:
                    # 显示提示信息
                    state = self.listener_service.current_state().name() if self.listener_service else "unknown"

                    if state == ListenerStateName.pdt_listening.value or state == ListenerStateName.listening.value:
                        prompt = "[bold green]正在录音... (再次按下Enter键结束录音)"
                    else:
                        prompt = f"[bold green]按下Enter键开始录音 (输入q退出){state}"

                    user_input = Prompt.ask(prompt, console=self.console, default="")

                    if user_input.lower() == "q":
                        self.console.print("[bold red]关闭中....[/bold red]")
                        self._quit_event.set()
                        break

                    # 处理Enter键操作
                    self._handle_enter_key_operation()

                except Exception as e:
                    if not self._quit_event.is_set():
                        self.console.print(f"[red]Input error: {e}[/red]")

        self._input_thread = threading.Thread(target=input_loop, daemon=True)
        self._input_thread.start()

    def _handle_enter_key_operation(self):
        """处理Enter键操作"""
        if not self.listener_service:
            return

        if self.is_streaming:
            self.interrupted = True
            if self.on_interrupt_callback:
                self.on_interrupt_callback()
                self.console.print("\n[yellow]Output interrupted[/yellow]\n")
        else:
            current_state = self.listener_service.current_state().name()
            if current_state == ListenerStateName.pdt_listening.value:
                # 松开按键效果 - 结束录音
                self.console.print("[yellow]结束录音并提交识别...[/yellow]")
                self.listener_service.commit()
            else:
                # 按下按键效果 - 开始录音
                self.console.print("[green]开始录音...[/green]")
                self.handle_voice_input("")
                self.listener_service.set_state(ListenerStateName.pdt_listening.value)

    def handle_voice_input(self, text: str):
        """处理语音输入"""
        # 添加用户消息到历史记录
        if text:
            self.add_user_message(text)

        # 调用输入处理回调
        if self.on_input_callback:
            self.on_input_callback(text)

    def _print_startup_info(self):
        """打印启动信息"""
        self.console.print("=== Chat Started ===")

        self.console.print(Panel(
            Markdown(""" 
当前是 Enter to Talk 模式，通过 Enter 键控制录音. 
操作方法: 
1. **按下 Enter 键**：开始录音 (切换到 pdt_listening 状态) 
2. **再次按下 Enter 键**：结束录音并提交给 Agent 处理 
3. 输入 `q` 然后回车：退出程序 
"""),
            title="Enter to Talk (Enter Key Control)",
        ))

    def add_user_message(self, message: str):
        """添加用户消息到历史记录"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        input_type = "Voice"
        self.console.print(f"\n\n[green][{timestamp}] User ({input_type}): {message}[/green]")
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": timestamp,
            "input_type": input_type.lower()
        })

    def start_ai_response(self):
        """开始AI回复"""
        self.ai_response_done.clear()
        self.current_ai_response = ""
        self.is_streaming = True
        self.interrupted = False
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"\n\n[white][{timestamp}] AI: [/white]")

    def update_ai_response(self, chunk: str, is_thinking: bool = False):
        """更新AI的流式回复"""
        if self.interrupted:
            return False
        if not self.is_streaming:
            self.start_ai_response()

        self.current_ai_response += chunk

        # 根据 is_thinking 参数选择颜色
        if is_thinking:
            self.console.print(f"[grey50]{chunk}[/grey50]", end="")
        else:
            self.console.print(f"[white]{chunk}[/white]", end="")

        # 检查是否被中断
        if self.interrupted:
            return False

        return True

    def finalize_ai_response(self):
        """完成AI回复"""
        if self.current_ai_response:
            timestamp = datetime.now().strftime("%H:%M:%S")

            # 添加换行
            self.console.print()

            # 保存到历史记录
            self.conversation_history.append({
                "role": "assistant",
                "content": self.current_ai_response,
                "timestamp": timestamp
            })

        self.console.print("\n")
        self.current_ai_response = None
        self.is_streaming = False
        self.interrupted = False
        self.ai_response_done.set()

        # 根据模式显示不同的提示
        self.console.print("> 按下Enter键开始录音: ")

    def print_exception(self, exception: Any, context: str = ""):
        """打印异常信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 格式化异常信息
        if isinstance(exception, Exception):
            exc_info = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
            error_msg = "".join(exc_info)
        else:
            error_msg = str(exception)

        # 添加上下文信息
        if context:
            error_msg = f"[{context}]\n{error_msg}"

        # 打印错误信息（红色）
        self.console.print(f"[red][{timestamp}] ERROR: {error_msg}[/red]")

    async def run(self):
        """运行聊天界面主循环"""
        try:
            await self._run_enter_to_talk_mode()
        finally:
            # 清理资源
            self._cleanup()

    async def _run_enter_to_talk_mode(self):
        """运行Enter to Talk模式"""
        try:
            # 等待退出事件
            while not self._quit_event.is_set():
                await asyncio.sleep(0.1)

            self.console.print("[yellow]Exiting Enter to Talk mode...[/yellow]")

        except Exception as e:
            self.print_exception(e, "Error in Enter to Talk mode")

    def _cleanup(self):
        """清理资源"""
        self.is_streaming = False
        self.interrupted = False
        self._quit_event.set()

        # 关闭Listener服务
        if self.listener_service:
            try:
                self.listener_service.shutdown()
            except Exception as e:
                self.console.print(f"[red]Error shutting down listener: {e}[/red]")

        # 等待输入线程结束
        if self._input_thread and self._input_thread.is_alive():
            self._input_thread.join(timeout=5.0)
