import asyncio
import logging
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
import traceback

try:
    from pynput import keyboard
    HAS_PYNPUT = os.getenv("ENABLE_PYNPUT", "") == "1"
except ImportError:
    HAS_PYNPUT = False
    # 如果需要使用空格键控制，请安装 pynput 库: pip install pynput

import numpy as np
from ghoshell_container import IoCContainer
from ghoshell_moss_contrib.agent.depends import check_agent
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from ghoshell_common.contracts import LoggerItf
from reachy_mini import ReachyMini

from framework.abcd.agent_event import AsrInvokeAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.audio.mic_hub import MicHub

if check_agent():
    from rich.markdown import Markdown
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from framework.listener.async_concepts import AsyncListenerService, AsyncListenerStateName, Recognition


class ConsolePTTChat(BaseChat):
    def __init__(
        self,
        debug: bool = False,
        logger: LoggerItf | None = None,
        eventbus: EventBus=None,
        *,
        container: IoCContainer | None = None,
    ):
        super().__init__()

        self.eventbus = eventbus

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
        self.logger = logger or logging.getLogger("ConsolePTTChat")

        # 异步Listener相关属性
        self.listener_service: Optional[AsyncListenerService] = None
        self._quit_event = asyncio.Event()
        self._input_task: Optional[asyncio.Task] = None

        self._container = container

        self.ai_response_done = asyncio.Event()

        # 键盘监听相关属性
        self.space_pressed = False
        self.keyboard_listener = None
        self._keyboard_quit_event = asyncio.Event()
        self._keyboard_events = asyncio.Queue()

    async def _setup_ptt_mode(self, mini: ReachyMini=None):
        """异步设置PTT（Push-to-Talk）模式，使用空格键控制录音"""
        try:
            # 异步初始化Listener服务
            await self._initialize_listener_service(mini)

            # 检查键盘监听库是否可用
            if not HAS_PYNPUT:
                self.console.print("[yellow]pynput library not available, falling back to Enter key mode.[/yellow]")
                # 回退到原来的Enter键模式
                self._input_task = asyncio.create_task(self._input_loop())
                return

            # 启动键盘监听
            self._input_task = asyncio.create_task(self._keyboard_listener_loop())

        except Exception as e:
            self.console.print(f"[red]Error initializing PTT mode: {e}[/red]")
            self.console.print("[yellow]Falling back to text input mode.[/yellow]")

    async def _initialize_listener_service(self, mini: ReachyMini=None):
        """异步初始化Listener服务"""
        from framework.listener.async_listener_service import AsyncListenerServiceImpl
        from framework.listener.configs import ListenerConfig

        config = ListenerConfig()

        audio_input = None
        if self._container is not None:
            try:
                hub = self._container.force_fetch(MicHub)
                # Dedicated subscription for ASR/PTT. Drop oldest on overflow to avoid blocking capture.
                audio_input = hub.new_audio_input(
                    name="ptt_asr",
                    max_queue=800,
                    drop_policy="drop_oldest",
                )
            except Exception as e:
                self.logger.warning("failed to init MicHub audio input, fallback to PyAudioInput: %s", e)

        # 创建异步Listener服务
        self.listener_service = AsyncListenerServiceImpl(
            config=config,
            logger=self.logger,
            audio_input=audio_input,
        )

        # 设置异步回调
        await self.listener_service.set_callback(self._create_listener_callback())
        # 异步启动服务
        await self.listener_service.bootstrap()

        self.logger.info("AsyncListenerService initialized")

    def _create_listener_callback(self):
        """创建异步Listener回调"""
        from framework.listener.async_concepts import AsyncListenerCallback

        class AsyncConsoleListenerCallback(AsyncListenerCallback):
            def __init__(self, console_chat):
                self.console_chat = console_chat

            async def on_recognition(self, result: Recognition) -> None:
                """异步处理语音识别结果"""
                if result.seq == 0:
                    await self.console_chat.handle_first_seq()
                    self.console_chat.console.print("")
                if result.text and result.text.strip():
                    # 将识别结果作为用户输入处理
                    if result.is_last:
                        self.console_chat.console.print("", end="\r\033[K")
                        self.console_chat.console.print(f"[cyan]Recognized: {result.text}[/cyan]")
                        await self.console_chat.handle_voice_input(result.text)
                    else:
                        self.console_chat.console.print(f"[green]Recognizing: {result.text}[/green]", end='\r')

            async def on_waken(self) -> None:
                """唤醒事件（跳过实现）"""
                self.console_chat.console.print("[yellow]Wake word detected[/yellow]")

            async def on_state_change(self, state: str) -> None:
                """状态变化事件"""
                if self.console_chat.debug:
                    self.console_chat.console.print(f"[blue]State changed to: {state}[/blue]")

            async def on_error(self, error: str) -> None:
                """错误处理"""
                self.console_chat.console.print(f"[red]Listener error: {error}[/red]")

            async def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
                """保存批次（可选实现）"""
                pass

        return AsyncConsoleListenerCallback(self)

    async def _input_loop(self):
        """异步输入循环"""
        from framework.listener.async_concepts import AsyncListenerStateName

        try:
            while not self._quit_event.is_set():
                try:
                    # 获取当前状态
                    state_name = "unknown"
                    if self.listener_service:
                        try:
                            current_state = await self.listener_service.current_state()
                            state_name = current_state.name().value
                        except Exception as e:
                            self.logger.warning(f"Failed to get current state: {e}")

                    # 显示提示信息
                    if state_name == AsyncListenerStateName.PDT_LISTENING.value:
                        prompt = "[bold green]正在录音... (再次按下Enter键结束录音)"
                    else:
                        prompt = f"[bold green]按下Enter键开始录音 (输入q退出) 当前状态: {state_name}"

                    # 异步等待用户输入
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: Prompt.ask(prompt, console=self.console, default="")
                    )

                    if user_input.lower() == "q":
                        self.console.print("[bold red]关闭中....[/bold red]")
                        self._quit_event.set()
                        break

                    # 处理Enter键操作
                    await self._handle_enter_key_operation()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if not self._quit_event.is_set():
                        self.console.print(f"[red]Input error: {e}[/red]")
                        await asyncio.sleep(0.5)  # 避免频繁错误循环

        except asyncio.CancelledError:
            self.logger.info("Input loop cancelled")
        except Exception as e:
            self.logger.exception(f"Error in input loop: {e}")

    async def _keyboard_listener_loop(self):
        """异步键盘监听循环，监听空格键按下和释放"""
        if not HAS_PYNPUT:
            self.logger.error("pynput not available, cannot start keyboard listener")
            return

        loop = asyncio.get_event_loop()

        def on_press(key):
            """键盘按下回调"""
            try:
                if key == keyboard.Key.space and not self.space_pressed:
                    self.space_pressed = True
                    # 将按下事件放入队列
                    asyncio.run_coroutine_threadsafe(
                        self._handle_space_key_press(),
                        loop
                    )
                elif key == keyboard.Key.enter:
                    # 处理回车键打断
                    asyncio.run_coroutine_threadsafe(
                        self._handle_enter_key_operation(),
                        loop
                    )
            except Exception as e:
                self.logger.error(f"Error in on_press: {e}")

        def on_release(key):
            """键盘释放回调"""
            try:
                if key == keyboard.Key.space and self.space_pressed:
                    self.space_pressed = False
                    # 将释放事件放入队列
                    asyncio.run_coroutine_threadsafe(
                        self._handle_space_key_release(),
                        loop
                    )
                # 检测 'q' 键退出
                if key == keyboard.KeyCode.from_char('q'):
                    self.console.print("[bold red]关闭中....[/bold red]")
                    self._quit_event.set()
                    return False  # 停止监听器
            except Exception as e:
                self.logger.error(f"Error in on_release: {e}")
            return True

        # 启动键盘监听器
        self.keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.keyboard_listener.start()
        self.console.print("[green]键盘监听已启动，长按空格说话，松开空格提交，按 '回车键' 打断输出，按 'q' 退出。[/green]")

        # 等待退出事件
        try:
            await self._quit_event.wait()
        except asyncio.CancelledError:
            self.logger.info("Keyboard listener loop cancelled")
        finally:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None

    async def _handle_space_key_press(self):
        """处理空格键按下事件（开始录音）"""
        from framework.listener.async_concepts import AsyncListenerStateName

        if not self.listener_service:
            return

        # 检查当前状态
        try:
            current_state = await self.listener_service.current_state()
            state_name = current_state.name().value
        except Exception as e:
            self.logger.warning(f"Failed to get current state: {e}")
            return

        # 如果当前不在录音状态，则开始录音
        if state_name != AsyncListenerStateName.PDT_LISTENING.value:
            self.console.print("[green]开始录音...[/green]")
            await self.listener_service.set_state(AsyncListenerStateName.PDT_LISTENING.value)
            # 等待状态切换完成
            for _ in range(10):  # 最多等待1秒
                await asyncio.sleep(0.1)
                try:
                    current = await self.listener_service.current_state()
                    if current.name() == AsyncListenerStateName.PDT_LISTENING:
                        break
                except Exception:
                    pass

    async def _handle_space_key_release(self):
        """处理空格键释放事件（结束录音并提交）"""
        from framework.listener.async_concepts import AsyncListenerStateName

        if not self.listener_service:
            return

        # 检查当前状态
        try:
            current_state = await self.listener_service.current_state()
            state_name = current_state.name().value
        except Exception as e:
            self.logger.warning(f"Failed to get current state: {e}")
            return

        # 如果当前在录音状态，则结束录音并提交
        if state_name == AsyncListenerStateName.PDT_LISTENING.value:
            self.console.print("[yellow]结束录音并提交识别...[/yellow]")
            await self.listener_service.commit()
            # 等待状态切换完成（回到等待状态）
            for _ in range(10):  # 最多等待1秒
                await asyncio.sleep(0.1)
                try:
                    current = await self.listener_service.current_state()
                    if current.name() == AsyncListenerStateName.PDT_WAITING:
                        break
                except Exception:
                    pass

    async def _handle_enter_key_operation(self):
        """异步处理Enter键操作"""
        from framework.listener.async_concepts import AsyncListenerStateName

        if not self.listener_service:
            return

        # 在pynput模式下，回车键仅用于打断流式输出，不用于控制录音
        if self.keyboard_listener and not self.is_streaming:
            # pynput模式已激活且不在流式输出状态，回车键不执行任何操作
            return

        if self.is_streaming:
            self.interrupted = True
            self.is_streaming = False
            await self.listener_service.clear_buffer()
            await self.listener_service.set_state(AsyncListenerStateName.PDT_WAITING.value)
            if self.on_interrupt_callback:
                self.on_interrupt_callback()
                self.console.print("\n[yellow]Output interrupted[/yellow]\n")
        else:
            try:
                current_state = await self.listener_service.current_state()
                state_name = current_state.name().value

                if state_name == AsyncListenerStateName.PDT_LISTENING.value:
                    # 松开按键效果 - 结束录音
                    self.console.print("[yellow]结束录音并提交识别...[/yellow]")
                    await self.listener_service.commit()

                    # 等待状态切换完成（回到等待状态）
                    for _ in range(10):  # 最多等待1秒
                        await asyncio.sleep(0.1)
                        try:
                            current = await self.listener_service.current_state()
                            if current.name() == AsyncListenerStateName.PDT_WAITING:
                                break
                        except Exception:
                            pass
                else:
                    # 按下按键效果 - 开始录音
                    self.console.print("[green]开始录音...[/green]")
                    await self.listener_service.set_state(AsyncListenerStateName.PDT_LISTENING.value)

                    # 等待状态切换完成
                    for _ in range(10):  # 最多等待1秒
                        await asyncio.sleep(0.1)
                        try:
                            current = await self.listener_service.current_state()
                            if current.name() == AsyncListenerStateName.PDT_LISTENING:
                                break
                        except Exception:
                            pass
            except Exception as e:
                self.console.print(f"[red]Error handling enter key: {e}[/red]")
                self.logger.exception(f"Error handling enter key: {e}")

    async def handle_voice_input(self, text: str):
        """异步处理语音输入"""
        # 添加用户消息到历史记录
        if text:
            self.add_user_message(text)

        # 调用输入处理回调
        if text and self.on_input_callback:
            # 注意：on_input_callback 可能是同步的，需要适配
            if asyncio.iscoroutinefunction(self.on_input_callback):
                await self.on_input_callback(text)
            else:
                # 在 executor 中运行同步回调
                await asyncio.get_event_loop().run_in_executor(
                    None, self.on_input_callback, text
                )

    def _print_startup_info(self):
        """打印启动信息"""
        self.console.print("=== Chat Started ===")

        self.console.print(Panel(
            Markdown("""
当前是 PTT（Push-to-Talk）模式，通过空格键控制录音.
操作方法:
1. **长按空格键**：开始录音 (切换到 pdt_listening 状态)
2. **松开空格键**：结束录音并提交给 Agent 处理
3. 按 `回车键`：打断 AI 的流式输出
4. 按 `q` 键：退出程序
"""),
            title="PTT (Space Key Control)",
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
        self.console.print("> 长按空格说话，松开空格提交，按回车键打断输出: ")

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
        """异步运行聊天界面主循环"""
        try:
            # 打印启动信息
            self._print_startup_info()
            # 异步设置 PTT（空格键控制）模式
            await self._setup_ptt_mode()
            # 运行主循环
            await self._run_main_loop()
        except Exception as e:
            self.print_exception(e, "Error in async chat")
        finally:
            # 异步清理资源
            await self._cleanup()


    async def _run_main_loop(self):
        """异步主循环"""
        try:
            # 等待退出事件
            await self._quit_event.wait()
            self.console.print("[yellow]Exiting async chat...[/yellow]")

        except asyncio.CancelledError:
            self.logger.info("Main loop cancelled")
        except Exception as e:
            self.print_exception(e, "Error in main loop")

    async def _cleanup(self):
        """异步清理资源"""
        self.is_streaming = False
        self.interrupted = False
        self._quit_event.set()

        # 停止键盘监听器
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

        # 取消输入任务
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
            try:
                await self._input_task
            except asyncio.CancelledError:
                pass

        # 关闭异步Listener服务
        if self.listener_service:
            try:
                await self.listener_service.shutdown()
            except Exception as e:
                self.console.print(f"[red]Error shutting down listener: {e}[/red]")

    async def handle_first_seq(self):
        """异步处理第一个序列（发送ASR调用事件）"""
        if not self.eventbus:
            return

        try:
            await self.eventbus.put(
                AsrInvokeAgentEvent(
                    priority=99,
                )
            )
        except Exception as e:
            self.logger.exception(f"Error sending ASR invoke event: {e}")
