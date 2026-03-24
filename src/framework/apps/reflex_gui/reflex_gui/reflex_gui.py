"""Welcome to Reflex! This file outlines the steps to create a basic app."""
import asyncio

import reflex as rx
from ghoshell_moss import PyChannel
from ghoshell_moss.transports.zmq_channel import ZMQChannelProvider
from reflex_gui.stream_gui_test import index as stream_gui_test_index

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


queue = asyncio.Queue()
provided = asyncio.Event()

emoji_queue = asyncio.Queue()


class State(rx.State):
    """The app state."""
    markdown_content: str = ""
    emoji: str = "love"

    @rx.event(background=True)
    async def refresh_markdown(self):
        """
        只能通过 @rx.event(background=True)
        在 app.add_page 的 on_load 里挂载异步事件
        才可以主动刷新页面
        """
        while True:
            chunk = await queue.get()
            async with self:
                if chunk is None:
                    self.markdown_content = ""
                else:
                    self.markdown_content += chunk
                yield

    @rx.event(background=True)
    async def switch_emoji(self):
        while True:
            emoji = await emoji_queue.get()
            async with self:
                if emoji is None:
                    break
                self.emoji = emoji
                yield

    async def provide_channel(self):
        if provided.is_set():
            return
        provided.set()
        chan = PyChannel(name="gui")

        async def append_markdown(chunks__):
            async for chunk in chunks__:
                await queue.put(chunk)

        async def clear_markdown():
            await queue.put(None)

        async def switch_emoji(emoji: str="love"):
            """
            切换屏幕展示的emoji表情
            现在可以展示的emoji有：
            - angry
            - cry
            - happy
            - laugh
            - love
            - sad
            - surprised

            :param emoji: 需要切换的表情
            :return:
            """
            await emoji_queue.put(emoji)

        chan.build.command()(append_markdown)
        chan.build.command()(clear_markdown)
        chan.build.command()(switch_emoji)

        # 用 arun_until_closed启动的协程 会比 run_in_thread 性能更好 响应更流畅
        provider = ZMQChannelProvider(
            address="tcp://127.0.0.1:9527",
        )
        provider._receive_interval_seconds = 3 # 默认是0.5 容错太低了 超时会导致丢弃事件 改成3秒数据抵达率提高了很多
        asyncio.create_task(provider.arun_until_closed(chan))

def index() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.hstack(
            rx.vstack(
                rx.heading("小灵的屏幕模式", size="6"),
                rx.image(
                    src=f"/{State.emoji}.gif",
                    width="300px",
                    height="auto",
                    alt="Animated GIF using asset function"
                ),
                spacing="5",
            ),
            rx.vstack(
                rx.markdown(State.markdown_content),
                min_height="85vh",
            ),
            spacing="5",
            # justify="center",
        ),
    )

logger.info("start app")
app = rx.App()
app.add_page(index, on_load=[State.provide_channel, State.refresh_markdown, State.switch_emoji])
app.add_page(stream_gui_test_index, route="/test")

