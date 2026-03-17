import asyncio

from ghoshell_moss import PyChannel, new_ctml_shell
from ghoshell_moss.transports.zmq_channel import ZMQChannelProvider, ZMQChannelProxy


async def append_markdown(chunks__):
    async for chunk in chunks__:
        print(chunk)


async def main():
    print("haha")
    chan = PyChannel(name="test")
    chan.build.command()(append_markdown)

    provider =ZMQChannelProvider(
        address="tcp://127.0.0.1:9527",
    )
    await provider.arun_until_closed(chan)


async def run_proxy():
    proxy = ZMQChannelProxy(
        name="test",
        address="tcp://127.0.0.1:9527",
    )
    async with proxy.bootstrap() as proxy_runtime:
        await proxy_runtime.wait_connected()
        cmd = proxy_runtime.get_command("clear_markdown")
        await cmd()

async def run_shell():
    proxy = ZMQChannelProxy(
        name="reflex",
        address="tcp://127.0.0.1:9527",
        recv_timeout=3,
        send_timeout=3,
    )
    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)
    async with shell:
        await shell.wait_connected("reflex")
        async with shell.interpreter_in_ctx() as interpreter:
            lines = [
                "# 实时生成的内容\n",
                "## 第二部分\n",
                "这是逐步添加的文本 1。\n",
                "这是逐步添加的文本 2。\n",
                "这是逐步添加的文本 3。\n",
                "这是逐步添加的文本 4。\n",
                "- 项目 1\n",
                "- 项目 2\n",
                "```python\nprint('done')\n```\n",
            ]
            interpreter.feed("<reflex:append_markdown>")
            for line in lines:
                await asyncio.sleep(0.5)
                interpreter.feed(line)
            interpreter.feed("</reflex:append_markdown>")

            # await asyncio.sleep(2)
            # interpreter.feed("<reflex:clear_markdown/>")

            interpreter.commit()
            await interpreter.wait_tasks()

if __name__ == '__main__':
    asyncio.run(run_shell())
