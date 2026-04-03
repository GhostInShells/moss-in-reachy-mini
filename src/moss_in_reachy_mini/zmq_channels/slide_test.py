import asyncio

from ghoshell_moss import new_ctml_shell
from ghoshell_moss.transports.zmq_channel import ZMQChannelProxy


async def run_shell():
    proxy = ZMQChannelProxy(
        name="slide",
        address="tcp://127.0.0.1:6666",
        recv_timeout=3,
        send_timeout=3,
    )
    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)
    async with shell:
        await shell.wait_connected("slide")
        async with shell.interpreter_in_ctx() as interpreter:
            # interpreter.feed("<slide:show/>")
            interpreter.feed("<slide:play name='蒹葭'/>")
            interpreter.commit()
            await interpreter.wait_tasks()

if __name__ == '__main__':
    asyncio.run(run_shell())
