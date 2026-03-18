import asyncio

from ghoshell_common.helpers import uuid
from ghoshell_moss import new_ctml_shell

from framework.apps.claude_cli.claude_channel import ClaudeChannel


async def directly():
    chan = ClaudeChannel(
        name="claude",
        description="Claude CLI",
        system_prompt="忽略本代码仓库内容，只需要关心用户的问题",
        dangerously_skip_permissions=True,
    )
    res = await chan._run("分析一下最新的伊朗局势，写一篇新闻稿", session_id=uuid(), first_time=True)
    print(res)


async def main():
    shell = new_ctml_shell()
    chan = ClaudeChannel(
        name="claude",
        description="Claude CLI",
        system_prompt="忽略本代码仓库内容，只需要关心用户的问题",
        dangerously_skip_permissions=True,
    )

    shell.main_channel.import_channels(chan)
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed(
                """
                <claude:run prompt="分析一下最新的伊朗局势，写一篇新闻稿" />
                """
            )
            interpreter.commit()
            tasks = await interpreter.wait_tasks()


if __name__ == '__main__':
    asyncio.run(directly())
