import pytest
from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_moss import PyChannel, new_ctml_shell

from framework.moss_contrib.ctml_repo import CtmlRepo


@pytest.mark.asyncio
async def test_ctml_repo():
    repo = CtmlRepo(storage=MemoryStorage(dir_=""))

    names = repo.list_ctml_names()
    assert names == []

    shell = new_ctml_shell()

    shell.main_channel.build.command()(repo.save_ctml)
    shell.main_channel.build.command(doc=repo.execute_ctml_docstring)(repo.execute_ctml)

    count = 0
    @shell.main_channel.build.command()
    async def foo():
        nonlocal count
        count += 1

    async with shell:
        await shell.wait_connected()
        async with await shell.interpreter() as interpreter:
            interpreter.feed("""<save_ctml name="test"><foo/></save_ctml>""")
            interpreter.commit()
            await interpreter.wait_tasks()
            assert count == 0
        async with await shell.interpreter() as interpreter:
            interpreter.feed("""<execute_ctml name="test" />""")
            interpreter.commit()
            await interpreter.wait_tasks()
            assert count == 1
        async with await shell.interpreter() as interpreter:
            interpreter.feed("""<loop times="3"><execute_ctml name="test" /></loop>""")
            interpreter.commit()
            await interpreter.wait_tasks()
            assert count == 4
