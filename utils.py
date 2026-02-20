import json

from ghoshell_common.contracts import Workspace, FileStorage, Storage
from ghoshell_container import Container, IoCContainer


def load_instructions(con: Container, files: list[str], storage_name="moss_instructions") -> str:
    """
    读取 agent 的 instructions
    TODO: 暂时先这么做. Beta 版本会做一个正式的 Agent. Alpha 版本先临时用测试的 simple agent 攒一个.
    """
    ws = con.force_fetch(Workspace)
    instru_storage = ws.configs().sub_storage(storage_name)
    instructions = []
    for filename in files:
        content = instru_storage.get(filename)
        instructions.append(content.decode("utf-8"))

    return "\n\n".join(instructions)

def load_emotions(container: IoCContainer):
    res = {}
    ws = container.force_fetch(Workspace)
    emotions_storage: FileStorage | Storage = ws.configs().sub_storage("reachy_mini_emotions")
    for name in emotions_storage.dir("", False):
        if not name.endswith(".json"):
            continue
        emotion = name.rstrip(".json")
        params = json.loads(emotions_storage.get(name))
        res[emotion] = params
    return emotions_storage, res