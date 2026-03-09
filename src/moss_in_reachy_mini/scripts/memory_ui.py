import pathlib

from ghoshell_common.contracts import Workspace
from ghoshell_moss_contrib.example_ws import workspace_container

from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.memory import StorageMemoryUI


def main(storage_name: str="memory"):
    ws_dir = pathlib.Path(__file__).parent.joinpath(".workspace")
    with workspace_container(ws_dir) as container:
        ws = container.force_fetch(Workspace)
        storage = ws.runtime().sub_storage(storage_name)
        _storage_memory = StorageMemory(storage)
        # 启动UI
        ui = StorageMemoryUI(_storage_memory)
        ui.render()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=str, default="memory")

    args = parser.parse_args()
    main(args.storage)

