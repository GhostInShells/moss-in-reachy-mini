import pathlib

from ghoshell_common.contracts import Workspace
from ghoshell_moss_contrib.example_ws import workspace_container

from framework.memory.storage_memory import StorageMemory
from framework.memory.storage_memory_streamlit import StorageMemoryUI


def main():
    ws_dir = pathlib.Path(__file__).parent.joinpath(".workspace")
    with workspace_container(ws_dir) as container:
        ws = container.force_fetch(Workspace)
        storage = ws.runtime().sub_storage("memory")
        _storage_memory = StorageMemory(storage)
        # 启动UI
        ui = StorageMemoryUI(_storage_memory)
        ui.render()

if __name__ == "__main__":
    main()
