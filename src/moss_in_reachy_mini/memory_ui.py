import pathlib

from ghoshell_moss_contrib.example_ws import workspace_container

from framework.memory.storage_memory import new_ws_storage_memory
from framework.memory.storage_memory_streamlit import StorageMemoryUI


def main():
    ws_dir = pathlib.Path(__file__).parent.joinpath(".workspace")
    with workspace_container(ws_dir) as container:
        _storage_memory = new_ws_storage_memory(container)
        # 启动UI
        ui = StorageMemoryUI(_storage_memory)
        ui.render()

if __name__ == "__main__":
    main()
