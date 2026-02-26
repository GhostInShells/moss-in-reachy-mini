# 记忆和人格

## 说明
存储位置：`src/moss_in_reach_mini/.workspace/runtime/memory`

你可以修改这个文件夹下的文件内容来手动更改其人格和记忆，也可以在对话中由小灵或者你要求她来完成自己的人格和记忆的更新。

- `.personality.md`是基础人格（可人为修改）
- `.behavior_preference.md`是行为偏好（可人为修改）
- `.mood_base.md`是情绪基地（可人为修改）
- `.autobiographical_memory.md`是自传记忆（可人为修改）
- `.summary_memory.md`是交互历史摘要（可人为修改）
- `thread_xxxxxx.json`是当前会话的所有上下文（不可人为修改），可以在对话中要求创建新的会话；当前不支持切换会话，后续可以提供一个UI页面来做更fancy的呈现。

## 启动UI页面
进入正确的虚拟环境并保证依赖安装到最新
```commandline
uv venv
source .venv/bin/activate
uv sync --all-extras
```
先启动reachy mini
```commandline
python src/moss_in_reachy_mini/main.py
```
再启动UI进程
```commandline

streamlit run src/framework/memory/storage_memory_streamlit.py
```