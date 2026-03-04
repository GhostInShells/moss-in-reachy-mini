# ReachyMini
## 启动
```commandline
source .venv/bin/activate
uv sync --all-extras
python src/moss_in_reachy_mini/main.py
```

## 记忆和人格

存储位置：`src/moss_in_reach_mini/.workspace/runtime/memory`

你可以修改这个文件夹下的文件内容来手动更改其人格和记忆，也可以在对话中由小灵或者你要求她来完成自己的人格和记忆的更新。

- `.personality.md`是基础人格（可人为修改）
- `.behavior_preference.md`是行为偏好（可人为修改）
- `.mood_base.md`是情绪基地（可人为修改）
- `.autobiographical_memory.md`是自传记忆（可人为修改）
- `.summary_memory.md`是交互历史摘要（可人为修改）
- `thread_xxxxxx.json`是当前会话的所有上下文（不可人为修改），可以在对话中要求创建新的会话；当前不支持切换会话，后续可以提供一个UI页面来做更fancy的呈现。

### 启动UI页面
```commandline
source src/moss_in_reachy_mini/start_memory_ui.sh
```

## 视觉

### 人脸识别
#### 手动录入人脸
在`src/moss_in_reachy_mini/.workspace/runtime/vision/faces`目录下创建文件夹，文件夹名字为人名，且只能用英文。

文件夹下导入图片，支持 .jpg .png .jpeg格式图片

示例
```
- .workspace/runtime/vision/faces
    - wangshiqi
        - img_1.jpg
        - img_2.png
        _ img_3.jpeg
    - xxx
        - xxx_1.jpg
```

运行生成人脸向量
```commandline
python scripts/train_face.py
```

## 直播
### 直播配置文件
[配置文件](.workspace/configs/douyin_live/douyin_live_config.yaml)

### 开启直播
.env文件中添加以下内容
```
REACHY_MINI_MODE="live"
REACHY_MINI_MEMORY_STORAGE="live_memory" # 直播模式下的记忆存储位置
```
运行main.py即可开启直播模式

