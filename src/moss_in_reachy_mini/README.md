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
- `.consciousness_memory.md`是意识记忆（可人为修改）
- `thread_xxxxxx.json`是当前会话的所有上下文（不可人为修改），可以在对话中要求创建新的会话；当前不支持切换会话，后续可以提供一个UI页面来做更fancy的呈现。

## 视觉
### 人脸识别

Reachy Mini 通过摄像头实时检测和识别人脸（基于 InsightFace buffalo_l 模型），支持以下能力：

- **实时人脸检测**：持续检测画面中的人脸，未识别的人标记为"陌生人"
- **已知人脸识别**：通过 512 维特征向量的余弦相似度匹配，实时识别已录入的用户并叫出名字
- **人脸跟随**：识别到人脸后，头部舵机自动跟随目标人脸转动

### 自动人脸录入

无需手动上传图片。对 Reachy Mini 说"记住我，我叫XX"即可启动全自动录入流程：

1. 机器人引导用户摆出三个姿势（正面、左侧、右侧），自动从摄像头拍摄三张照片
2. 系统自动提取人脸特征向量并训练
3. 训练完成后自动验证识别效果（"让我看看能不能认出你"）
4. 验证通过后回到正常状态，开始跟随新录入的用户

整个过程由语音引导，用户只需要配合转头即可。

### 手动录入人脸（备用方式）

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

## 音乐与舞蹈

详细文档见 [music_and_dance.md](../../../music_and_dance.md)

### 语音点歌

用自然语言点歌，无需 App 操作：
- **精确点歌**："播放周杰伦的晴天"
- **模糊推荐**："我今天心情不太好，放点治愈的歌" → AI 自主推荐具体歌曲，边说推荐理由边播放
- **搜索选歌**："搜一下五月天的歌" → 返回列表让用户挑选

### 连续播放（AI DJ 模式）

说"播一组适合晴天的歌，每首30秒"，Reachy Mini 变身 AI DJ：
- AI 根据心情/场景/风格持续自主选歌
- 每首歌播完自动切到下一首，歌间用一两句话自然点评
- 后台预下载下一首，切歌无延迟
- 随时语音打断，所有旧状态干净清空

### AI 自主编舞

每首歌播放时，系统将歌曲的 BPM、时长、节拍点发送给 AI，AI 实时编排舞蹈动作序列：
- 20 种预设舞蹈 + 80 余种表情动作 + 头部律动 + 天线摆动
- 风格匹配：快歌激烈甩头，慢歌优雅摇摆
- 高潮段密集动作，安静段留白停顿

### 节拍对齐（Beat Sync）

用 librosa 分析音频提取精确节拍时间点，每个动作执行前自动等待到最近的节拍边界，所有动作都踩在拍子上。

### 播放控制

| 指令 | 说明 |
|------|------|
| "播放XX" | 搜索并播放 |
| "暂停音乐" | 暂停 |
| "继续播放" | 恢复播放并重新编排剩余时长的舞蹈 |
| "停止音乐" | 完全停止，清空所有状态 |

## 直播
### 直播配置文件（目前只支持抖音直播）
[配置文件](.workspace/configs/douyin_live/douyin_live_config.yaml)

### 开启直播
.env文件中添加以下内容
```
# 模式切换
REACHY_MINI_MODE="live"
# 隔离记忆
REACHY_MINI_MEMORY="live_memory"
```
运行main.py即可开启直播模式


## Slide Studio

第一步：启动Slide Studio
```
python src/moss_in_reachy_mini/zmq_channels/slide.py
```

第二步：启动Reachy Mini
```
python src/moss_in_reachy_mini/main.py
```

Slide Studio Assets位置
```
.workspace/assets/slide_studio
```

每个幻灯片是一个文件夹，文件夹下通过图片和每张图片的.md后缀文件来组织内容

.md格式说明
```
---
title: ...
outline: ...
---

# 正文

```
- title和outline组成了幻灯片的标题和大纲，当选中一个幻灯片播放时，AI可以看到每一页的标题和大纲。
- 图片和正文组成了单页幻灯片的内容，当播放到某一页时，AI可以看到该图片和正文的内容。

