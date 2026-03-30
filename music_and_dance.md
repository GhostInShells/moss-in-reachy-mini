# Reachy Mini 音乐与舞蹈系统

Reachy Mini 不只是一个能听懂你说话的桌面机器人——它能听歌、跳舞、当 DJ，还会在歌曲之间像朋友一样聊两句。

## 核心能力一览

### 1. 语音点歌

用自然语言和 Reachy Mini 说话就能点歌，无需任何 App 操作。

- **精确点歌**："播放周杰伦的晴天"
- **模糊描述**："我今天心情不太好，放点治愈的歌" → AI 自主推荐具体歌曲（如朴树《平凡之路》），边说推荐理由边开始播放
- **搜索选歌**："搜一下五月天的歌" → 返回列表让你挑选

音乐来源为 Bilibili 音频，播放过的歌曲自动缓存，再次点播秒播无延迟。

### 2. 连续播放（AI DJ 模式）

对 Reachy Mini 说"播一组适合晴天的歌，每首30秒"，它就变身 AI DJ：

- **自主选歌**：AI 根据你描述的心情/场景/风格，从自己的音乐知识库中持续挑选合适的歌曲
- **自动切歌**：每首歌播放指定时长后自动切到下一首，无需干预
- **歌间评论**：每首歌播完，用一两句话自然点评（"《晴天》的前奏一响就像回到了夏天的教室走廊~"），然后无缝衔接下一首
- **后台预下载**：当前歌播放的同时后台准备下一首，切歌无等待

### 3. AI 自主编舞

每首歌开始播放时，系统自动将歌曲信息（BPM、时长、节拍点）发送给 AI，AI 实时编排覆盖整首歌的舞蹈动作序列。

编舞特点：

- **风格匹配**：快歌用 headbanger_combo、chicken_peck 等激烈动作；慢歌用 pendulum_swing、side_to_side_sway 等柔和动作
- **多样不重复**：每首歌的编排都不同，同一首歌也不会简单地循环几个动作
- **高潮/安静感知**：高潮段动作密集，安静段用停顿留白
- **多通道表现力**：不只舞蹈，还穿插头部律动（左右转、上下点）、天线摆动、表情动作

### 4. 节拍对齐（Beat Sync）

Reachy Mini 的动作不是随意播放的——每个动作都踩在音乐节拍上。

技术实现：
- 用 **librosa** 分析音频，提取精确的节拍时间点数组
- 每个动作（舞蹈、表情、头部、天线）执行前自动等待到**最近的节拍边界**
- AI 在动作之间插入节拍整数倍的停顿（`sleep`），进一步强化节奏感

### 5. 随时打断，干净切换

播歌跳舞过程中，你随时可以按下说话键打断。Reachy Mini 会：

- 立刻停止当前音乐和舞蹈动作
- 清空所有旧状态（歌单、下载任务、定时器）
- 全神贯注地听你说话并回应
- 你可以给出新指令（"换首嗨的"、"停止音乐"、"播一组爵士乐"），它从零开始执行

### 6. 播放控制

| 指令 | 说明 |
|------|------|
| "播放XX" | 搜索并播放指定歌曲 |
| "暂停音乐" | 暂停当前播放 |
| "继续播放" | 恢复播放，并重新编排剩余时长的舞蹈 |
| "停止音乐" | 完全停止，清空所有状态 |

---

## 舞蹈动作库

### 预设舞蹈（20 种）

每个舞蹈动作的时长固定（基于内部 114 BPM），不随歌曲 BPM 变化。

| 动作名 | 时长 | 风格 | 描述 |
|--------|------|------|------|
| `simple_nod` | ~2.6s | 基础 | 连续上下点头 |
| `side_to_side_sway` | ~2.6s | 基础 | 左右平滑摇摆 |
| `pendulum_swing` | ~2.6s | 基础 | 钟摆式摆动 |
| `groovy_sway_and_roll` | ~2.6s | 基础 | 摇摆+翻滚的律动感 |
| `head_tilt_roll` | ~2.6s | 基础 | 头部侧向翻滚 |
| `headbanger_combo` | ~2.6s | 激烈 | 用力甩头+垂直弹跳 |
| `chicken_peck` | ~2.6s | 激烈 | 鸡啄式前冲 |
| `neck_recoil` | ~2.6s | 激烈 | 脖子快速后仰弹回 |
| `chin_lead` | ~2.6s | 激烈 | 下巴前探带动 |
| `interwoven_spirals` | ~4.7s | 花式 | 三轴不同频率的螺旋运动 |
| `polyrhythm_combo` | ~3.7s | 花式 | 3 拍摇摆 + 2 拍点头的复合节奏 |
| `jackson_square` | ~5.8s | 花式 | 沿矩形轨迹移动，到达每个角时抖动 |
| `side_peekaboo` | ~5.8s | 花式 | 多阶段左右躲猫猫表演 |
| `dizzy_spin` | ~2.6s | 趣味 | 眩晕式圆圈运动 |
| `stumble_and_recover` | ~2.6s | 趣味 | 踉跄后恢复平衡 |
| `yeah_nod` | ~2.6s | 趣味 | 强调式两段"耶"点头 |
| `uh_huh_tilt` | ~2.6s | 趣味 | 同意式歪头+点头 |
| `grid_snap` | ~2.6s | 机械 | 机器人式网格卡点 |
| `sharp_side_tilt` | ~3.7s | 机械 | 三角波快速锐利侧倾 |
| `side_glance_flick` | ~2.6s | 机械 | 快速侧瞥后归位 |

### 表情动作（80+ 种）

通过 emoji 触发全身表情动作。部分示例：

| 类别 | emoji | 名称 |
|------|-------|------|
| 开心 | 😊 cheerful | 😄 laughing | 🤩 enthusiastic | 🥳 success |
| 伤感 | 😢 sad | 😭 crying | 😔 downcast | 🥺 lonely |
| 害怕 | 😱 scared | 😨 fear | 💀 dying |
| 思考 | 🤔 thoughtful | 😕 confused | 🤷 incomprehensible |
| 舞蹈 | 💃 dance1 | 🕺 dance2 | 💃🏻 dance3 |
| 社交 | 🙏 grateful | 🤗 come | 👋 go away | 🎉 welcoming |
| 骄傲 | 🏆 proud | 😎 proud2 | 🏅 success |
| 生气 | 😡 irritated | 😤 displeased | 🤬 furious | 💢 rage |
| 放松 | 😌 relief | 🧘 calming | 🧘‍♀️ serenity |

### 辅助动作

| 通道 | 参数 | 说明 |
|------|------|------|
| **head_move** | yaw, pitch, duration | 头部转向（左右 yaw、上下 pitch），可设持续时间 |
| **antennas_move** | left, right, duration | 两只天线（耳朵）独立控制角度 |
| **head_reset** | - | 头部回正 |
| **antennas_reset** | - | 天线回到默认位置 |
| **sleep** | duration | 节拍停顿，用于踩准节奏 |

---

## 技术架构简述

```
用户语音 → ASR → LLM（大语言模型）→ CTML 指令流
                                        ↓
                               ┌────────┴────────┐
                               ↓                  ↓
                          sound 通道          reachy_mini 通道
                          (TTS语音)          (动作执行，单轨阻塞)
                                                  ↓
                                        dance / emotion /
                                        head_move / antennas_move
                                        (全部经过 beat-sync 对齐)
```

**关键组件**：
- **音频分析**：librosa 提取 BPM + 节拍时间点数组
- **Beat Sync**：`bisect.bisect_right` 在节拍数组上二分查找最近的节拍边界
- **编舞引擎**：系统将歌曲信息（BPM、时长、动作时长表）发送给 LLM，LLM 实时生成 CTML 动作序列
- **状态管理**：`_stopping` flag 防止 worker 线程回调竞态；`_continuous` flag 区分单曲/连播模式
- **缓存系统**：本地磁盘缓存 + JSON 索引，避免重复下载