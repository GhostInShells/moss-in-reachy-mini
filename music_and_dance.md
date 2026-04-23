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

### 3. AI 自主编舞（支持三种表演模式）

每首歌开始播放时，系统根据 `REACHY_MINI_PERFORMANCE_MODE` 环境变量选择编舞策略：

| 模式 | 说明 | Token 消耗 | 效果 |
|------|------|-----------|------|
| `rich`（默认）| LLM 一次性编排整首歌的完整动作序列 | 多 | 最佳，完全自由创作 |
| `loop` | LLM 生成约 20 秒的循环单元，系统自动重复直到歌曲结束 | 约为 rich 的 1/10 | 有节拍感，每段有变化 |
| `auto` | 程序化循环，不调用 LLM | 零 | 固定模式，完全节省 token |

#### rich 模式

AI 根据歌曲 BPM 和时长一次性生成覆盖整首歌的 CTML 动作序列，包含舞蹈、头部律动、天线摆动、RGB 灯光等全部通道，效果最丰富。

#### loop 模式（Token 优化重构）

循环单元时长根据 BPM 自适应计算，对齐到 4 拍小节边界：

```
目标约 20 秒 → loop_beats = round(20 / beat_dur / 4) * 4（最少 8 拍）
BPM=60  → 20 拍 × 1.0s = 20.0s
BPM=120 → 40 拍 × 0.5s = 20.0s
BPM=180 → 48 拍 × 0.33s ≈ 16.0s
```

LLM 仅在每首歌开始时调用**一次**，将循环单元保存为 `music_loop.ctml`。系统轮询等待保存完成后，用 `<loop times="N">` 原语将其重复覆盖整首歌剩余时长，不再反复调用 LLM。换歌时自动删除旧的 `music_loop.ctml`，重新生成。

RGB 灯光在单元内依次切换 all / alter / running / gradient 四种 mode。

#### auto 模式

完全不调用 LLM。按 BPM 高低选择动作风格：
- **快歌（BPM ≥ 100）**：headbanger_combo、chicken_peck、yeah_nod、neck_recoil、grid_snap 等
- **慢歌（BPM < 100）**：pendulum_swing、side_to_side_sway、simple_nod、groovy_sway_and_roll 等

每 8 拍为一个自动循环段，RGB 灯光模式逐段轮换，天线随机变换角度。

#### 配置方式

在 `.env` 中设置：
```
REACHY_MINI_PERFORMANCE_MODE=rich   # 或 loop / auto
```

编舞特点（rich / loop 模式均适用）：

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
| "保存这段舞蹈" | 保存当前（或最近）歌曲的编排供下次复用（仅 rich 模式）|
| "重新编舞播放XX" / "换个舞蹈动作" | 忽略已保存的编排，强制 LLM 重新生成（`refresh=True`）|

### 7. 编舞记忆（Choreography Cache）

播放过的歌曲编舞可以保存下来，下次播放同一首歌时直接复用，跳过 LLM 调用。

**自动保存：**`extract_choreography.py` 脚本从 `terminal.log` 中提取 rich 模式的编舞 CTML，自动识别并跳过 loop 模式输出（`save_ctml name="music_loop"`），选取同一首歌多次出现中标签数和类型数最多的版本覆盖保存。

```bash
python src/moss_in_reachy_mini/scripts/extract_choreography.py [--dry-run]
```

**手动保存：**对 Reachy Mini 说"保存这段舞蹈"，它会重新生成当前（或最近播放）歌曲的编排并存档。仅在 rich 模式有效。

**强制重新生成：**对 Reachy Mini 说"重新编舞"或"换个舞蹈动作"，LLM 会以 `refresh=True` 调用 `play_music`，忽略已保存的编排重新创作。新编排不会自动覆盖旧文件，需要再说"保存这段舞蹈"才会更新。

保存路径：`.workspace/runtime/ctml_repo/song_{bvid}.ctml`

**terminal.log：** `main.py` 启动时自动将终端输出追加写入 `.workspace/runtime/logs/terminal.log`，供 `extract_choreography.py` 使用。可通过 `.env` 中 `TERMINAL_LOG=no` 关闭。

---

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
- **编舞记忆**：`ctml_repo` 存储已保存的编舞，按 `song_{bvid}.ctml` 命名；`loop` 模式的循环单元存为 `music_loop.ctml`，换歌时自动清除
- **TTS 长连接**：`TTS_DISCONNECT_ON_IDLE`（默认 3600s）防止播放长歌曲时 WebSocket 超时导致 TTS 静默