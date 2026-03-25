"""直播用户评估脚本

每次直播结束后手动运行，对所有用户进行两层评估：
1. stats（本地计算，不耗 token）：到访次数/频率、聊天/送礼/点赞统计、钻石总量等
2. ai_profile（LLM 生成）：性格特点、关注话题、与主播关系、互动建议

- 自动去重（同一 event_id 只保留一条），去重后写回 JSON
- 所有用户都会生成 stats；只有聊天过的用户才调 LLM 生成 ai_profile
- assessment 字段从 string 改为 dict: {"stats": {...}, "ai_profile": "..."}

用法:
    # 分析最近一场直播
    python -m moss_in_reachy_mini.scripts.assess_live_users

    # 指定 live_id
    python -m moss_in_reachy_mini.scripts.assess_live_users --live-id live_id_238956696661

    # 强制重新评估所有用户（即使已有 ai_profile）
    python -m moss_in_reachy_mini.scripts.assess_live_users --force

    # dry run，只看统计，不调 AI
    python -m moss_in_reachy_mini.scripts.assess_live_users --dry-run
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import openai
from dotenv import load_dotenv

from framework.apps.live.douyin_live import DouyinLiveUserHistory, DouyinLiveEventType

# 加载 .env
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

# LLM 配置（复用主项目的环境变量）
_LLM_BASE_URL = os.getenv("MOSS_LLM_BASE_URL", "")
_LLM_MODEL = os.getenv("MOSS_LLM_MODEL", "")
_LLM_API_KEY = os.getenv("MOSS_LLM_API_KEY", "")

# workspace 路径
_WS_DIR = Path(__file__).resolve().parent.parent / ".workspace"
_LIVE_MEMORY_DIR = _WS_DIR / "runtime" / "live_memory"

# 钻石值正则：匹配 "小心心x47（钻石=47）" 中的钻石数
_DIAMOND_RE = re.compile(r"钻石=(\d+)")

_SYSTEM_PROMPT = """\
你是直播间运营的用户分析助手。根据用户的行为数据和聊天记录，为主播生成一份用户画像。

你会收到两部分数据：
1. **行为统计**（已由系统计算）：到访次数、聊天数、送礼钻石等数字指标
2. **聊天记录**（原始内容）：用户在直播间发的弹幕

你的任务是基于聊天内容做**定性分析**，数字统计已经有了，不需要你重复。请输出：

## 性格与沟通风格
从聊天内容推断的性格特点、说话方式、情绪倾向（1-2 句）

## 关注话题
用户在直播间主要聊什么、关心什么（1-2 句）

## 与主播的关系
路人/普通观众/活跃粉丝/核心粉丝/铁粉 —— 判断依据是什么（1 句）

## 互动建议
主播下次见到这个用户时，怎么互动最合适（1-2 句，具体可执行）

要求：
- 中文输出，语气像给主播写的内部备忘
- 简洁，每个小节 1-2 句话，总共不超过 150 字
- 只分析有数据支撑的维度，没有依据的不要瞎猜
- 如果聊天记录很少（<5 条），相应缩短篇幅
"""


# ── 数据清洗 ──────────────────────────────────────────────


def _dedup_history(history: list[dict]) -> list[dict]:
    """按 event_id 去重，保留第一条。"""
    seen = set()
    result = []
    for event in history:
        eid = event.get("event_id")
        if eid and eid in seen:
            continue
        if eid:
            seen.add(eid)
        result.append(event)
    return result


def _parse_diamonds(content: str) -> int:
    """从礼物内容中提取钻石数，如 '小心心x47（钻石=47）' → 47。"""
    m = _DIAMOND_RE.search(content or "")
    return int(m.group(1)) if m else 0


def _parse_like_count(content: str) -> int:
    """从点赞内容中提取次数，如 '6次' → 6。"""
    m = re.search(r"(\d+)次", content or "")
    return int(m.group(1)) if m else 1


# ── 本地统计（不耗 token）──────────────────────────────────


def _compute_stats(history: list[dict], data: dict) -> dict:
    """基于去重后的完整 history（含 enter）计算统计指标。"""
    enters = [e for e in history if e.get("event_type") == "enter"]
    chats = [e for e in history if e.get("event_type") == "chat"]
    gifts = [e for e in history if "gift" in (e.get("event_type") or "")]
    likes = [e for e in history if e.get("event_type") == "like"]
    socials = [e for e in history if e.get("event_type") == "social"]

    # 时间范围
    all_times = [e.get("create_at", 0) for e in history if e.get("create_at")]
    first_seen = min(all_times) if all_times else 0
    last_seen = max(all_times) if all_times else 0

    # enter 时间戳列表（用于计算频率）
    enter_times = sorted(e.get("create_at", 0) for e in enters if e.get("create_at"))

    # 活跃天数（按日期去重）
    active_dates = set()
    for t in all_times:
        if t:
            active_dates.add(time.strftime("%Y-%m-%d", time.localtime(t)))

    # 钻石总量
    total_diamonds = sum(_parse_diamonds(e.get("content", "")) for e in gifts)

    # 点赞总次数（累加内容中的数字）
    total_likes = sum(_parse_like_count(e.get("content", "")) for e in likes)

    # 每次到访的平均间隔（秒）
    avg_visit_interval = None
    if len(enter_times) >= 2:
        intervals = [enter_times[i + 1] - enter_times[i] for i in range(len(enter_times) - 1)]
        avg_visit_interval = round(sum(intervals) / len(intervals))

    stats = {
        "visit_count": len(enters),
        "first_seen": time.strftime("%Y-%m-%d %H:%M", time.localtime(first_seen)) if first_seen else None,
        "last_seen": time.strftime("%Y-%m-%d %H:%M", time.localtime(last_seen)) if last_seen else None,
        "active_days": len(active_dates),
        "avg_visit_interval_seconds": avg_visit_interval,
        "chat_count": len(chats),
        "gift_count": len(gifts),
        "gift_diamonds_total": total_diamonds,
        "gift_types": dict(Counter(e.get("content", "").split("（")[0] for e in gifts)) if gifts else {},
        "like_total": total_likes,
        "social_count": len(socials),
        "is_core_user": data.get("is_core_user", False),
    }
    return stats


# ── LLM 调用 ──────────────────────────────────────────────


def _build_llm_input(data: dict, history: list[dict], stats: dict) -> str:
    """构建发送给 LLM 的上下文。stats 作为背景，聊天记录作为分析素材。"""
    lines = [
        f"# 用户: {data.get('user_name', '未知')}",
        "",
        "## 行为统计（已由系统计算）",
        f"- 到访 {stats['visit_count']} 次，活跃 {stats['active_days']} 天",
        f"- 首次出现: {stats['first_seen'] or '未知'}，最近出现: {stats['last_seen'] or '未知'}",
        f"- 发言 {stats['chat_count']} 条",
    ]

    if stats["gift_count"]:
        lines.append(f"- 送礼 {stats['gift_count']} 次，共 {stats['gift_diamonds_total']} 钻石")
        if stats["gift_types"]:
            top_gifts = ", ".join(f"{k}x{v}" for k, v in sorted(stats["gift_types"].items(), key=lambda x: -x[1])[:5])
            lines.append(f"- 礼物类型: {top_gifts}")
    if stats["like_total"]:
        lines.append(f"- 点赞 {stats['like_total']} 次")
    if stats["social_count"]:
        lines.append(f"- 社交互动 {stats['social_count']} 次")

    lines.append(f"- 核心用户: {'是' if stats['is_core_user'] else '否'}")

    # 聊天记录（LLM 分析的核心素材）
    chats = [e for e in history if e.get("event_type") == "chat"]
    if chats:
        # 最多 80 条，优先最近的
        sample = chats[-80:]
        lines.append(f"\n## 聊天记录（共 {len(chats)} 条，展示最近 {len(sample)} 条）")
        for e in sample:
            ts = time.strftime("%m-%d %H:%M", time.localtime(e.get("create_at", 0)))
            lines.append(f"[{ts}] {e.get('content', '')}")

    return "\n".join(lines)


async def _call_llm(llm_input: str) -> str:
    """调用 LLM 生成 ai_profile。"""
    client = openai.AsyncClient(
        api_key=_LLM_API_KEY,
        base_url=_LLM_BASE_URL,
    )

    response = await client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": llm_input},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ── 主流程 ────────────────────────────────────────────────


async def process_live_dir(live_dir: Path, *, force: bool = False, dry_run: bool = False, user_filter: str = None):
    """处理一个 live_id 目录下的所有用户 JSON。"""
    json_files = sorted(live_dir.glob("live_user_history_*.json"))
    if user_filter:
        json_files = [f for f in json_files if user_filter in f.name]
    print(f"目录: {live_dir.name}")
    print(f"用户文件总数: {len(json_files)}")

    needs_ai = []  # 需要 LLM 的用户
    stats_only_count = 0  # 只需更新 stats 的用户
    skipped_ai_count = 0  # 已有 ai_profile 的用户
    dedup_removed_total = 0

    for fp in json_files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        history_model = DouyinLiveUserHistory.model_validate(data)

        # 跳过非核心用户
        if not history_model.is_core_user:
            continue
        # 之前版本的核心用户逻辑比较松，此处再double check一下
        if not history_model.check_core_user():
            continue

        raw_history = data.get("history", [])

        deduped = _dedup_history(raw_history)
        dedup_removed = len(raw_history) - len(deduped)
        dedup_removed_total += dedup_removed

        # 计算 stats（所有用户都算）
        stats = _compute_stats(deduped, data)

        # 去重后写回 + 更新 stats
        # history 保留完整（含 enter），stats 存到 assessment.stats
        existing_assessment = data.get("assessment", "")
        if isinstance(existing_assessment, str):
            # 旧格式（纯字符串）→ 迁移为新结构
            assessment = {"stats": stats, "ai_profile": existing_assessment}
        else:
            assessment = existing_assessment or {}
            assessment["stats"] = stats

        data["history"] = deduped
        data["assessment"] = assessment

        if not dry_run:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        # 判断是否需要 LLM
        chats = [e for e in deduped if e.get("event_type") == "chat"]
        if not chats:
            stats_only_count += 1
            continue

        existing_profile = assessment.get("ai_profile", "")
        if existing_profile and not force:
            # 有新的有意义事件才重新评估（enter 不算）
            last_assessed = assessment.get("last_assessed_at", 0)
            latest_meaningful_time = max(
                (e.get("create_at", 0) for e in deduped if e.get("event_type") != "enter"),
                default=0,
            )
            if latest_meaningful_time <= last_assessed:
                skipped_ai_count += 1
                continue
            # 有新数据 → 需要重新评估

        needs_ai.append((fp, data, deduped, stats))

    print(f"去重移除事件数: {dedup_removed_total}")
    print(f"纯统计（无聊天）: {stats_only_count}")
    print(f"跳过（已有 AI 画像）: {skipped_ai_count}")
    print(f"需要 AI 分析: {len(needs_ai)}")

    if dry_run:
        for fp, data, history, stats in needs_ai:
            chat_count = stats["chat_count"]
            diamonds = stats["gift_diamonds_total"]
            extra = f", {diamonds}💎" if diamonds else ""
            print(f"  [DRY] {data.get('user_name')} - {chat_count} 条聊天{extra}")
        return

    if not needs_ai:
        print("\n所有用户 stats 已更新。无需 AI 分析。")
        return

    print(f"\n开始 AI 画像（共 {len(needs_ai)} 个用户）...\n")

    for i, (fp, data, history, stats) in enumerate(needs_ai, 1):
        user_name = data.get("user_name", "未知")
        chat_count = stats["chat_count"]
        print(f"[{i}/{len(needs_ai)}] {user_name} ({chat_count} 条聊天)...", end=" ", flush=True)

        try:
            llm_input = _build_llm_input(data, history, stats)
            ai_profile = await _call_llm(llm_input)

            data["assessment"]["ai_profile"] = ai_profile
            data["assessment"]["last_assessed_at"] = int(time.time())
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            # 截断显示首行
            first_line = ai_profile.split("\n")[0].strip()
            if len(first_line) > 80:
                first_line = first_line[:80] + "..."
            print(f"OK -> {first_line}")

        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\n完成！stats 已全量更新，AI 画像生成 {len(needs_ai)} 个。")


def main():
    parser = argparse.ArgumentParser(description="直播用户评估工具（stats + AI 画像）")
    parser.add_argument(
        "--live-id",
        default=None,
        help="指定 live_id 目录名。不指定则处理最新的。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新生成所有 AI 画像（stats 始终更新）",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="只处理匹配该关键词的用户（模糊匹配文件名）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计和更新 stats，不调用 AI",
    )
    args = parser.parse_args()

    if not args.dry_run and (not _LLM_MODEL or not _LLM_API_KEY):
        print("错误: 请在 .env 中配置 MOSS_LLM_MODEL 和 MOSS_LLM_API_KEY")
        sys.exit(1)

    # 找到 live_id 目录
    if args.live_id:
        live_dir = _LIVE_MEMORY_DIR / args.live_id
    else:
        live_dirs = sorted(
            [d for d in _LIVE_MEMORY_DIR.iterdir() if d.is_dir() and d.name.startswith("live_id_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not live_dirs:
            print("错误: 没有找到 live_id_ 目录")
            sys.exit(1)
        live_dir = live_dirs[0]

    if not live_dir.exists():
        print(f"错误: 目录不存在 {live_dir}")
        sys.exit(1)

    asyncio.run(process_live_dir(live_dir, force=args.force, dry_run=args.dry_run, user_filter=args.user))


if __name__ == "__main__":
    main()
