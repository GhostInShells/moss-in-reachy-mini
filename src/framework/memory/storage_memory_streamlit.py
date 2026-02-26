# 独立的Streamlit UI
from datetime import datetime
import re

import streamlit as st
import aiohttp
import asyncio

from ghoshell_moss import Message, Text

from framework.memory.storage_memory import TurnAddition

# API基础配置
API_BASE_URL = "http://127.0.0.1:8088/memory"


# ========== 异步API调用封装 ==========
async def async_api_call(method, url, **kwargs):
    """异步调用API"""
    async with aiohttp.ClientSession() as session:
        try:
            if method == "GET":
                async with session.get(url, **kwargs) as resp:
                    return await resp.json()
            elif method == "POST":
                async with session.post(url, **kwargs) as resp:
                    return await resp.json()
        except Exception as e:
            st.error(f"API调用失败：{str(e)}")
            return {"code": 1, "msg": str(e)}


def run_async(coro):
    """适配Streamlit的同步环境执行异步协程"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ========== Streamlit UI类 ==========
class StorageMemoryUI:
    def __init__(self):
        # 初始化页面配置
        st.set_page_config(
            page_title="Agent记忆编辑器",
            page_icon="🧠",
            layout="wide"
        )
        # 自动刷新配置
        st.sidebar.divider()
        if st.sidebar.button("🔄 手动刷新所有数据"):
            st.rerun()

    def render(self):
        st.title("🧠 Agent Memory")
        st.text("ps:本页面所有模块的数据都是Agent可见的数据")
        st.divider()

        # 主布局
        tab1, tab2 = st.tabs(["记忆配置（可编辑）", "会话历史（可读）"])

        with tab1:
            self._render_memory_editors()

        with tab2:
            col1, col2 = st.columns([1, 3])
            with col1:
                self._render_memory_limitation()
                self._render_session_management()
            with col2:
                self._render_session_history()

    def _render_session_management(self):
        """会话管理"""
        st.subheader("会话管理")
        with st.container(border=True):
            # 获取当前会话信息
            session_info = run_async(async_api_call("GET", f"{API_BASE_URL}/session_info"))

            if session_info["code"] == 0:
                st.text(f"当前会话ID：\n{session_info['session_id']}")

            # 新建会话按钮
            if st.button("📝 新建会话", type="primary"):
                with st.spinner("正在创建新会话..."):
                    res = run_async(async_api_call("POST", f"{API_BASE_URL}/new_session"))
                    if res["code"] == 0:
                        st.success(f"新会话创建成功！ID：{res['session_id']}")
                        st.rerun()

    def _render_memory_limitation(self):
        """记忆限制配置"""
        st.subheader("记忆限制配置")
        with st.container(border=True):
            # 获取当前配置
            session_info = run_async(async_api_call("GET", f"{API_BASE_URL}/session_info"))
            current_turns = session_info.get("turn_rounds", 10) if session_info["code"] == 0 else 10
            current_tokens = session_info.get("max_tokens", -1) if session_info["code"] == 0 else -1

            # 输入新配置
            new_turns = st.number_input(
                "对话轮次限制",
                min_value=0,
                value=current_turns,
                help="设置Agent能访问的最大对话轮次（0=无历史）"
            )
            new_tokens = st.number_input(
                "Token上限",
                min_value=-1,
                value=current_tokens,
                help="-1=无限制，0=不允许访问历史内容"
            )

            # 保存配置
            if st.button("💾 保存配置"):
                with st.spinner("正在保存配置..."):
                    res = run_async(async_api_call(
                        "POST",
                        f"{API_BASE_URL}/set_limitation",
                        json={"turn_rounds": new_turns, "max_tokens": new_tokens}
                    ))
                    if res["code"] == 0:
                        st.success(res["msg"])

    def _render_memory_editors(self):
        """记忆模块编辑"""
        st.subheader("记忆内容编辑")
        # 记忆模块映射
        memory_modules = {
            "人格设定": "personality",
            "行为偏好": "behavior_preference",
            "情绪基底": "mood_base",
            "自传记忆": "autobiographical",
            "摘要记忆": "summary"
        }

        # Tabs布局
        tabs = st.tabs(list(memory_modules.keys()))
        for idx, (module_name, module_key) in enumerate(memory_modules.items()):
            with tabs[idx]:
                self._render_single_editor(module_name, module_key)

    def _render_single_editor(self, module_name: str, module_key: str):
        """单个记忆模块编辑"""
        edit_col, preview_col = st.columns(2)

        with edit_col:
            st.subheader(f"{module_name} - 编辑")
            # 读取当前内容
            res = run_async(async_api_call("GET", f"{API_BASE_URL}/read/{module_key}"))
            current_content = res.get("content", "") if res["code"] == 0 else ""

            # 编辑框
            new_content = st.text_area(
                f"{module_name}内容",
                value=current_content,
                height=400,
                placeholder=f"请输入{module_name}内容（Markdown格式）...",
                key=f"editor_{module_key}"
            )

            # 保存按钮
            if st.button(f"💾 保存{module_name}", key=f"save_{module_key}"):
                with st.spinner(f"正在保存{module_name}..."):
                    res = run_async(async_api_call(
                        "POST",
                        f"{API_BASE_URL}/refresh/{module_key}",
                        json={"content": new_content}
                    ))
                    if res["code"] == 0:
                        st.success(f"{module_name}保存成功！")

        with preview_col:
            st.subheader(f"{module_name} - 预览")
            st.markdown(new_content if new_content else f"> 暂无{module_name}内容")

    # 在MemoryEditorUI类中替换_render_session_history方法
    def _render_session_history(self):
        """会话历史展示（类聊天软件界面，严格区分用户/AI消息）"""
        st.subheader("📜 对话历史（仅展示AI可见的轮次数据）")
        with st.container(border=True):
            # 获取会话历史
            res = run_async(async_api_call("GET", f"{API_BASE_URL}/session_history"))
            if res["code"] != 0:
                st.error(f"获取会话历史失败：{res['msg']}")
                return

            history = res["history"]
            if not history:
                st.info("当前会话暂无对话记录")
                return

            # ========== 1. 按对话轮次分组（基于session_turn的turn_id） ==========
            turn_groups = {}
            for msg_dict in history:
                try:
                    msg = Message(**msg_dict)
                    # 提取对话轮次ID（从additional中获取session_turn）

                    turn = TurnAddition.read(msg)
                    turn_id = turn.turn_id if turn else None

                    # 按轮次分组消息
                    if turn_id not in turn_groups:
                        turn_groups[turn_id] = []
                    turn_groups[turn_id].append(msg)
                except Exception as e:
                    st.warning(f"解析消息失败：{str(e)}")
                    continue

            # ========== 2. 按对话时间排序轮次 ==========
            # 每个轮次取最早的消息时间作为排序依据
            sorted_turns = sorted(
                turn_groups.values(),
                key=lambda msgs: min(
                    msg.meta.created_at for msg in msgs if hasattr(msg.meta, "created_at")
                ) if msgs else 0
            )

            # ========== 3. 逐轮次展示对话 ==========
            for turn_idx, turn_msgs in enumerate(sorted_turns, 1):
                # 提取当前轮次的时间
                turn_time = ""
                if turn_msgs:
                    first_msg = turn_msgs[0]
                    if hasattr(first_msg.meta, "created_at") and first_msg.meta.created_at:
                        dt = datetime.fromtimestamp(first_msg.meta.created_at)
                        turn_time = dt.strftime("%Y-%m-%d %H:%M:%S")

                # 轮次标题
                with st.expander(f"🔄 第 {turn_idx} 轮对话 · {turn_time}", expanded=True):
                    # 遍历当前轮次的所有消息
                    for msg in turn_msgs:
                        # 处理消息文本（转换AI动作标签为友好格式）
                        text_contents = []
                        if msg.contents:
                            for content in msg.contents:
                                text_content = Text.from_content(content)
                                if text_content:
                                    # 替换AI动作标签（如<reachy_mini.asleep:wake_up/>）
                                    formatted_text = re.sub(
                                        r'<.*?\.(.*?):(.*?)\/>',
                                        r'<span style="background-color:#ffc107; padding:2px 6px; border-radius:4px; font-size:0.8em;">[动作：\2]</span>',
                                        text_content.text
                                    )
                                    text_contents.append(formatted_text)
                        full_text = "".join(text_contents)

                        # ========== 区分用户/AI消息，用聊天气泡展示 ==========
                        if msg.role == "user":
                            # 用户消息：蓝色气泡右对齐
                            st.markdown(f"""
                            <div style="display: flex; justify-content: flex-end; margin: 8px 0;">
                                <div style="background-color: #007bff; color: white; padding: 10px 14px; border-radius: 18px 18px 4px 18px; max-width: 75%;">
                                    <div style="font-size: 0.9em; white-space: pre-wrap;">{full_text}</div>
                                    <div style="font-size: 0.7em; text-align: right; opacity: 0.8; margin-top: 4px;">
                                        {datetime.fromtimestamp(msg.meta.created_at).strftime("%H:%M") if hasattr(msg.meta, "created_at") else ""}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif msg.role == "assistant":
                            # AI回复：灰色气泡左对齐，带AI标识
                            st.markdown(f"""
                            <div style="display: flex; justify-content: flex-start; margin: 8px 0;">
                                <div style="background-color: #f8f9fa; color: #333; padding: 10px 14px; border-radius: 18px 18px 18px 4px; max-width: 75%;">
                                    <div style="font-size: 0.9em; white-space: pre-wrap;">{full_text}</div>
                                    <div style="font-size: 0.7em; text-align: right; opacity: 0.6; margin-top: 4px;">
                                        {datetime.fromtimestamp(msg.meta.created_at).strftime("%H:%M") if hasattr(msg.meta, "created_at") else ""}
                                        <span style="margin-left: 8px; color: #007bff;">🤖</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # 其他角色（如system）：中性样式
                            st.markdown(f"""
                            <div style="display: flex; justify-content: center; margin: 8px 0;">
                                <div style="background-color: #e9ecef; color: #666; padding: 8px 12px; border-radius: 8px; max-width: 75%; font-size: 0.9em;">
                                    <strong>[{msg.role.upper()}]</strong> {full_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # 调试选项：查看当前轮次的原始数据
                    if st.checkbox(f"查看第 {turn_idx} 轮原始数据", key=f"turn_debug_{turn_idx}"):
                        st.json([msg.dump() for msg in turn_msgs])

# ========== 启动UI ==========
if __name__ == "__main__":
    ui = StorageMemoryUI()
    ui.render()