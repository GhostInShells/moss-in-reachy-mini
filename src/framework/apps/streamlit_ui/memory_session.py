from datetime import datetime
import re
import html
import streamlit as st
import asyncio
from ghoshell_moss import Message, Text
from framework.agent.response import AgentEventAddition
from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.session.storage_session import StorageSession, TurnAddition


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
    def __init__(self, storage_memory: StorageMemory, storage_session: StorageSession):
        self.storage_memory = storage_memory
        self.storage_session = storage_session

        # 初始化页面配置
        st.set_page_config(
            page_title="Agent记忆编辑器",
            page_icon="🧠",
            layout="wide"
        )
        st.sidebar.divider()
        if st.sidebar.button("🔄 手动刷新所有数据"):
            st.rerun()


    def render(self):
        st.title("🧠 Agent Memory & Session")
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
        """会话管理：直接操作StorageMemory实例"""
        st.subheader("会话管理")
        with st.container(border=True):
            # 直接从StorageMemory获取会话ID
            st.text(f"当前会话ID：\n{self.storage_session.meta_config.current_session_id}")

            # 新建会话按钮：直接调用StorageMemory方法
            if st.button("📝 新建会话", type="primary"):
                with st.spinner("正在创建新会话..."):
                    run_async(self.storage_session.new_session())
                    st.success(f"新会话创建成功！ID：{self.storage_session.meta_config.current_session_id}")
                    st.rerun()

    def _render_memory_limitation(self):
        """记忆限制配置：直接读写StorageMemory的meta_config"""
        st.subheader("记忆限制配置")
        with st.container(border=True):
            # 直接从StorageMemory获取当前配置
            current_turns = self.storage_session.meta_config.turn_rounds
            current_tokens = self.storage_session.meta_config.max_tokens

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

            # 保存配置：直接调用StorageMemory方法
            if st.button("💾 保存配置"):
                with st.spinner("正在保存配置..."):
                    run_async(self.storage_session.set_limitation(new_turns, new_tokens))
                    st.success("记忆限制配置已保存")

    def _render_memory_editors(self):
        """记忆模块编辑：直接读写StorageMemory的MD文件"""
        st.subheader("记忆内容编辑")
        # 记忆模块映射（对应StorageMemory的方法和配置）
        memory_modules = {
            "人格设定": ("personality", self.storage_memory.refresh_personality,
                         self.storage_memory.meta_config.personality_md),
            "行为偏好": ("behavior_preference", self.storage_memory.refresh_behavior_preference,
                         self.storage_memory.meta_config.behavior_preference_md),
            "情绪基底": ("mood_base", self.storage_memory.refresh_mood_base,
                         self.storage_memory.meta_config.mood_base_md),
            "自传记忆": ("autobiographical", self.storage_memory.refresh_autobiographical_memory,
                         self.storage_memory.meta_config.autobiographical_memory_md),
            "摘要记忆": ("summary", self.storage_memory.refresh_summary_memory,
                         self.storage_memory.meta_config.summary_memory_md)
        }

        # Tabs布局
        tabs = st.tabs(list(memory_modules.keys()))
        for idx, (module_name, (_, refresh_method, md_path)) in enumerate(memory_modules.items()):
            with tabs[idx]:
                self._render_single_editor(module_name, refresh_method, md_path)

    def _render_single_editor(self, module_name: str, refresh_method, md_path: str):
        """单个记忆模块编辑：直接读写StorageMemory的MD文件"""
        edit_col, preview_col = st.columns(2)

        with edit_col:
            st.subheader(f"{module_name} - 编辑")
            # 直接读取MD文件内容
            current_content = run_async(self.storage_memory.read_md(md_path))

            # 编辑框
            new_content = st.text_area(
                f"{module_name}内容",
                value=current_content,
                height=400,
                placeholder=f"请输入{module_name}内容（Markdown格式）...",
                key=f"editor_{module_name}"
            )

            # 保存按钮：直接调用StorageMemory的refresh方法
            if st.button(f"💾 保存{module_name}", key=f"save_{module_name}"):
                with st.spinner(f"正在保存{module_name}..."):
                    run_async(refresh_method(new_content))
                    st.success(f"{module_name}已保存！")

        with preview_col:
            st.subheader(f"{module_name} - 预览")
            st.markdown(new_content if new_content else f"> 暂无{module_name}内容")

    def _render_session_history(self):
        """会话历史展示：直接从StorageMemory获取会话历史"""
        st.subheader("📜 对话历史（按轮次分组）")
        with st.container(border=True):
            # 直接从StorageMemory获取会话历史
            history_msgs = run_async(self.storage_session.get_session_history())
            if not history_msgs:
                st.info("当前会话暂无对话记录")
                return

            # ========== 1. 按对话轮次分组（基于TurnAddition的turn_id） ==========
            turn_groups = {}
            for msg in history_msgs:
                turn = TurnAddition.read(msg)
                turn_id = turn.turn_id if turn else "unknown_turn"
                if turn_id not in turn_groups:
                    turn_groups[turn_id] = []
                turn_groups[turn_id].append(msg)

            # ========== 2. 按对话时间排序轮次 ==========
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
                    for msg in turn_msgs:
                        # 1. 提取AgentEventAddition信息
                        event_info = AgentEventAddition.read(msg)
                        event_html = "<span></span>"
                        if event_info:
                            event_color_map = {
                                "user_trigger": "#d4edda",
                                "agent_action": "#fff3cd",
                                "system_event": "#cce5ff",
                                "memory_update": "#f8d7da"
                            }
                            bg_color = event_color_map.get(event_info.event_type, "#e9ecef")

                            # 单行HTML避免渲染异常
                            event_html = (
                                f'<div style="font-size:0.65em; text-align:right; opacity:0.8; margin-top:2px; white-space:nowrap;">'
                                f'<span style="background-color:{bg_color}; padding:1px 4px; border-radius:3px;">事件：{event_info.event_type}</span>'
                                f'<span style="color:#666; margin-left:4px;">ID:{event_info.event_id}</span>'
                                f'</div>'
                            )

                        # 2. 处理消息文本：高亮动作标签 + 转义HTML
                        text_contents = []
                        if msg.contents:
                            for content in msg.contents:
                                text_content = Text.from_content(content)
                                if text_content:
                                    # 先转义所有HTML，避免原始HTML被解析
                                    escaped_text = html.escape(text_content.text)
                                    # 高亮自闭合动作标签
                                    formatted_text = re.sub(
                                        r'&lt;(.*)/&gt;',
                                        r'<span style="background-color:#ffc107; padding:2px 6px; border-radius:4px; font-size:0.9em; color:#212529;">&lt;\1/&gt;</span>',
                                        escaped_text
                                    )
                                    # 高亮成对动作标签
                                    formatted_text = re.sub(
                                        r'&lt;(.*)&gt;(.*?)&lt;(.*)gt;',
                                        r'<span style="background-color:#ffc107; padding:2px 6px; border-radius:4px; font-size:0.9em; color:#212529; display:inline-block;">&lt;\1&gt;\2&lt;/\3&gt;</span>',
                                        formatted_text
                                    )
                                    text_contents.append(formatted_text)
                        full_text = "".join(text_contents)

                        # 3. 渲染消息气泡
                        self._render_message_bubble(msg, full_text, event_html)

    def _render_message_bubble(self, msg: Message, full_text: str, event_html: str):
        """统一渲染聊天气泡（用户/AI/系统消息）"""
        if msg.role == "user":
            time_text = datetime.fromtimestamp(msg.meta.created_at).strftime("%H:%M") if hasattr(msg.meta,
                                                                                                 "created_at") else ""
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 8px 0;">
                <div style="background-color: #007bff; color: white; padding: 10px 14px; border-radius: 18px 18px 4px 18px; max-width: 75%;">
                    <div style="font-size: 0.9em; white-space: pre-wrap;">{full_text}</div>
                    <div style="font-size: 0.7em; text-align: right; opacity: 0.8; margin-top: 4px;">{time_text}</div>
                    {event_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif msg.role == "assistant":
            time_text = datetime.fromtimestamp(msg.meta.created_at).strftime("%H:%M") if hasattr(msg.meta,
                                                                                                 "created_at") else ""
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin: 8px 0;">
                <div style="background-color: #f8f9fa; color: #333; padding: 10px 14px; border-radius: 18px 18px 18px 4px; max-width: 75%;">
                    <div style="font-size: 0.9em; white-space: pre-wrap;">{full_text}</div>
                    <div style="font-size: 0.7em; text-align: right; opacity: 0.6; margin-top: 4px;">
                        {time_text} <span style="margin-left: 8px; color: #007bff;">🤖</span>
                    </div>
                    {event_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 8px 0;">
                <div style="background-color: #e9ecef; color: #666; padding: 8px 12px; border-radius: 8px; max-width: 75%; font-size: 0.9em;">
                    <strong>[{msg.role.upper()}]</strong> {full_text}
                    {event_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
