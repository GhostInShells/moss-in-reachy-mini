import reflex as rx


"""
验证了流式生成组件是完全可行的，只需要定义好所有的组件类型，约定好数据格式，就可以实现动态添加组件的功能。

待验证：样式
待验证：布局之间的关系
"""


class State(rx.State):
    components: list[dict] = []
    card_text: str = ""
    button_label: str = ""

    def add_card(self):
        if self.card_text:
            # 使用不可变更新确保触发重新渲染
            self.components = self.components + [{"type": "card", "text": self.card_text}]
            self.card_text = ""

    def add_button(self):
        if self.button_label:
            self.components = self.components + [{"type": "button", "label": self.button_label}]
            self.button_label = ""

    def button_handler(self):
        return rx.window_alert("动态按钮被点击")

def render_component(comp: dict):
    return rx.match(
        comp["type"],
        ("card", rx.card(rx.text(comp["text"]), header="卡片")),
        ("button", rx.button(comp["label"], on_click=State.button_handler)),
        rx.text("未知组件"),
    )

def index() -> rx.Component:
    return rx.vstack(
        rx.heading("动态组件演示"),
        rx.hstack(
            rx.input(placeholder="卡片内容", value=State.card_text, on_change=State.set_card_text),
            rx.button("添加卡片", on_click=State.add_card),
        ),
        rx.hstack(
            rx.input(placeholder="按钮文字", value=State.button_label, on_change=State.set_button_label),
            rx.button("添加按钮", on_click=State.add_button),
        ),
        rx.divider(),
        rx.foreach(State.components, render_component),
        padding="2em",
    )
