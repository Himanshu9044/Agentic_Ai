

import sys
import traceback
import uuid

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
)

from healthcare_multi_agent import run_query  # convenience wrapper w/ memory

# ---------------------------------------------------------------------------
# Palette — light, clean, clinical
# ---------------------------------------------------------------------------

BG = "#f5f8fb"
SIDEBAR_BG = "#ffffff"
BORDER = "#e3e8ee"
PRIMARY = "#2f6fed"
PRIMARY_DARK = "#2456bf"
TEXT_DARK = "#1b2733"
TEXT_MUTED = "#67758a"
USER_BUBBLE_BG = "#2f6fed"
USER_BUBBLE_TEXT = "#ffffff"
AGENT_BUBBLE_BG = "#ffffff"

AGENTS_INFO = [
    ("help_desk", "🗂", "Help Desk", "#3a7bd5", "Appointments & insurance"),
    ("symptom_checker", "🩺", "Symptom Checker", "#2e9e6c", "General symptom info"),
    ("medication_info", "💊", "Medication Info", "#a06fd6", "Medication education"),
    ("report_explainer", "📋", "Report Explainer", "#c98a2c", "Lab report terms"),
    ("emergency", "🚨", "Emergency Triage", "#d64545", "Urgent situations"),
]
ROUTE_COLORS = {key: color for key, _, _, color, _ in AGENTS_INFO}
ROUTE_LABELS = {key: f"{icon} {name}" for key, icon, name, _, _ in AGENTS_INFO}

DISCLAIMER_TEXT = (
    "⚠️ General information only — not a substitute for professional medical "
    "advice. In an emergency, call your local emergency number immediately."
)

APP_STYLESHEET = f"""
QMainWindow {{
    background-color: {BG};
}}
QWidget#sidebar {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
}}
QLabel#brandTitle {{
    font-size: 17px;
    font-weight: 700;
    color: {TEXT_DARK};
}}
QLabel#brandSubtitle {{
    font-size: 11px;
    color: {TEXT_MUTED};
}}
QLabel#sectionLabel {{
    font-size: 11px;
    font-weight: 700;
    color: {TEXT_MUTED};
    letter-spacing: 1px;
    padding: 10px 2px 2px 2px;
}}
QPushButton#newConvButton {{
    padding: 9px 12px;
    border-radius: 8px;
    background-color: {PRIMARY};
    color: white;
    font-size: 13px;
    font-weight: 600;
    border: none;
    text-align: left;
}}
QPushButton#newConvButton:hover {{
    background-color: {PRIMARY_DARK};
}}
QListWidget {{
    border: none;
    background: transparent;
    font-size: 13px;
    color: {TEXT_DARK};
    outline: none;
}}
QListWidget::item {{
    padding: 8px 8px;
    border-radius: 6px;
    margin: 1px 0px;
}}
QListWidget::item:selected {{
    background-color: #e8f0fe;
    color: {PRIMARY_DARK};
}}
QListWidget::item:hover {{
    background-color: #f0f3f7;
}}
QLabel#chatHeaderTitle {{
    font-size: 16px;
    font-weight: 700;
    color: {TEXT_DARK};
}}
QLabel#chatHeaderSubtitle {{
    font-size: 11px;
    color: {TEXT_MUTED};
}}
QFrame#chatHeader {{
    background-color: {SIDEBAR_BG};
    border-bottom: 1px solid {BORDER};
}}
QTextEdit#chatView {{
    background-color: {BG};
    border: none;
    padding: 6px;
}}
QLineEdit#inputBox {{
    padding: 11px 14px;
    border: 1px solid {BORDER};
    border-radius: 20px;
    font-size: 13px;
    background: white;
    color: {TEXT_DARK};
}}
QLineEdit#inputBox:focus {{
    border: 1px solid {PRIMARY};
}}
QPushButton#sendButton {{
    padding: 10px 22px;
    border-radius: 20px;
    background-color: {PRIMARY};
    color: white;
    font-size: 13px;
    font-weight: 600;
    border: none;
}}
QPushButton#sendButton:hover {{
    background-color: {PRIMARY_DARK};
}}
QPushButton#sendButton:disabled {{
    background-color: #b7c6ee;
}}
QLabel#disclaimer {{
    font-size: 10px;
    color: #9aa4b2;
    padding: 4px 2px 0px 2px;
}}
QLabel#statusLabel {{
    font-size: 11px;
    color: {PRIMARY};
    font-weight: 600;
    padding: 2px;
}}
QFrame#agentCard {{
    border-radius: 8px;
    padding: 2px;
}}
"""


# ---------------------------------------------------------------------------
# Background worker so the GUI never freezes while waiting on the LLM
# ---------------------------------------------------------------------------


class AgentWorker(QThread):
    finished = pyqtSignal(str, str)  # (route, reply_text)
    error = pyqtSignal(str)

    def __init__(self, user_text: str, thread_id: str):
        super().__init__()
        self.user_text = user_text
        self.thread_id = thread_id

    def run(self):
        try:
            route, reply = run_query(self.user_text, thread_id=self.thread_id)
            self.finished.emit(route, reply)
        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Agent status card
# ---------------------------------------------------------------------------


class AgentCard(QFrame):
    def __init__(self, key: str, icon: str, name: str, color: str, desc: str):
        super().__init__()
        self.key = key
        self.color = color
        self.setObjectName("agentCard")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.dot = QLabel()
        self.dot.setFixedSize(9, 9)

        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        name_label = QLabel(f"{icon} {name}")
        name_label.setStyleSheet(f"font-size:12px; font-weight:600; color:{TEXT_DARK};")
        desc_label = QLabel(desc)
        desc_label.setStyleSheet(f"font-size:10px; color:{TEXT_MUTED};")
        text_col.addWidget(name_label)
        text_col.addWidget(desc_label)

        layout.addWidget(self.dot, alignment=Qt.AlignTop)
        layout.addLayout(text_col, stretch=1)

        self.set_idle()

    def set_idle(self):
        self.setStyleSheet("QFrame#agentCard { background-color: transparent; }")
        self.dot.setStyleSheet("background-color:#c8ccd2; border-radius:4px;")

    def set_active(self):
        self.setStyleSheet(
            f"QFrame#agentCard {{ background-color: {self.color}15; "
            f"border-left: 3px solid {self.color}; }}"
        )
        self.dot.setStyleSheet(f"background-color:{self.color}; border-radius:4px;")


class AgentStatusPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.cards = {}
        for key, icon, name, color, desc in AGENTS_INFO:
            card = AgentCard(key, icon, name, color, desc)
            self.cards[key] = card
            layout.addWidget(card)

    def highlight(self, route: str):
        for key, card in self.cards.items():
            card.set_active() if key == route else card.set_idle()

    def reset(self):
        for card in self.cards.values():
            card.set_idle()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


class Sidebar(QWidget):
    new_conversation_requested = pyqtSignal()
    conversation_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 14)
        layout.setSpacing(6)

        brand_title = QLabel("⚕ HealthAssist AI")
        brand_title.setObjectName("brandTitle")
        brand_subtitle = QLabel("Multi-Agent Care Assistant")
        brand_subtitle.setObjectName("brandSubtitle")
        layout.addWidget(brand_title)
        layout.addWidget(brand_subtitle)
        layout.addSpacing(14)

        self.new_conv_button = QPushButton("＋  New Conversation")
        self.new_conv_button.setObjectName("newConvButton")
        self.new_conv_button.clicked.connect(self.new_conversation_requested.emit)
        layout.addWidget(self.new_conv_button)

        conv_label = QLabel("CONVERSATIONS")
        conv_label.setObjectName("sectionLabel")
        layout.addWidget(conv_label)

        self.conv_list = QListWidget()
        self.conv_list.itemClicked.connect(
            lambda item: self.conversation_selected.emit(item.data(Qt.UserRole))
        )
        self.conv_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout.addWidget(self.conv_list, stretch=1)

        agents_label = QLabel("AGENTS")
        agents_label.setObjectName("sectionLabel")
        layout.addWidget(agents_label)

        self.agent_panel = AgentStatusPanel()
        layout.addWidget(self.agent_panel)

    def add_conversation(self, thread_id: str, title: str):
        item = QListWidgetItem(title)
        item.setData(Qt.UserRole, thread_id)
        self.conv_list.insertItem(0, item)
        self.conv_list.setCurrentItem(item)

    def update_conversation_title(self, thread_id: str, title: str):
        for i in range(self.conv_list.count()):
            item = self.conv_list.item(i)
            if item.data(Qt.UserRole) == thread_id:
                item.setText(title)
                return

    def select_conversation_silently(self, thread_id: str):
        for i in range(self.conv_list.count()):
            item = self.conv_list.item(i)
            if item.data(Qt.UserRole) == thread_id:
                self.conv_list.blockSignals(True)
                self.conv_list.setCurrentItem(item)
                self.conv_list.blockSignals(False)
                return

    def set_enabled(self, enabled: bool):
        self.new_conv_button.setEnabled(enabled)
        self.conv_list.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class HealthcareAssistantWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HealthAssist AI — Multi-Agent Healthcare Assistant")
        self.resize(1040, 680)

        self.worker = None
        self.conversations = {}  # thread_id -> {"title": str, "turns": [(kind, route, text)]}
        self.current_thread_id = None

        self.thinking_timer = QTimer(self)
        self.thinking_timer.timeout.connect(self._tick_thinking)
        self._thinking_dots = 0

        self._build_ui()
        self._create_new_conversation()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.new_conversation_requested.connect(self._create_new_conversation)
        self.sidebar.conversation_selected.connect(self._switch_conversation)
        root_layout.addWidget(self.sidebar)

        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)
        root_layout.addWidget(chat_panel, stretch=1)

        # header
        header = QFrame()
        header.setObjectName("chatHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 14, 24, 12)
        header_layout.setSpacing(1)
        self.chat_header_title = QLabel("New conversation")
        self.chat_header_title.setObjectName("chatHeaderTitle")
        self.chat_header_subtitle = QLabel(
            "Ask about appointments, symptoms, medications, or lab reports"
        )
        self.chat_header_subtitle.setObjectName("chatHeaderSubtitle")
        header_layout.addWidget(self.chat_header_title)
        header_layout.addWidget(self.chat_header_subtitle)
        chat_layout.addWidget(header)

        # chat transcript
        self.chat_view = QTextEdit()
        self.chat_view.setObjectName("chatView")
        self.chat_view.setReadOnly(True)
        self.chat_view.setFont(QFont("Segoe UI", 10))
        chat_layout.addWidget(self.chat_view, stretch=1)

        # bottom area
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(24, 8, 24, 12)
        bottom_layout.setSpacing(4)

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        bottom_layout.addWidget(self.status_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        self.input_box = QLineEdit()
        self.input_box.setObjectName("inputBox")
        self.input_box.setPlaceholderText(
            "e.g. 'Book a cardiology appointment' or 'What is Ibuprofen for?'"
        )
        self.input_box.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("sendButton")
        self.send_button.clicked.connect(self.send_message)
        input_row.addWidget(self.input_box, stretch=1)
        input_row.addWidget(self.send_button)
        bottom_layout.addLayout(input_row)

        disclaimer = QLabel(DISCLAIMER_TEXT)
        disclaimer.setObjectName("disclaimer")
        disclaimer.setWordWrap(True)
        bottom_layout.addWidget(disclaimer)

        chat_layout.addWidget(bottom)

    # -- conversation management ---------------------------------------------

    def _create_new_conversation(self):
        thread_id = str(uuid.uuid4())
        greeting = (
            "Hi! I can help with appointments, general symptom info, "
            "medication questions, or explaining a lab report. How can I help today?"
        )
        self.conversations[thread_id] = {
            "title": "New conversation",
            "turns": [("system", None, greeting)],
        }
        self.sidebar.add_conversation(thread_id, "New conversation")
        self.current_thread_id = thread_id
        self.sidebar.agent_panel.reset()
        self._render_current_conversation()

    def _switch_conversation(self, thread_id: str):
        if thread_id == self.current_thread_id or thread_id not in self.conversations:
            return
        self.current_thread_id = thread_id
        self._render_current_conversation()
        last_route = None
        for kind, route, _ in reversed(self.conversations[thread_id]["turns"]):
            if kind == "agent":
                last_route = route
                break
        if last_route:
            self.sidebar.agent_panel.highlight(last_route)
        else:
            self.sidebar.agent_panel.reset()

    def _render_current_conversation(self):
        conv = self.conversations[self.current_thread_id]
        self.chat_view.clear()
        for kind, route, text in conv["turns"]:
            if kind == "user":
                self._append_user_message(text)
            elif kind == "agent":
                self._append_agent_message(route, text)
            else:
                self._append_system_message(text)
        self.chat_header_title.setText(conv["title"])
        self.sidebar.select_conversation_silently(self.current_thread_id)

    # -- rendering helpers -----------------------------------------------------

    def _append_html(self, html: str):
        self.chat_view.moveCursor(QTextCursor.End)
        self.chat_view.insertHtml(html)
        self.chat_view.moveCursor(QTextCursor.End)

    def _append_user_message(self, text: str):
        html = (
            f'<div align="right" style="margin:8px 0;">'
            f'<span style="background-color:{USER_BUBBLE_BG}; color:{USER_BUBBLE_TEXT}; '
            f'padding:8px 12px; border-radius:10px; display:inline-block; max-width:70%;">'
            f"{self._escape(text)}</span></div>"
        )
        self._append_html(html)

    def _append_agent_message(self, route: str, text: str):
        color = ROUTE_COLORS.get(route, PRIMARY)
        label = ROUTE_LABELS.get(route, "Assistant")
        html = (
            f'<div align="left" style="margin:8px 0;">'
            f'<div style="font-size:10px; font-weight:600; color:{color}; margin-bottom:2px;">{label}</div>'
            f'<span style="background-color:{AGENT_BUBBLE_BG}; color:{TEXT_DARK}; '
            f"border:1px solid {BORDER}; border-left:3px solid {color}; "
            f'padding:8px 12px; border-radius:10px; display:inline-block; max-width:70%;">'
            f"{self._escape(text)}</span></div>"
        )
        self._append_html(html)

    def _append_system_message(self, text: str):
        html = (
            f'<div align="center" style="margin:10px 0;">'
            f'<span style="color:{TEXT_MUTED}; font-style:italic; font-size:11px;">'
            f"{self._escape(text)}</span></div>"
        )
        self._append_html(html)

    def _append_error_message(self, text: str):
        html = (
            f'<div align="center" style="margin:10px 0; color:#d64545;">'
            f"<b>Error:</b> {self._escape(text)}</div>"
        )
        self._append_html(html)

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    # -- actions -----------------------------------------------------------

    def send_message(self):
        text = self.input_box.text().strip()
        if not text:
            return

        conv = self.conversations[self.current_thread_id]
        if conv["title"] == "New conversation":
            title = text[:40] + ("…" if len(text) > 40 else "")
            conv["title"] = title
            self.sidebar.update_conversation_title(self.current_thread_id, title)
            self.chat_header_title.setText(title)

        conv["turns"].append(("user", None, text))
        self._append_user_message(text)
        self.input_box.clear()
        self._set_busy(True)

        self.worker = AgentWorker(text, self.current_thread_id)
        self.worker.finished.connect(self._on_agent_reply)
        self.worker.error.connect(self._on_agent_error)
        self.worker.start()

    def _set_busy(self, busy: bool):
        self.send_button.setEnabled(not busy)
        self.input_box.setEnabled(not busy)
        self.sidebar.set_enabled(not busy)
        if busy:
            self._thinking_dots = 0
            self.thinking_timer.start(400)
        else:
            self.thinking_timer.stop()
            self.status_label.setText("")

    def _tick_thinking(self):
        self._thinking_dots = (self._thinking_dots + 1) % 4
        self.status_label.setText("Thinking" + "." * self._thinking_dots)

    def _on_agent_reply(self, route: str, reply: str):
        self._set_busy(False)
        self.conversations[self.current_thread_id]["turns"].append(("agent", route, reply))
        self._append_agent_message(route, reply)
        self.sidebar.agent_panel.highlight(route)

    def _on_agent_error(self, error_text: str):
        self._set_busy(False)
        self._append_error_message(
            "Something went wrong talking to the assistant. "
            "Check your GROQ_API_KEY and network connection."
        )
        print(error_text)  # full traceback to console for debugging


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = HealthcareAssistantWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()