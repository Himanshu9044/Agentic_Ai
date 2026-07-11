"""
Multi-Agent Healthcare Assistant (LangGraph + Groq)
====================================================

Architecture
------------
                        ┌──────────────────┐
                        │ emergency_guard   │  <- deterministic keyword pre-check
                        └─────────┬─────────┘
                       not urgent │  urgent -> emergency (short-circuit)
                                  ▼
                        ┌────────────┐
                        │ Supervisor │  <- LLM classifies intent
                        └─────┬──────┘
        ┌───────────┬─────────┼─────────┬───────────┐
        ▼           ▼         ▼         ▼            ▼
   help_desk   symptom_checker  medication_info  report_explainer  emergency
        │           │         │         │            │
        └───────────┴─────────┴─────────┴────────────┘
                         ▼
                        END

What changed from v1
---------------------
1. **Conversation memory** — the graph is compiled with a `MemorySaver`
   checkpointer, keyed by `thread_id`. Callers (e.g. the PyQt5 GUI) no longer
   need to resend the full message history on every turn — just invoke with
   the new HumanMessage and the same thread_id, and LangGraph restores prior
   state automatically.
2. **Deterministic emergency fast-path** — red-flag detection no longer
   relies solely on the LLM classifier. A keyword/pattern pre-check
   (`emergency_guard_node`) runs first and short-circuits straight to the
   emergency agent on an obvious match. This is a defense-in-depth measure:
   the LLM classifier can still catch phrasing the keyword list misses, but
   a clear-cut case ("chest pain", "can't breathe", "suicidal") no longer
   depends on a single LLM call getting the classification right.
3. **`run_query()` helper** — a small convenience function
   (`run_query(user_text, thread_id)`) that the GUI (or any caller) can use
   without touching LangGraph internals directly.
4. **Safer routing fallback + basic logging** of routing decisions to make
   debugging misroutes easier.

Design notes / safety guardrails (unchanged from v1)
-----------------------------------------------------
- No agent diagnoses a condition or prescribes/doses medication.
- symptom_checker gives general info + urgency level, always recommends
  professional consultation for anything non-trivial.
- medication_info gives general education about drugs but explicitly
  refuses to give personalized dosing; always defers to a pharmacist/doctor.
- report_explainer explains what lab values *mean in general* without
  interpreting the user's personal results as a diagnosis.
- emergency short-circuits everything else and tells the user to seek
  immediate care (call emergency services / go to ER).
"""

import logging
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("healthcare_assistant")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str  # which specialist agent handled/will handle this turn


# ---------------------------------------------------------------------------
# Emergency keyword fast-path
# ---------------------------------------------------------------------------
# Deterministic, cheap, and doesn't depend on the LLM getting classification
# right on the very messages where getting it wrong matters most.

_EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bcan'?t breathe\b",
    r"\bdifficult(y)? breathing\b",
    r"\bshortness of breath\b",
    r"\bsevere bleeding\b",
    r"\bheavy bleeding\b",
    r"\bunconscious\b",
    r"\bnot breathing\b",
    r"\bstroke\b",
    r"\bface (is )?droop(ing)?\b",
    r"\bslurred speech\b",
    r"\bsuicid",
    r"\bkill myself\b",
    r"\bwant to die\b",
    r"\bself[- ]harm\b",
    r"\boverdose\b",
    r"\bpoisoned\b",
    r"\banaphylaxis\b",
    r"\bsevere allergic reaction\b",
    r"\bseizure\b",
    r"\bheart attack\b",
    r"\bcan'?t feel (my|one side)\b",
]
_EMERGENCY_REGEX = re.compile("|".join(_EMERGENCY_PATTERNS), re.IGNORECASE)


def looks_like_emergency(text: str) -> bool:
    return bool(_EMERGENCY_REGEX.search(text))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def book_appointment(department: str, preferred_date: str) -> str:
    """Book a hospital/clinic appointment.

    Args:
        department (str): e.g. "Cardiology", "General Physician", "Dermatology"
        preferred_date (str): e.g. "2026-07-15" or "next Monday"

    Returns:
        str: confirmation message with a mock appointment id
    """
    return (
        f"Appointment request received for {department} on {preferred_date}. "
        f"Confirmation ID: APT-{abs(hash(department + preferred_date)) % 10000}. "
        f"A staff member will confirm the exact time via SMS/email."
    )


@tool
def check_insurance_coverage(plan_name: str, service: str) -> str:
    """Check whether a given service is covered under an insurance plan (mock lookup).

    Args:
        plan_name (str): name of the insurance plan
        service (str): the medical service to check, e.g. "MRI scan"

    Returns:
        str: mock coverage info
    """
    return (
        f"[Mock data] Under '{plan_name}', '{service}' is typically covered at "
        f"80% after deductible. Please confirm exact terms with your insurer, "
        f"as this is illustrative data only."
    )


@tool
def lookup_medication_info(medication_name: str) -> str:
    """Look up general educational information about a medication.
    Does NOT provide personalized dosing.

    Args:
        medication_name (str): name of the medication, e.g. "Ibuprofen"

    Returns:
        str: general info (class, common uses, common side effects)
    """
    # In production, replace with a real drug-info API (e.g. OpenFDA, RxNorm).
    return (
        f"[Mock lookup] '{medication_name}' — general class/use/side-effect "
        f"information would be retrieved here from a verified drug database. "
        f"Always confirm dosing with a licensed pharmacist or doctor."
    )


@tool
def explain_lab_term(term: str) -> str:
    """Explain what a lab report term/metric generally means (not a diagnosis).

    Args:
        term (str): lab value name, e.g. "Hemoglobin", "LDL Cholesterol"

    Returns:
        str: plain-language explanation of what the metric measures
    """
    # In production, replace with a curated medical glossary lookup.
    return (
        f"[Mock glossary] '{term}' is a lab measurement used to assess a "
        f"specific aspect of health. A doctor should interpret your specific "
        f"value in the context of your full health picture."
    )


help_desk_tools = [book_appointment, check_insurance_coverage]
medication_tools = [lookup_medication_info]
report_tools = [explain_lab_term]

# ---------------------------------------------------------------------------
# System prompts (safety guardrails live here)
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are a routing supervisor for a healthcare assistant.
Classify the user's LATEST message into exactly one category:

- "emergency": mentions severe/life-threatening symptoms (e.g. chest pain,
  difficulty breathing, severe bleeding, stroke signs, suicidal ideation,
  loss of consciousness) or any situation needing IMMEDIATE care.
- "help_desk": appointments, hospital logistics, insurance, billing, general
  facility questions.
- "symptom_checker": general (non-emergency) symptoms, "what could this be",
  general health questions.
- "medication_info": questions about a medicine, drug interactions, side
  effects, what a medication is used for.
- "report_explainer": questions about lab results, test reports, medical
  report terminology.

Respond with ONLY one word: emergency, help_desk, symptom_checker,
medication_info, or report_explainer.
"""

HELP_DESK_PROMPT = """You are the Help Desk agent for a healthcare platform.
You assist with appointment booking, hospital logistics, and insurance
questions using the tools available to you. Be concise and practical.
You do not give medical advice — redirect medical questions to the
appropriate specialist by telling the user to ask about symptoms/medication
directly."""

SYMPTOM_CHECKER_PROMPT = """You are a Symptom Information agent.
You are NOT a doctor and must never provide a diagnosis. You may:
- Explain general possible causes for a symptom (as *possibilities*, not conclusions)
- Suggest an appropriate urgency level (self-care / see a doctor soon / seek urgent care)
- Always end with a recommendation to consult a licensed medical professional
  for an actual diagnosis, especially for anything persistent or severe.
Keep tone calm, clear, and non-alarming."""

MEDICATION_PROMPT = """You are a Medication Information agent.
You provide general educational information about medications using the
lookup_medication_info tool: what a drug class is typically used for and
common side effects. You NEVER give personalized dosing instructions,
never tell someone to start/stop/change a medication, and always instruct
the user to confirm dosing and interactions with a licensed pharmacist or
doctor."""

REPORT_PROMPT = """You are a Lab Report Explainer agent.
Using the explain_lab_term tool, you explain what lab report metrics
generally measure and what "normal ranges" typically represent in plain
language. You do NOT interpret the user's personal results or provide a
diagnosis — you always tell them their doctor is the right person to
interpret their specific results in context."""

EMERGENCY_PROMPT = """You are an Emergency Triage agent. The user's message
suggests a potentially serious/urgent medical situation. Respond calmly and
clearly:
1. Tell them to call their local emergency number or go to the nearest ER
   immediately if this is a real, present situation.
2. Do not attempt to diagnose or delay them with questions.
3. If relevant, mention they can also contact a crisis line for mental
   health emergencies.
Keep the response short, direct, and reassuring — not alarmist."""

VALID_ROUTES = {
    "emergency",
    "help_desk",
    "symptom_checker",
    "medication_info",
    "report_explainer",
}

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def emergency_guard_node(state: State) -> dict:
    """Deterministic pre-check. Only sets a route when it detects an obvious
    red flag; otherwise leaves routing decision to the supervisor node."""
    last_user_msg = state["messages"][-1]
    text = getattr(last_user_msg, "content", "") or ""
    if looks_like_emergency(text):
        logger.info("emergency_guard: keyword match -> routing to emergency")
        return {"route": "emergency"}
    return {"route": ""}  # unset; supervisor will decide


def guard_decision(state: State) -> str:
    return "emergency" if state.get("route") == "emergency" else "supervisor"


def supervisor_node(state: State) -> dict:
    last_user_msg = state["messages"][-1]
    result = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT), last_user_msg])
    route = result.content.strip().lower()
    if route not in VALID_ROUTES:
        logger.warning("supervisor: unrecognized route '%s', defaulting to help_desk", route)
        route = "help_desk"  # safe default
    logger.info("supervisor: routed to '%s'", route)
    return {"route": route}


def route_decision(state: State) -> str:
    return state["route"]


def help_desk_node(state: State) -> dict:
    llm_with_tools = llm.bind_tools(help_desk_tools)
    messages = [SystemMessage(content=HELP_DESK_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


def symptom_checker_node(state: State) -> dict:
    messages = [SystemMessage(content=SYMPTOM_CHECKER_PROMPT)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


def medication_info_node(state: State) -> dict:
    llm_with_tools = llm.bind_tools(medication_tools)
    messages = [SystemMessage(content=MEDICATION_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


def report_explainer_node(state: State) -> dict:
    llm_with_tools = llm.bind_tools(report_tools)
    messages = [SystemMessage(content=REPORT_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


def emergency_node(state: State) -> dict:
    messages = [SystemMessage(content=EMERGENCY_PROMPT)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(State)

builder.add_node("emergency_guard", emergency_guard_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("help_desk", help_desk_node)
builder.add_node("symptom_checker", symptom_checker_node)
builder.add_node("medication_info", medication_info_node)
builder.add_node("report_explainer", report_explainer_node)
builder.add_node("emergency", emergency_node)

# tool nodes for the agents that use tools
builder.add_node("help_desk_tools", ToolNode(help_desk_tools))
builder.add_node("medication_tools", ToolNode(medication_tools))
builder.add_node("report_tools", ToolNode(report_tools))

builder.add_edge(START, "emergency_guard")

builder.add_conditional_edges(
    "emergency_guard",
    guard_decision,
    {"emergency": "emergency", "supervisor": "supervisor"},
)

builder.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "emergency": "emergency",
        "help_desk": "help_desk",
        "symptom_checker": "symptom_checker",
        "medication_info": "medication_info",
        "report_explainer": "report_explainer",
    },
)

# help_desk may call tools, then must come back to help_desk to compose the
# final answer, then end
builder.add_conditional_edges(
    "help_desk", tools_condition, {"tools": "help_desk_tools", END: END}
)
builder.add_edge("help_desk_tools", "help_desk")

builder.add_conditional_edges(
    "medication_info", tools_condition, {"tools": "medication_tools", END: END}
)
builder.add_edge("medication_tools", "medication_info")

builder.add_conditional_edges(
    "report_explainer", tools_condition, {"tools": "report_tools", END: END}
)
builder.add_edge("report_tools", "report_explainer")

builder.add_edge("symptom_checker", END)
builder.add_edge("emergency", END)

# ---------------------------------------------------------------------------
# Compile with memory (per-thread conversation persistence)
# ---------------------------------------------------------------------------
# NOTE: MemorySaver is in-process/in-memory only — conversation history is
# lost when the process exits. For persistence across app restarts, swap in
# langgraph.checkpoint.sqlite.SqliteSaver or a Postgres-backed checkpointer.

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Convenience API for callers (e.g. the PyQt5 GUI)
# ---------------------------------------------------------------------------


def run_query(user_text: str, thread_id: str = "default") -> tuple[str, str]:
    """Send a single user message into the graph and get back (route, reply).

    Conversation history for a given thread_id is automatically maintained
    by the checkpointer — callers do not need to resend prior messages.

    Args:
        user_text: the new user message
        thread_id: conversation/session identifier (e.g. one per GUI window,
            or one per logged-in user session)

    Returns:
        (route, reply_text)
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
    route = result.get("route", "help_desk")
    reply = result["messages"][-1].content
    return route, reply


def reset_conversation(thread_id: str = "default") -> None:
    """Clear stored history for a thread by simply starting a new thread_id
    convention on the caller side. MemorySaver has no explicit per-thread
    delete API, so the practical pattern is: caller generates a fresh
    thread_id (e.g. uuid4()) to start a new conversation."""
    logger.info("reset_conversation called for thread_id='%s' (start a new thread_id to reset)", thread_id)


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("healthcare_graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved as healthcare_graph.png")
    except Exception as e:
        print(f"Could not render graph png: {e}")

    # -----------------------------------------------------------------------
    # Demo runs — includes a multi-turn example to show memory in action
    # -----------------------------------------------------------------------

    print("\n" + "#" * 70)
    print("# Single-turn tests")
    print("#" * 70)

    test_queries = [
        "I want to book a cardiology appointment for next Monday.",
        "I've had a mild headache for two days, what could it be?",
        "What is Ibuprofen used for and what are the side effects?",
        "My report shows Hemoglobin 10.5, what does that measure?",
        "I have severe chest pain and can't breathe properly.",
    ]

    for i, q in enumerate(test_queries):
        print("\n" + "=" * 70)
        print("USER:", q)
        route, reply = run_query(q, thread_id=f"demo-single-{i}")
        print("ROUTE:", route)
        print("ASSISTANT:", reply)

    print("\n" + "#" * 70)
    print("# Multi-turn memory test (same thread_id)")
    print("#" * 70)

    memory_thread = "demo-memory-1"
    turn1 = "What is Paracetamol generally used for?"
    turn2 = "And what side effects should I watch out for with it?"

    print("\nUSER:", turn1)
    route, reply = run_query(turn1, thread_id=memory_thread)
    print("ROUTE:", route)
    print("ASSISTANT:", reply)

    print("\nUSER:", turn2)
    route, reply = run_query(turn2, thread_id=memory_thread)
    print("ROUTE:", route)
    print("ASSISTANT:", reply)