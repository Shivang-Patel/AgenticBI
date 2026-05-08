"""
AgenticBI Vizro frontend.
Layout: chart dashboard (left ~62%) | chat (right ~38%), Cursor-style dark UI.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime
from typing import Literal

import plotly.graph_objects as go
import plotly.io as pio
from dash import callback, dcc, html, Input, Output, State, no_update, ALL, MATCH, callback_context, clientside_callback, Patch
import vizro.models as vm
from vizro import Vizro

# ---------------------------------------------------------
# sys.path — _AGENTS_DIR first so agents' bare "from config import" works
# ---------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC         = os.path.join(_PROJECT_ROOT, "src")
_AGENTS_DIR  = os.path.join(_SRC, "agents")
_ASSETS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

for _p in (_AGENTS_DIR, _SRC, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pio.templates["agenticbi_dark"] = go.layout.Template(
    layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#171a1f",
        "font": {"family": "Inter, Segoe UI, system-ui, sans-serif", "color": "#d6dde6", "size": 12},
        "colorway": ["#4f8cff", "#4fd1a5", "#f2b84b", "#ff7d73", "#9aa6ff", "#5fd3f3", "#c6d04f"],
        "margin": {"l": 56, "r": 22, "t": 48, "b": 54},
        "hoverlabel": {
            "bgcolor": "#232830",
            "bordercolor": "#3e4652",
            "font": {"color": "#f5f7fa"},
        },
        "xaxis": {
            "gridcolor": "rgba(148,163,184,0.11)",
            "linecolor": "rgba(148,163,184,0.18)",
            "zerolinecolor": "rgba(148,163,184,0.18)",
            "tickfont": {"color": "#9aa7b5"},
            "title": {"font": {"color": "#b8c2ce"}},
        },
        "yaxis": {
            "gridcolor": "rgba(148,163,184,0.13)",
            "linecolor": "rgba(148,163,184,0.18)",
            "zerolinecolor": "rgba(148,163,184,0.18)",
            "tickfont": {"color": "#9aa7b5"},
            "title": {"font": {"color": "#b8c2ce"}},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#aab5c2"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    }
)
pio.templates.default = "agenticbi_dark"

# ---------------------------------------------------------
# Singletons
# ---------------------------------------------------------
_check_prompt_fn   = None
_orchestrator_inst = None


def _guardrails():
    global _check_prompt_fn
    if _check_prompt_fn is None:
        from guardrails_agent import check_prompt
        _check_prompt_fn = check_prompt
    return _check_prompt_fn


def _orchestrator():
    global _orchestrator_inst
    if _orchestrator_inst is None:
        from orchestrator_agent import OrchestratorAgent
        _orchestrator_inst = OrchestratorAgent()
    return _orchestrator_inst


# ---------------------------------------------------------
# Live progress: chips and Run-button narration update in real time during
# a pipeline run.
#
# The orchestrator pipeline (NL2SQL → SQL exec → viz → insights) takes
# 10–30 s. Rather than spawning a worker process (which crashes on macOS
# under `multiprocess` + HuggingFace fork-unsafety), we run the pipeline
# synchronously in the Flask thread that handles the Run callback, and
# update a thread-safe shared dict as each LangGraph node completes.
#
# A second callback driven by `dcc.Interval` polls that dict every ~700ms
# and pushes the latest chip state and button text to the browser. Flask
# `threaded=True` (the default in modern Flask/Werkzeug) lets the polling
# request run in a separate thread while the Run callback is still in
# progress, so the UI stays responsive.
# ---------------------------------------------------------
_NODE_TO_STAGE = {
    "generate_sql":      "sql",
    "execute_sql":       "query",
    "check_execution":   "query",
    "visualize":         "chart",
    "generate_insights": "memo",
}

_STAGE_BUTTON_TEXT = {
    None:    "Generating…",
    "sql":   "Running query…",
    "query": "Drawing chart…",
    "chart": "Writing memo…",
    "memo":  "Almost done…",
}

_PROGRESS_LOCK = threading.Lock()
_PROGRESS_STATE: dict = {
    # in_flight: True only while a run is being processed
    "in_flight": False,
    # tick: monotonically increasing — clients can detect "no change"
    "tick": 0,
    # the meta dict that _render_progress_chips consumes
    "meta": None,
    # current Run-button label
    "btn_text": "Run",
}


def _set_progress(meta: dict | None, btn_text: str, in_flight: bool):
    with _PROGRESS_LOCK:
        _PROGRESS_STATE["in_flight"] = in_flight
        _PROGRESS_STATE["meta"] = meta
        _PROGRESS_STATE["btn_text"] = btn_text
        _PROGRESS_STATE["tick"] += 1


def _get_progress_snapshot() -> dict:
    with _PROGRESS_LOCK:
        return dict(_PROGRESS_STATE)


def _streaming_meta(stages_done: set, running: str | None = None,
                    failed: bool = False, kind: str | None = None) -> dict:
    return {
        "sql":   "sql"   in stages_done,
        "query": "query" in stages_done,
        "chart": "chart" in stages_done,
        "memo":  "memo"  in stages_done,
        "running": running,
        "failed": failed,
        "kind":   kind,
    }


def _next_running(stages_done: set) -> str | None:
    for s in ("sql", "query", "chart", "memo"):
        if s not in stages_done:
            return s
    return None


# ---------------------------------------------------------
# Style tokens
# ---------------------------------------------------------
BG_MAIN   = "#0f1115"
BG_PANEL  = "#15181d"
BG_CARD   = "#1b2027"
BG_CARD_2 = "#141820"
BG_INPUT  = "#202631"
BG_MSG_US = "#1f3354"
BG_MSG_AI = "#1a1f26"
BORDER    = "1px solid #2a313a"
BORDER_SOFT = "1px solid #232a33"
ACCENT    = "#4f8cff"
ACCENT_SOFT = "#14233a"
TEXT      = "#e5e7eb"
TEXT_DIM  = "#8e99a8"
RADIUS    = "8px"
FONT      = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

# ---------------------------------------------------------
# Suggested queries for empty state
# ---------------------------------------------------------
SUGGESTED_QUERIES = [
    "Show total sales by region",
    "Top 10 products by revenue",
    "Sales trend over time",
    "Employee vacation hours",
    "Online vs in-store orders",
]


_WORKSPACE_INFO_CACHE: dict | None = None


def _workspace_info() -> dict:
    """Cheap one-shot probe of the connected DB so the empty state and
    workspace badge can show real signals instead of decorative copy.
    Cached for the lifetime of the process — schemas don't change mid-session."""
    global _WORKSPACE_INFO_CACHE
    if _WORKSPACE_INFO_CACHE is not None:
        return _WORKSPACE_INFO_CACHE

    info = {"connected": False, "db": None, "tables": None, "schemas": None, "error": None}
    try:
        import psycopg2
        from config import DB_CONFIG
        conn = psycopg2.connect(connect_timeout=2, **DB_CONFIG)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database()")
                info["db"] = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
                )
                info["tables"] = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(distinct table_schema) FROM information_schema.tables "
                    "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
                )
                info["schemas"] = cur.fetchone()[0]
                info["connected"] = True
        finally:
            conn.close()
    except Exception as exc:
        info["error"] = str(exc)[:120]

    _WORKSPACE_INFO_CACHE = info
    return info


_WORKSPACE_TREE_CACHE: dict | None = None


def _workspace_schema_tree(max_per_schema: int = 8) -> dict:
    """Map of schema → list of table names. Cached for the process lifetime;
    feeds the schema-preview panel in the right rail so users can craft
    questions without guessing what's available."""
    global _WORKSPACE_TREE_CACHE
    if _WORKSPACE_TREE_CACHE is not None:
        return _WORKSPACE_TREE_CACHE

    tree: dict = {}
    try:
        import psycopg2
        from config import DB_CONFIG
        conn = psycopg2.connect(connect_timeout=2, **DB_CONFIG)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
                    "ORDER BY table_schema, table_name"
                )
                for schema, table in cur.fetchall():
                    tree.setdefault(schema, []).append(table)
        finally:
            conn.close()
    except Exception:
        tree = {}

    _WORKSPACE_TREE_CACHE = tree
    return tree


def _created_label() -> str:
    return datetime.now().strftime("%b %d, %I:%M %p").replace(" 0", " ")


def _records_from_json(df_json: str | None) -> list[dict]:
    if not df_json:
        return []
    try:
        data = json.loads(df_json)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _data_preview(records: list[dict], max_rows: int = 5):
    if not records:
        return html.Div("No row preview available.", className="detail-empty")

    columns = list(records[0].keys())[:5]
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in columns])),
            html.Tbody(
                [
                    html.Tr([html.Td(str(row.get(col, ""))) for col in columns])
                    for row in records[:max_rows]
                ]
            ),
        ],
        className="data-preview-table",
    )


_PROGRESS_STAGES = [("sql", "SQL"), ("query", "Query"), ("chart", "Chart"), ("memo", "Memo")]


def _render_progress_chips(meta: dict | None):
    """Stage chips reflect the run state. Recognised meta keys:

    - sql/query/chart/memo : booleans, whether that stage *completed*
    - running              : optional stage key (e.g. "query") currently in flight
    - failed               : whether the run failed overall
    - kind                 : failure flavour (used to amber-tint the SQL chip on
                             schema misses or full SQL failures)
    """
    if not meta:
        return [
            html.Span(label, className="step-chip")
            for _, label in _PROGRESS_STAGES
        ]

    failed = bool(meta.get("failed"))
    running = meta.get("running")
    chips = []
    for key, label in _PROGRESS_STAGES:
        classes = ["step-chip"]
        if meta.get(key):
            classes.append("completed")
        elif key == running:
            classes.append("running")
        elif failed and key == "sql":
            classes.append("failed")
        chips.append(html.Span(label, className=" ".join(classes)))
    return chips


def _polish_figure(fig, title: str, featured: bool = False):
    # The chart card already shows the title in its own header; suppressing the
    # in-canvas Plotly title removes duplication and frees vertical space.
    fig.update_layout(
        template="agenticbi_dark",
        title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#171a1f",
        margin={"l": 62 if featured else 52, "r": 24, "t": 24 if featured else 18, "b": 58},
        font={"family": "Inter, Segoe UI, system-ui, sans-serif", "color": "#d6dde6"},
    )

    # Re-apply currency tick formatting after JSON deserialisation. The viz
    # agent sets tickprefix="$" but tickformat="~s" gets dropped through the
    # Plotly JSON round-trip, so we redo it here with values that survive.
    for axis_attr in ("xaxis", "yaxis"):
        axis = getattr(fig.layout, axis_attr, None)
        if axis is not None and getattr(axis, "tickprefix", "") == "$":
            axis.tickformat = "~s"
    fig.update_xaxes(
        automargin=True,
        gridcolor="rgba(148,163,184,0.10)",
        linecolor="rgba(148,163,184,0.20)",
        tickfont={"color": "#9aa7b5"},
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor="rgba(148,163,184,0.13)",
        linecolor="rgba(148,163,184,0.20)",
        tickfont={"color": "#9aa7b5"},
    )
    for trace_index, trace in enumerate(fig.data):
        if getattr(trace, "type", "") == "bar":
            trace.update(
                marker={
                    "line": {"width": 0},
                    "opacity": 0.9,
                },
                hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
            )
        elif getattr(trace, "type", "") in {"pie", "funnelarea"}:
            trace.update(
                textfont={"color": "#f5f7fa"},
                hovertemplate="<b>%{label}</b><br>%{value}<extra></extra>",
            )
        elif getattr(trace, "type", "") == "scatter":
            trace.update(
                line={"width": 3},
                marker={"size": 8, "line": {"width": 0}},
                hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
            )
    return fig


# ---------------------------------------------------------
# Custom Vizro component
# ---------------------------------------------------------
class AgenticBIPage(vm.VizroBaseModel):
    type: Literal["agentic_bi_page"] = "agentic_bi_page"

    def build(self):
        S = self.id  # id prefix

        return html.Div(
            [
                # ── Hidden state store (local persistence) ───────
                dcc.Store(
                    id=f"{S}_store",
                    data={"charts": [], "messages": [], "all_charts": [], "saved_queries": []},
                    storage_type="local",
                ),

                # ── CSV download sink — server callback writes here, the
                #    browser turns it into a file download.
                dcc.Download(id=f"{S}_csv_download"),

                # ── Live-progress poll: ticks while a pipeline is running so
                #    chips and Run-button text update mid-stream. Dormant
                #    when no run is in flight.
                dcc.Interval(
                    id=f"{S}_progress_tick",
                    interval=600,    # ms
                    disabled=False,
                    n_intervals=0,
                ),

                # ── Hidden tick counter — bumped each time the polling
                #    callback sees a real change. Used as a tracking signal.
                dcc.Store(id=f"{S}_progress_tick_seen", data=0),

                # ── Run-completed pulse — bumped by poll_progress when it
                #    detects the in-flight flag flip from True → False, so
                #    render_ui can refresh to pick up the new store data.
                dcc.Store(id=f"{S}_run_completed_tick", data=0),


                # ── Loading target (spinner removed) ───────────
                html.Div(
                    html.Div(
                        id=f"{S}_loading_target",
                        style={"minHeight": "1px", "position": "absolute", "top": "0", "left": "0"},
                    ),
                    id=f"{S}_loading",
                    style={"display": "none"},
                ),

                # ── Top command bar ────────────────────────────
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("AgenticBI", className="app-title"),
                                html.Span(
                                    id=f"{S}_workspace_badge",
                                    children="Connected",
                                    className="status-pill ok",
                                ),
                            ],
                            className="brand-block",
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    id=f"{S}_chat_input",
                                    type="text",
                                    placeholder="Ask anything about your data — try \"Top 10 products by revenue\"",
                                    debounce=False,
                                    n_submit=0,
                                    disabled=False,
                                    className="command-input",
                                    style={"width": "100%"},
                                ),
                                html.Kbd("/", className="kbd-hint"),
                                html.Button(
                                    "Run",
                                    id=f"{S}_send_btn",
                                    n_clicks=0,
                                    disabled=False,
                                    className="command-button",
                                ),
                            ],
                            className="command-box",
                        ),
                    ],
                    id=f"{S}_command_bar",
                    className="command-bar",
                ),

                # ── Main row: LEFT (charts) | RIGHT (insights) ──
                html.Div(
                    [
                        # ═══ LEFT PANEL — Dashboard ═══════════════
                        html.Div(
                            [
                                # Left panel header: label + filter + sort
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span("DASHBOARD", className="panel-eyebrow"),
                                                html.Div(
                                                    id=f"{S}_progress_chips",
                                                    children=_render_progress_chips(None),
                                                    className="progress-steps",
                                                ),
                                            ],
                                            className="panel-title-block",
                                        ),
                                        dcc.Input(
                                            id=f"{S}_filter_query",
                                            type="text",
                                            placeholder="Filter charts by title or query",
                                            debounce=True,
                                            className="filter-input",
                                            style={"width": "100%"},
                                        ),
                                        dcc.Dropdown(
                                            id=f"{S}_sort_by",
                                            options=[
                                                {"label": "Latest first", "value": "latest"},
                                                {"label": "A → Z",        "value": "az"},
                                                {"label": "Z → A",        "value": "za"},
                                            ],
                                            value="latest",
                                            clearable=False,
                                            className="sort-dropdown",
                                            style={"width": "150px", "fontSize": "12px"},
                                        ),
                                    ],
                                    className="panel-header",
                                ),
                                # Charts scroll area
                                html.Div(
                                    id=f"{S}_dashboard",
                                    style={
                                        "flex": "1",
                                        "overflowY": "auto",
                                        "padding": "16px",
                                    },
                                ),
                            ],
                            style={
                                "flex": "1",
                                "display": "flex",
                                "flexDirection": "column",
                                "background": BG_MAIN,
                                "minWidth": "0",
                                "height": "100%",
                            },
                        ),

                        # ── Vertical divider ─────────────────────
                        html.Div(
                            style={
                                "width": "1px",
                                "background": "#30353b",
                                "flexShrink": "0",
                            }
                        ),

                        # ═══ RIGHT PANEL — Insight timeline ═══════
                        html.Div(
                            [
                                # Chat header with clear button
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span("INSIGHTS", className="panel-eyebrow"),
                                                html.Span("Question-to-answer trail", className="panel-subtitle"),
                                            ],
                                            className="panel-title-block",
                                        ),
                                        html.Button(
                                            "Clear",
                                            id=f"{S}_clear_btn",
                                            n_clicks=0,
                                            className="subtle-button",
                                        ),
                                    ],
                                    className="panel-header insight-header",
                                ),
                                # Messages scroll area
                                html.Div(
                                    id=f"{S}_chat_messages",
                                    style={
                                        "flex": "1",
                                        "overflowY": "auto",
                                        "padding": "14px",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "gap": "10px",
                                        "fontFamily": FONT,
                                    },
                                ),
                            ],
                            className="chat-panel",
                            style={
                                "width": "36%",
                                "minWidth": "300px",
                                "maxWidth": "560px",
                                "flexShrink": "0",
                                "display": "flex",
                                "flexDirection": "column",
                                "background": BG_PANEL,
                                "height": "100%",
                            },
                        ),
                    ],
                    className="main-layout",
                    style={
                        "display": "flex",
                        "flex": "1",
                        "height": "100%",
                        "minHeight": "0",
                        "overflow": "hidden",
                    },
                ),

                # ── Undo toast container — slides up from bottom-center
                #    when a chart is soft-deleted. Both children are
                #    permanent so their callbacks bind at app-build time;
                #    render_ui controls visibility via className + text.
                html.Div(
                    [
                        html.Span(id=f"{S}_undo_text", className="undo-text"),
                        html.Button(
                            "Undo",
                            id=f"{S}_undo_delete",
                            n_clicks=0,
                            className="undo-btn",
                        ),
                    ],
                    id=f"{S}_undo_toast",
                    className="undo-toast hidden",
                ),

                # ── Notice toast — shown when a duplicate question is
                #    resubmitted; the action button scrolls the canvas to
                #    the existing chart card.
                html.Div(
                    [
                        html.Span(id=f"{S}_notice_text", className="notice-text"),
                        html.Button(
                            "Open chart",
                            id=f"{S}_notice_action",
                            n_clicks=0,
                            className="notice-btn",
                        ),
                    ],
                    id=f"{S}_notice_toast",
                    className="notice-toast hidden",
                ),
                # Hidden field that holds the chart_id the notice action should
                # scroll to — read by a clientside callback when the button fires.
                dcc.Store(id=f"{S}_notice_target", data=None),

                # ── Database Connection Modal (Mock) ────────────
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Available Databases", style={"margin": "0 0 16px 0", "color": "#e5e7eb", "fontSize": "16px"}),
                                        html.Button("✕", id=f"{S}_close_db_modal", className="subtle-button", style={"position": "absolute", "top": "12px", "right": "12px", "padding": "4px 8px"}),
                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.RadioItems(
                                            id=f"{S}_db_container_select",
                                            options=[
                                                {"label": "local_adventureworks (PostgreSQL)", "value": "local_adventureworks"},
                                                {"label": "local_adventureworks_test (PostgreSQL)", "value": "local_adventureworks_test"}
                                            ],
                                            value="local_adventureworks",
                                            className="db-radio-group",
                                        )
                                    ],
                                    style={"marginBottom": "20px"}
                                ),
                                html.Div(
                                    [
                                        html.Button("Connect", id=f"{S}_confirm_db_connect", n_clicks=0, className="connect-db-btn", style={"width": "100%"}),
                                    ]
                                ),
                                dcc.Loading(
                                    html.Div(id=f"{S}_db_connect_result", style={"marginTop": "16px", "color": "#4fd1a5", "textAlign": "center", "minHeight": "24px", "fontSize": "13px", "fontWeight": "500"}),
                                    type="circle",
                                    color="#4f8cff",
                                )
                            ],
                            className="modal-content",
                        )
                    ],
                    id=f"{S}_db_modal",
                    className="db-modal hidden",
                ),
            ],
            id=self.id,
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "width": "100vw",
                "background": BG_MAIN,
                "color": TEXT,
                "fontFamily": FONT,
                "overflow": "hidden",
                "position": "fixed",
                "top": "0",
                "left": "0",
            },
        )


vm.Page.add_type("components", AgenticBIPage)


# ---------------------------------------------------------
# Callback 1 — Send / Enter → pipeline → update Store
#
# Runs synchronously in the Flask request thread. While streaming through the
# LangGraph pipeline, it pushes per-stage updates to _PROGRESS_STATE which a
# separate dcc.Interval-driven callback polls and reflects in the chips +
# Run button. Avoids the macOS multiprocess fork issues entirely.
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data"),
    Output("agentic_bi_page_chat_input", "value"),
    Output("agentic_bi_page_chat_input", "disabled"),
    Output("agentic_bi_page_send_btn", "disabled"),
    Output("agentic_bi_page_send_btn", "children"),
    Output("agentic_bi_page_loading_target", "children"),
    Input("agentic_bi_page_send_btn", "n_clicks"),
    Input("agentic_bi_page_chat_input", "n_submit"),
    Input({"type": "suggested-query", "index": ALL}, "n_clicks"),
    Input({"type": "suggested-query-chat", "index": ALL}, "n_clicks"),
    Input({"type": "saved-query", "index": ALL}, "n_clicks"),
    State("agentic_bi_page_chat_input", "value"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_send(_clicks, _submit, _suggested_dashboard, _suggested_chat, _saved, user_text, store):
    # Pattern-matching ALL inputs fire spuriously during initial page
    # hydration with all n_clicks=0/None. Reject those: only proceed if at
    # least one of the actual click/submit signals carries a positive value.
    has_run_click = bool(_clicks)
    has_submit    = bool(_submit)
    has_dash_chip = any(c for c in (_suggested_dashboard or []) if c)
    has_chat_chip = any(c for c in (_suggested_chat or []) if c)
    has_saved     = any(c for c in (_saved or []) if c)
    if not (has_run_click or has_submit or has_dash_chip or has_chat_chip or has_saved):
        return no_update, no_update, no_update, no_update, no_update, no_update

    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    question = ""

    if triggered.startswith("{"):
        try:
            trigger_id = json.loads(triggered)
            ttype = trigger_id.get("type")
            if ttype in {"suggested-query", "suggested-query-chat"}:
                question = SUGGESTED_QUERIES[int(trigger_id["index"])]
            elif ttype == "saved-query":
                # The button's id index is the position in saved_queries; the
                # store still has the source-of-truth list.
                saved_qs = (store or {}).get("saved_queries", [])
                idx = int(trigger_id["index"])
                if 0 <= idx < len(saved_qs):
                    question = saved_qs[idx]
        except Exception:
            question = ""
    else:
        question = (user_text or "").strip()

    if not question:
        return no_update, no_update, no_update, no_update, no_update, no_update

    store    = store or {"charts": [], "messages": [], "all_charts": [], "saved_queries": []}
    charts   = list(store.get("charts", []))
    all_charts = list(store.get("all_charts", []))
    messages = list(store.get("messages", []))

    # Duplicate detection — if the same question already has a chart on the
    # canvas, surface a notice toast pointing at it instead of running the
    # pipeline again. The user can still re-run by clearing or asking a fresh
    # question; this just prevents the canvas filling with identical cards.
    norm_q = question.strip().lower()
    existing = next(
        (c for c in charts if (c.get("query") or "").strip().lower() == norm_q),
        None,
    )
    if existing:
        import time as _time
        patched = Patch()
        patched["notice"] = {
            "message": "You already have this chart on the canvas.",
            "chart_id": existing.get("id"),
            "expires_at": _time.time() + 6,
        }
        return patched, "", False, False, "Run", ""

    messages.append({"role": "user", "content": question, "timestamp": uuid.uuid4().hex[:8]})

    # Guardrails
    try:
        allowed, deny_msg = _guardrails()(question)
    except Exception:
        allowed, deny_msg = True, ""

    if not allowed:
        messages.append({
            "role": "assistant",
            "content": deny_msg,
            "meta": {"sql": False, "query": False, "chart": False, "memo": True, "failed": True, "kind": "blocked"},
        })
        return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Run", ""

    # Build conversation history for context
    history = [m["content"] for m in messages if m.get("role") == "user"]

    # --- Stream the orchestrator and push chip updates as stages complete ---
    stages_done: set = set()

    def _push_progress():
        running = _next_running(stages_done)
        meta = _streaming_meta(stages_done, running=running)
        _set_progress(meta, _STAGE_BUTTON_TEXT.get(running, "Generating…"), in_flight=True)

    # Initial state — SQL stage about to start
    _set_progress(_streaming_meta(set(), running="sql"), _STAGE_BUTTON_TEXT["sql"], in_flight=True)

    result: dict | None = None
    try:
        gen = _orchestrator().run_streaming(question, conversation_history=history)
        try:
            while True:
                node_name, snapshot = next(gen)
                stage = _NODE_TO_STAGE.get(node_name)

                # Per-stage success rules: only mark a stage complete when the
                # state carries the artifact that stage actually produced
                # (except memo — the generate_insights node always runs last,
                # so we mark it done whenever it fires).
                if stage == "sql" and snapshot.get("sql_query"):
                    stages_done.add("sql")
                elif stage == "query" and snapshot.get("sql_success") and snapshot.get("result_dict"):
                    stages_done.add("sql")
                    stages_done.add("query")
                elif stage == "chart" and snapshot.get("viz_success"):
                    stages_done.update({"sql", "query", "chart"})
                elif stage == "memo":
                    stages_done.update({"sql", "query", "chart", "memo"})

                _push_progress()
        except StopIteration as stop:
            result = stop.value
    except Exception as exc:
        _set_progress(None, "Run", in_flight=False)
        messages.append({
            "role": "assistant",
            "content": f"Pipeline error: {exc}",
            "meta": {"sql": False, "query": False, "chart": False, "memo": True, "failed": True, "kind": "crash"},
        })
        return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Run", ""

    if result is None:
        result = {"success": False, "error": "Orchestrator stream produced no result"}

    # Push a final chip snapshot reflecting how the run actually ended, then
    # flip in_flight=False. The next poll picks this up and bumps the
    # run_completed_tick so render_ui re-fires.
    raw_err = (result.get("error") or "").lower() if not result.get("success") else ""
    final_kind = None
    if not result.get("success"):
        final_kind = "schema_miss" if "insufficient schema context" in raw_err else "sql_failed"
    final_meta = _streaming_meta(
        stages_done,
        running=None,
        failed=not result.get("success"),
        kind=final_kind,
    )
    _set_progress(final_meta, "Run", in_flight=False)

    # Per-stage flags for the final memo meta
    has_sql   = bool(result.get("sql"))
    has_data  = result.get("df") is not None and len(result["df"]) > 0
    has_chart = result.get("figure") is not None
    has_memo  = bool(result.get("markdown"))

    if result.get("success") and has_chart:
        title = (result.get("chart_spec") or {}).get("title") or "Chart"
        chart_id = str(uuid.uuid4())[:8]
        chart_data = {
            "id":          chart_id,
            "query":       question,
            "title":       title,
            "figure_json": result["figure"].to_json(),
            "sql":         result.get("sql"),
            "df_json":     result["df"].to_json(orient="records") if result.get("df") is not None else None,
            "row_count":   int(len(result["df"])) if result.get("df") is not None else 0,
            "chart_type":  (result.get("chart_spec") or {}).get("chart_type"),
            "created_at":  _created_label(),
            "timestamp":   uuid.uuid4().hex[:8],
        }
        charts.insert(0, chart_data)
        all_charts.insert(0, chart_data)
        reply = result.get("markdown") or f"Chart ready: **{title}**"
        meta = {"sql": True, "query": True, "chart": True, "memo": has_memo, "failed": False, "kind": "ok"}

    elif result.get("success") and result.get("viz_failed"):
        reply = (
            "Query ran but a chart couldn't be rendered for these results.\n\n"
            + (result.get("viz_error") or "")
            + ("\n\n" + result["markdown"] if result.get("markdown") else "")
        )
        meta = {"sql": True, "query": True, "chart": False, "memo": has_memo, "failed": True, "kind": "viz_failed"}

    else:
        raw = result.get("error") or "Could not answer that. Try rephrasing."
        if "insufficient schema context" in raw.lower():
            reply = (
                "I couldn't answer this from the connected dataset.\n\n"
                "The columns or relationships you're asking about don't appear "
                "in the AdventureWorks schema I have indexed. Try rephrasing "
                "with terms closer to the data — for example, sales territory "
                "instead of acquisition channel."
            )
            kind = "schema_miss"
        else:
            reply = "Couldn't run that query.\n\n" + raw
            kind = "sql_failed"
        meta = {"sql": has_sql, "query": False, "chart": False, "memo": False, "failed": True, "kind": kind}

    messages.append({"role": "assistant", "content": reply, "meta": meta})
    return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Run", ""




# ---------------------------------------------------------
# Callback 1b — Clear chat
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input("agentic_bi_page_clear_btn", "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_clear(_clicks, store):
    store = store or {"charts": [], "messages": [], "all_charts": [], "saved_queries": []}
    return {**store, "charts": [], "messages": [], "all_charts": []}


# ---------------------------------------------------------
# Callback 1c — Delete chart
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input({"type": "delete-chart", "index": ALL}, "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_delete_chart(n_clicks, store):
    if not n_clicks or not any(n_clicks):
        return no_update
    store = store or {"charts": [], "messages": [], "all_charts": [], "saved_queries": []}
    charts = list(store.get("charts", []))
    all_charts = list(store.get("all_charts", []))
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    chart_id = json.loads(triggered)["index"]

    # Capture the chart and its position(s) for a 5-second undo window.
    target = next((c for c in charts if c.get("id") == chart_id), None) or \
             next((c for c in all_charts if c.get("id") == chart_id), None)
    if target is None:
        return no_update
    charts_idx = next((i for i, c in enumerate(charts) if c.get("id") == chart_id), -1)
    all_idx = next((i for i, c in enumerate(all_charts) if c.get("id") == chart_id), -1)

    import time as _time
    pending_delete = {
        "chart": target,
        "charts_index": charts_idx,
        "all_index": all_idx,
        "expires_at": _time.time() + 5,
    }

    charts = [c for c in charts if c.get("id") != chart_id]
    all_charts = [c for c in all_charts if c.get("id") != chart_id]
    return {
        **store,
        "charts": charts,
        "all_charts": all_charts,
        "pending_delete": pending_delete,
    }


# ---------------------------------------------------------
# Callback 1g — Undo a recent delete (within the 5s window)
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input("agentic_bi_page_undo_delete", "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_undo_delete(_clicks, store):
    if not _clicks:
        return no_update
    store = store or {}
    pd_state = store.get("pending_delete")
    if not pd_state:
        return no_update

    chart = pd_state.get("chart")
    if not chart:
        return no_update

    # Patch the only fields we care about so a concurrent prune / pin / star
    # write can't clobber the restored chart with a stale State snapshot.
    charts_idx = max(0, pd_state.get("charts_index", 0))
    all_idx = max(0, pd_state.get("all_index", 0))
    patched = Patch()
    patched["charts"].insert(charts_idx, chart)
    patched["all_charts"].insert(all_idx, chart)
    patched["pending_delete"] = None
    return patched


# ---------------------------------------------------------
# Callback 1h — Periodic prune of expired pending_delete
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input("agentic_bi_page_progress_tick", "n_intervals"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def prune_pending_delete(_n, store):
    # Use Patch so we only touch the pending_delete / notice fields —
    # otherwise a stale State snapshot from this 600ms-tick callback can race
    # with on_undo_delete and clobber the just-restored charts list.
    if not store:
        return no_update
    import time as _time
    now = _time.time()
    pd_state = store.get("pending_delete")
    notice = store.get("notice")

    pd_expired = pd_state and now >= pd_state.get("expires_at", 0)
    notice_expired = notice and now >= notice.get("expires_at", 0)
    if not (pd_expired or notice_expired):
        return no_update

    patched = Patch()
    if pd_expired:
        patched["pending_delete"] = None
    if notice_expired:
        patched["notice"] = None
    return patched


# ---------------------------------------------------------
# Callback 1d — Toggle star on a chart card → add/remove from saved_queries
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input({"type": "star-chart", "index": ALL}, "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_toggle_star(n_clicks, store):
    if not n_clicks or not any(c for c in (n_clicks or []) if c):
        return no_update
    store = store or {"charts": [], "messages": [], "all_charts": [], "saved_queries": []}
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    try:
        chart_id = json.loads(triggered)["index"]
    except Exception:
        return no_update

    # Find the chart's question (search both visible and historical lists)
    target_q = None
    for c in store.get("all_charts", []) + store.get("charts", []):
        if c.get("id") == chart_id:
            target_q = (c.get("query") or "").strip()
            break
    if not target_q:
        return no_update

    saved = list(store.get("saved_queries", []))
    if target_q.lower() in {q.lower() for q in saved}:
        saved = [q for q in saved if q.lower() != target_q.lower()]
    else:
        saved.insert(0, target_q)
    return {**store, "saved_queries": saved}


# ---------------------------------------------------------
# Callback 1f — Toggle pin on a chart card → sticky at top of dashboard
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input({"type": "pin-chart", "index": ALL}, "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_toggle_pin(n_clicks, store):
    if not n_clicks or not any(c for c in (n_clicks or []) if c):
        return no_update
    store = store or {"charts": [], "messages": [], "all_charts": [], "saved_queries": []}
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    try:
        chart_id = json.loads(triggered)["index"]
    except Exception:
        return no_update

    def _toggle(lst):
        out = []
        for c in lst:
            if c.get("id") == chart_id:
                c = {**c, "pinned": not c.get("pinned")}
            out.append(c)
        return out

    return {
        **store,
        "charts": _toggle(store.get("charts", [])),
        "all_charts": _toggle(store.get("all_charts", [])),
    }


# ---------------------------------------------------------
# Callback 1e — Remove a question from the saved-queries rail
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input({"type": "saved-remove", "index": ALL}, "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_remove_saved(n_clicks, store):
    if not n_clicks or not any(c for c in (n_clicks or []) if c):
        return no_update
    store = store or {"charts": [], "messages": [], "all_charts": [], "saved_queries": []}
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    try:
        idx = int(json.loads(triggered)["index"])
    except Exception:
        return no_update
    saved = list(store.get("saved_queries", []))
    if 0 <= idx < len(saved):
        saved.pop(idx)
    return {**store, "saved_queries": saved}


# ---------------------------------------------------------
# Callback 1i — Download chart data as CSV
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_csv_download", "data"),
    Input({"type": "export-csv", "index": ALL}, "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_export_csv(n_clicks, store):
    if not n_clicks or not any(c for c in (n_clicks or []) if c):
        return no_update
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    try:
        chart_id = json.loads(triggered)["index"]
    except Exception:
        return no_update

    store = store or {}
    chart = next(
        (c for c in (store.get("all_charts", []) or []) + (store.get("charts", []) or [])
         if c.get("id") == chart_id),
        None,
    )
    if not chart or not chart.get("df_json"):
        return no_update

    try:
        records = json.loads(chart["df_json"])
    except Exception:
        return no_update
    if not records:
        return no_update

    import io as _io
    import csv as _csv
    columns = list(records[0].keys())
    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

    safe_title = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in (chart.get("title") or "chart")).strip().replace(" ", "_")
    filename = f"{safe_title or 'chart'}.csv"
    return {"content": buf.getvalue(), "filename": filename, "type": "text/csv"}


# ---------------------------------------------------------
# Callback 1j — Download chart as PNG via Plotly's client-side API
# ---------------------------------------------------------
clientside_callback(
    """
    function(n_clicks_list) {
        if (!n_clicks_list || !n_clicks_list.some(c => c)) {
            return window.dash_clientside.no_update;
        }
        const ctx = window.dash_clientside.callback_context;
        const trig = ctx.triggered && ctx.triggered[0];
        if (!trig || !trig.value) return window.dash_clientside.no_update;
        let chartId;
        try { chartId = JSON.parse(trig.prop_id.split('.')[0]).index; }
        catch (e) { return window.dash_clientside.no_update; }

        // The chart graph wrapper is the Plotly node whose id encodes the
        // matching index. We find it by scanning all .js-plotly-plot nodes
        // and matching their parent id (Dash assigns id={"type":"chart-graph","index":...}).
        const all = document.querySelectorAll('.js-plotly-plot');
        let target = null;
        all.forEach(node => {
            // Walk up to find the Dash component wrapper id
            let p = node;
            while (p && p.id !== '' && !p.id.includes('chart-graph')) {
                p = p.parentElement;
            }
            if (p && p.id && p.id.includes('"index":"' + chartId + '"')) {
                target = node;
            }
        });
        if (!target) {
            // Fallback: just take the most-recently rendered Plotly node
            target = all[all.length - 1] || null;
        }
        if (target && window.Plotly) {
            const safe = chartId.replace(/[^a-zA-Z0-9_-]/g, '_');
            window.Plotly.downloadImage(target, {
                format: 'png',
                filename: 'chart_' + safe,
                width: target.clientWidth || 1200,
                height: target.clientHeight || 600,
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("agentic_bi_page_csv_download", "data", allow_duplicate=True),
    Input({"type": "export-png", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------
# Callback 2 — Store / filter / sort → render UI
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_dashboard", "children"),
    Output("agentic_bi_page_chat_messages", "children"),
    Output("agentic_bi_page_workspace_badge", "children"),
    Output("agentic_bi_page_workspace_badge", "className"),
    Output("agentic_bi_page_undo_text", "children"),
    Output("agentic_bi_page_undo_toast", "className"),
    Output("agentic_bi_page_notice_text", "children"),
    Output("agentic_bi_page_notice_toast", "className"),
    Output("agentic_bi_page_notice_target", "data"),
    Input("agentic_bi_page_store", "data"),
    Input("agentic_bi_page_filter_query", "value"),
    Input("agentic_bi_page_sort_by", "value"),
    Input("agentic_bi_page_run_completed_tick", "data"),
)
def render_ui(store, filter_q, sort_by, _completed_tick):
    store    = store or {"charts": [], "messages": [], "all_charts": [], "saved_queries": []}
    all_charts = list(store.get("all_charts", []))
    charts   = list(store.get("charts", []))
    messages = list(store.get("messages", []))

    # Track last assistant message's meta — used for the workspace badge
    # and the failed-memo styling. Chips are now owned solely by the polling
    # callback (live during runs, default afterward).
    last_assistant_meta = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant_meta = m.get("meta")
            break

    # Filter
    fq = (filter_q or "").strip().lower()
    if fq:
        charts = [
            c for c in charts
            if fq in (c.get("query") or "").lower()
            or fq in (c.get("title") or "").lower()
        ]

    # Sort by chosen order, then stable-partition pinned charts to the top.
    if sort_by == "az":
        charts = sorted(charts, key=lambda c: (c.get("title") or "").lower())
    elif sort_by == "za":
        charts = sorted(charts, key=lambda c: (c.get("title") or "").lower(), reverse=True)
    pinned = [c for c in charts if c.get("pinned")]
    unpinned = [c for c in charts if not c.get("pinned")]
    charts = pinned + unpinned

    # Single live status row: chart count + last-run summary. Replaces the
    # previous 4-up KPI grid (which mixed counters with an event card).
    kpi_strip = None
    if all_charts:
        latest = all_charts[0]
        latest_meta = (last_assistant_meta or {})
        latest_failed = bool(latest_meta.get("failed"))
        kpi_strip = html.Div(
            [
                html.Span(f"{len(all_charts)} chart{'s' if len(all_charts) != 1 else ''}", className="status-counter"),
                html.Span("·", className="status-divider"),
                html.Span(
                    [
                        html.Span("Last run · ", className="status-key"),
                        html.Strong(latest.get("title") or "Ready", className="status-value"),
                        html.Span(
                            f"  ({latest.get('row_count', 0)} rows · {latest.get('created_at', 'just now')})",
                            className="status-meta",
                        ),
                    ],
                    className="status-last",
                ),
                html.Span(
                    "Most recent question failed" if latest_failed else "All recent runs OK",
                    className=f"status-health {'warn' if latest_failed else 'ok'}",
                ),
            ],
            className="status-strip",
        )

    # ── Dashboard ───────────────────────────────────────
    if not charts:
        if all_charts and fq:
            # Charts exist but filter hides them
            empty_msg = f'No charts match "{fq}". Try a different filter.'
            dash_out = [
                kpi_strip,
                html.Div(
                    [
                        html.Div("No matching charts", className="empty-title"),
                        html.Div(
                            empty_msg,
                            className="empty-copy",
                        ),
                    ],
                    className="empty-panel",
                )
            ]
        else:
            # Truly empty — show proof of life + suggested queries
            info = _workspace_info()
            if info["connected"]:
                conn_pills = [
                    html.Span([html.Span("●", className="dot ok"), info["db"] or "postgres"], className="conn-pill"),
                    html.Span(f"{info['tables']} tables", className="conn-pill muted"),
                    html.Span(f"{info['schemas']} schemas", className="conn-pill muted"),
                ]
            else:
                conn_pills = [
                    html.Span([html.Span("●", className="dot err"), "Database unreachable"], className="conn-pill error"),
                    html.Span(info.get("error") or "", className="conn-pill muted"),
                ]
            dash_out = [
                html.Div(
                    [
                        html.Div(conn_pills, className="conn-row"),
                        html.Div("Ask anything about your data.", className="empty-title"),
                        html.Div(
                            "AgenticBI generates SQL, runs the query, draws the chart, and writes a short narrative — all from one prompt.",
                            className="empty-copy",
                        ),
                        html.Div("Try one of these to see the pipeline in action:", className="empty-cta"),
                        html.Div(
                            [
                                html.Button(
                                    q,
                                    id={"type": "suggested-query", "index": i},
                                    n_clicks=0,
                                    className="suggested-chip",
                                )
                                for i, q in enumerate(SUGGESTED_QUERIES)
                            ],
                            className="suggestion-row",
                        ),
                    ],
                    className="empty-panel",
                ),
                html.Div(
                    [
                        html.Button(
                            "Connect Database",
                            id="agentic_bi_page_open_db_modal",
                            n_clicks=0,
                            className="connect-db-btn",
                        ),
                        html.Div(
                            id="agentic_bi_page_main_db_status",
                            style={"marginTop": "16px", "color": "#4fd1a5", "fontSize": "14px", "fontWeight": "600"}
                        )
                    ],
                    style={"textAlign": "center", "width": "100%", "marginTop": "24px"}
                )
            ]
    else:
        saved_set = {q.lower() for q in store.get("saved_queries", [])}
        chart_cards = []
        for i, c in enumerate(charts):
            try:
                fig = pio.from_json(c["figure_json"])
                featured = i == 0
                title = c.get("title") or "Chart"
                fig = _polish_figure(fig, title, featured=featured)
                chart_id = c.get("id", "")
                records = _records_from_json(c.get("df_json"))
                row_count = c.get("row_count")
                if row_count is None:
                    row_count = len(records)
                chart_type = (c.get("chart_type") or "chart").replace("_", " ").title()
                question_text = c.get("query") or ""
                is_saved = question_text.lower() in saved_set
                chart_cards.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(title, className="chart-title"),
                                            html.Div(
                                                [
                                                    html.Span(chart_type, className="chart-meta-label"),
                                                    html.Span(f"{row_count} rows", className="chart-meta-label neutral"),
                                                    html.Span(question_text, className="chart-meta-query"),
                                                ],
                                                className="chart-meta",
                                            ),
                                        ],
                                        className="chart-title-block",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "★" if is_saved else "☆",
                                                id={"type": "star-chart", "index": chart_id},
                                                n_clicks=0,
                                                title="Remove from saved" if is_saved else "Save question",
                                                className=f"chart-action-button star{' is-saved' if is_saved else ''}",
                                            ),
                                            html.Button(
                                                "📌" if c.get("pinned") else "📍",
                                                id={"type": "pin-chart", "index": chart_id},
                                                n_clicks=0,
                                                title="Unpin from top" if c.get("pinned") else "Pin to top",
                                                className=f"chart-action-button pin{' is-pinned' if c.get('pinned') else ''}",
                                            ),
                                            html.Button(
                                                "CSV",
                                                id={"type": "export-csv", "index": chart_id},
                                                n_clicks=0,
                                                title="Download rows as CSV",
                                                className="chart-action-button text-button",
                                            ),
                                            html.Button(
                                                "PNG",
                                                id={"type": "export-png", "index": chart_id},
                                                n_clicks=0,
                                                title="Download chart as PNG",
                                                className="chart-action-button text-button",
                                            ),
                                            html.Button(
                                                "✕",
                                                id={"type": "delete-chart", "index": chart_id},
                                                n_clicks=0,
                                                title="Remove from dashboard",
                                                className="chart-action-button danger",
                                            ),
                                        ],
                                        className="chart-actions",
                                    ),
                                ],
                                className="chart-card-header",
                            ),
                            html.Div(
                                dcc.Graph(
                                    id={"type": "chart-graph", "index": chart_id},
                                    figure=fig,
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                        "toImageButtonOptions": {
                                            "format": "png",
                                            "filename": f"chart_{chart_id}",
                                        },
                                    },
                                    style={
                                        "height": "390px" if featured else "250px",
                                        "minHeight": "300px" if featured else "220px",
                                    },
                                ),
                                className="chart-figure-wrap",
                            ),
                            html.Div(
                                [
                                    html.Details(
                                        [
                                            html.Summary("SQL"),
                                            html.Pre(c.get("sql") or "No SQL available."),
                                        ],
                                        className="detail-drawer",
                                    ),
                                    html.Details(
                                        [
                                            html.Summary("Data preview"),
                                            _data_preview(records),
                                        ],
                                        className="detail-drawer",
                                    ),
                                ],
                                className="chart-drawers",
                            ),
                        ],
                        className=" ".join(filter(None, [
                            "chart-card",
                            "featured-chart-card" if featured else "",
                            "pinned-chart" if c.get("pinned") else "",
                        ])),
                    )
                )
            except Exception:
                pass
        dash_out = [kpi_strip, html.Div(chart_cards, className="chart-grid")]

    # ── Saved questions section (above memos) ────────────
    saved_queries = store.get("saved_queries", []) or []
    saved_panel = None
    if saved_queries:
        saved_chips = []
        for i, q in enumerate(saved_queries):
            label = q if len(q) <= 60 else q[:57].rstrip() + "…"
            saved_chips.append(
                html.Div(
                    [
                        html.Button(
                            label,
                            id={"type": "saved-query", "index": i},
                            n_clicks=0,
                            title=f"Re-run · {q}",
                            className="saved-chip",
                        ),
                        html.Button(
                            "✕",
                            id={"type": "saved-remove", "index": i},
                            n_clicks=0,
                            title="Remove from saved",
                            className="saved-chip-remove",
                        ),
                    ],
                    className="saved-chip-wrap",
                )
            )
        saved_panel = html.Div(
            [
                html.Div(
                    [
                        html.Span(f"SAVED · {len(saved_queries)}", className="saved-eyebrow"),
                        html.Span("click to re-run", className="saved-hint"),
                    ],
                    className="saved-header",
                ),
                html.Div(saved_chips, className="saved-list"),
            ],
            className="saved-panel",
        )

    # ── Schema preview — collapsible per-schema list of tables ───────
    schema_panel = None
    schema_tree = _workspace_schema_tree()
    if schema_tree:
        schema_sections = []
        for schema_name, tables in schema_tree.items():
            shown = tables[:8]
            more = len(tables) - len(shown)
            schema_sections.append(
                html.Details(
                    [
                        html.Summary(
                            [
                                html.Span(schema_name, className="schema-name"),
                                html.Span(str(len(tables)), className="schema-count"),
                            ],
                            className="schema-summary",
                        ),
                        html.Div(
                            [html.Span(t, className="schema-table") for t in shown]
                            + ([html.Span(f"+ {more} more", className="schema-table more")] if more > 0 else []),
                            className="schema-tables",
                        ),
                    ],
                    className="schema-section",
                )
            )
        schema_panel = html.Div(
            [
                html.Div(
                    [
                        html.Span("SCHEMA", className="schema-eyebrow"),
                        html.Span("AdventureWorks", className="schema-hint"),
                    ],
                    className="schema-header",
                ),
                html.Div(schema_sections, className="schema-list"),
            ],
            className="schema-panel",
        )

    # ── Chat messages ────────────────────────────────────
    if not messages:
        chat_out = [
            html.Div(
                [
                    html.Div("How the trail works", className="memo-title"),
                    html.Div(
                        "Each question generates SQL, runs it, draws a chart, and writes a short narrative — they all land here. The latest sits on top; older runs collapse to one line and expand on click.",
                        className="memo-copy",
                    ),
                    html.Div(
                        [
                            html.Span("01 ", className="step-num"), "Question is parsed and grounded in the schema.",
                        ],
                        className="memo-step",
                    ),
                    html.Div(
                        [
                            html.Span("02 ", className="step-num"), "SQL is generated and executed against Postgres.",
                        ],
                        className="memo-step",
                    ),
                    html.Div(
                        [
                            html.Span("03 ", className="step-num"), "Chart is drawn from the result and a memo is written.",
                        ],
                        className="memo-step",
                    ),
                ],
                className="welcome-panel",
            )
        ]
    else:
        chat_out = []
        exchanges = []
        pending_question = None
        for m in messages:
            if m.get("role") == "user":
                pending_question = m.get("content", "")
            elif m.get("role") == "assistant":
                exchanges.append({
                    "question": pending_question,
                    "answer": m.get("content", ""),
                    "meta": m.get("meta") or {},
                })
                pending_question = None

        for idx, exchange in enumerate(reversed(exchanges)):
            is_latest = idx == 0
            meta = exchange.get("meta") or {}
            failed = bool(meta.get("failed"))

            card_classes = ["memo-card"]
            card_classes.append("primary-memo" if is_latest else "history-memo")
            if failed:
                card_classes.append("failed-memo")

            if failed:
                kind = meta.get("kind") or "failed"
                status_label = {
                    "schema_miss": "Out of scope",
                    "sql_failed":  "Query failed",
                    "viz_failed":  "Chart not rendered",
                    "blocked":     "Blocked",
                    "crash":       "Pipeline error",
                }.get(kind, "Failed")
                status_class = "memo-status failed"
            else:
                status_label = "Generated from live SQL"
                status_class = "memo-status"

            question_text = exchange.get("question") or "Question unavailable"
            answer_md = exchange.get("answer") or ""

            # Latest memo renders fully expanded; older memos collapse to a
            # one-line summary that opens on click.
            if is_latest:
                body = [
                    html.Div(
                        [
                            html.Span("Latest memo", className="message-label"),
                            html.Span(status_label, className=status_class),
                        ],
                        className="memo-card-topline",
                    ),
                    html.Div(question_text, className="memo-question"),
                    dcc.Markdown(answer_md, className="insight-markdown"),
                ]
                card = html.Div(body, className=" ".join(card_classes))
            else:
                preview = (question_text or "").strip()
                if len(preview) > 80:
                    preview = preview[:77] + "…"
                card = html.Details(
                    [
                        html.Summary(
                            [
                                html.Span(preview, className="history-question"),
                                html.Span(status_label, className=f"{status_class} small"),
                            ],
                            className="history-summary",
                        ),
                        dcc.Markdown(answer_md, className="insight-markdown muted"),
                    ],
                    className=" ".join(card_classes + ["collapsed-memo"]),
                )

            chat_out.append(card)

    # Prepend the saved-queries panel above memos / welcome.
    if saved_panel is not None:
        chat_out = [saved_panel] + chat_out
    if schema_panel is not None:
        # Schema preview goes at the bottom — fills the dead space below the
        # memo trail without competing with the memos themselves for attention.
        chat_out = chat_out + [schema_panel]

    # Workspace badge — reflects connection + most recent run state
    info = _workspace_info()
    if not info["connected"]:
        badge_text = "DB unreachable"
        badge_class = "status-pill err"
    elif last_assistant_meta and last_assistant_meta.get("failed"):
        badge_text = "Last run failed"
        badge_class = "status-pill warn"
    else:
        badge_text = "Connected"
        badge_class = "status-pill ok"

    # Undo toast — visible only while pending_delete exists and hasn't expired.
    pd_state = store.get("pending_delete")
    toast_text = no_update
    toast_class = "undo-toast hidden"
    if pd_state:
        import time as _time
        if _time.time() < pd_state.get("expires_at", 0):
            chart = pd_state.get("chart") or {}
            title = chart.get("title") or "chart"
            toast_text = [html.Strong(title), " removed."]
            toast_class = "undo-toast"

    # Notice toast — surfaces duplicate-detection result. Distinct from undo.
    notice = store.get("notice")
    notice_text = no_update
    notice_class = "notice-toast hidden"
    notice_target = no_update
    if notice:
        import time as _time
        if _time.time() < notice.get("expires_at", 0):
            notice_text = notice.get("message") or ""
            notice_class = "notice-toast"
            notice_target = notice.get("chart_id")

    return (
        dash_out, chat_out, badge_text, badge_class,
        toast_text, toast_class,
        notice_text, notice_class, notice_target,
    )


# ---------------------------------------------------------
# Live-progress polling — ticks while a run is in flight, pushes chip + button
# updates from the shared in-process state dict.
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_progress_chips", "children"),
    Output("agentic_bi_page_send_btn", "children", allow_duplicate=True),
    Output("agentic_bi_page_progress_tick_seen", "data"),
    Output("agentic_bi_page_run_completed_tick", "data"),
    Input("agentic_bi_page_progress_tick", "n_intervals"),
    State("agentic_bi_page_progress_tick_seen", "data"),
    State("agentic_bi_page_run_completed_tick", "data"),
    prevent_initial_call=True,
)
def poll_progress(_n, last_seen, completed_tick):
    snap = _get_progress_snapshot()
    if snap["tick"] == (last_seen or 0):
        return no_update, no_update, no_update, no_update

    # Run just finished — write the final chip state and bump the completion
    # tick so render_ui re-reads the store and refreshes the dashboard.
    if not snap["in_flight"]:
        chips = _render_progress_chips(snap["meta"]) if snap["meta"] else no_update
        return chips, "Run", snap["tick"], (completed_tick or 0) + 1

    return _render_progress_chips(snap["meta"]), snap["btn_text"], snap["tick"], no_update


# ---------------------------------------------------------
# Clientside callbacks — instant feedback that doesn't need server roundtrip
# ---------------------------------------------------------

# 1) On Run click: instantly disable input/button and narrate the button.
#    Adds .is-running to the chips container only briefly — once the worker
#    starts streaming, set_progress overwrites the chip children with real
#    per-stage state and the per-chip .running animation takes over.
clientside_callback(
    """
    function(n_clicks, n_submit) {
        if (!n_clicks && !n_submit) {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        const chips = document.getElementById('agentic_bi_page_progress_chips');
        if (chips) chips.classList.add('is-running');
        return [true, true, 'Generating…'];
    }
    """,
    [
        Output("agentic_bi_page_chat_input", "disabled", allow_duplicate=True),
        Output("agentic_bi_page_send_btn", "disabled", allow_duplicate=True),
        Output("agentic_bi_page_send_btn", "children", allow_duplicate=True),
    ],
    [
        Input("agentic_bi_page_send_btn", "n_clicks"),
        Input("agentic_bi_page_chat_input", "n_submit"),
    ],
    prevent_initial_call=True,
)

# 2) When chips children update (set_progress or final render), remove the
#    is-running spin-up class so it doesn't compete with real .running chips.
clientside_callback(
    """
    function(_children) {
        const chips = document.getElementById('agentic_bi_page_progress_chips');
        if (chips) chips.classList.remove('is-running');
        return '';
    }
    """,
    Output("agentic_bi_page_progress_chips", "title"),
    Input("agentic_bi_page_progress_chips", "children"),
    prevent_initial_call=True,
)

# 2.5) Notice action — scroll the existing duplicate chart into view.
clientside_callback(
    """
    function(_n, target) {
        if (!_n || !target) return window.dash_clientside.no_update;
        const sel = '[id*="\\\\\"chart-graph\\\\\"\\\\,\\\\\"index\\\\\":\\\\\"' + target + '\\\\\"]';
        // Simpler: find Plotly graph wrappers, match by id substring
        const all = document.querySelectorAll('[id*="chart-graph"]');
        let target_node = null;
        all.forEach(n => {
            if (n.id && n.id.indexOf('"index":"' + target + '"') !== -1) {
                target_node = n;
            }
        });
        const card = target_node ? target_node.closest('.chart-card') : null;
        if (card) {
            card.scrollIntoView({behavior: 'smooth', block: 'center'});
            card.classList.add('flash-pulse');
            setTimeout(() => card.classList.remove('flash-pulse'), 1500);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("agentic_bi_page_notice_target", "data", allow_duplicate=True),
    Input("agentic_bi_page_notice_action", "n_clicks"),
    State("agentic_bi_page_notice_target", "data"),
    prevent_initial_call=True,
)


# 3) Keyboard shortcut: pressing "/" anywhere outside an input focuses the
#    question box. Registered once via a module-scoped guard.
# 2.7) Skeleton shimmer — when the chips container has any .running chip,
#      add `is-pipeline-running` to the dashboard div. CSS draws a shimmering
#      placeholder card at the top so the canvas reflects the in-flight run.
clientside_callback(
    """
    function() {
        if (window.__agenticbi_skeleton_observer) return window.dash_clientside.no_update;
        window.__agenticbi_skeleton_observer = true;
        const start = setInterval(function() {
            const chips = document.getElementById('agentic_bi_page_progress_chips');
            const dash = document.getElementById('agentic_bi_page_dashboard');
            if (!chips || !dash) return;
            clearInterval(start);
            const update = function() {
                const running = chips.querySelector('.step-chip.running') !== null;
                dash.classList.toggle('is-pipeline-running', running);
            };
            new MutationObserver(update).observe(chips, {
                childList: true, subtree: true, attributes: true,
                attributeFilter: ['class']
            });
            update();
        }, 60);
        return window.dash_clientside.no_update;
    }
    """,
    Output("agentic_bi_page_progress_chips", "style"),
    Input("agentic_bi_page_store", "data"),
    prevent_initial_call=False,
)


clientside_callback(
    """
    function(_data) {
        if (window.__agenticbi_kb_bound) return '';
        window.__agenticbi_kb_bound = true;
        document.addEventListener('keydown', function(e) {
            if (e.key !== '/') return;
            const tag = (document.activeElement && document.activeElement.tagName) || '';
            if (tag === 'INPUT' || tag === 'TEXTAREA') return;
            const input = document.getElementById('agentic_bi_page_chat_input');
            if (input) {
                e.preventDefault();
                input.focus();
                input.select && input.select();
            }
        });
        return '';
    }
    """,
    Output("agentic_bi_page_chat_input", "title"),
    Input("agentic_bi_page_store", "data"),
)




# ---------------------------------------------------------
# Mock Database Connection Callbacks
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_db_modal", "className"),
    Input("agentic_bi_page_open_db_modal", "n_clicks"),
    Input("agentic_bi_page_close_db_modal", "n_clicks"),
    State("agentic_bi_page_db_modal", "className"),
    prevent_initial_call=True,
)
def toggle_db_modal(open_clicks, close_clicks, current_class):
    ctx = callback_context
    if not ctx.triggered:
        return current_class
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_id == "agentic_bi_page_open_db_modal":
        if open_clicks and open_clicks > 0:
            return "db-modal"
        return current_class
    else:
        if close_clicks and close_clicks > 0:
            return "db-modal hidden"
        return current_class

@callback(
    Output("agentic_bi_page_db_connect_result", "children"),
    Output("agentic_bi_page_main_db_status", "children"),
    Output("agentic_bi_page_db_modal", "className", allow_duplicate=True),
    Input("agentic_bi_page_confirm_db_connect", "n_clicks"),
    State("agentic_bi_page_db_container_select", "value"),
    prevent_initial_call=True,
)
def mock_db_connection(n_clicks, selected_db):
    if not n_clicks:
        return no_update, no_update, no_update
    import time
    time.sleep(3) # Mock the loader delay
    msg = f"Database '{selected_db}' Successfully Connected"
    return msg, msg, "db-modal hidden"


# ---------------------------------------------------------
# Build and run
# ---------------------------------------------------------
def run_app(host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
    page = vm.Page(
        title="",          # suppress Vizro page title heading
        layout=vm.Flex(),
        components=[AgenticBIPage(id="agentic_bi_page")],
    )
    dashboard = vm.Dashboard(
        pages=[page],
        title="",          # suppress Vizro dashboard header title
    )
    app = Vizro(assets_folder=_ASSETS_DIR).build(dashboard)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_app(debug=os.environ.get("DEBUG", "0") == "1")
