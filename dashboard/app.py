"""Streamlit dashboard for the prompt A/B platform.

It only talks to the API (default http://127.0.0.1:8000) and never touches the database.
Run:  uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import difflib
import html
import math
import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

TEXT, MUTED, LINE = "#E6EDF3", "#8B98A9", "#2A3441"
TEAL, GREEN, AMBER = "#2DD4BF", "#34D399", "#FBBF24"
# One fixed colour per variant position (the control is always the first), on every chart.
VARIANT_COLORS = ["#60A5FA", "#F59E0B", "#A78BFA", "#F472B6", "#22D3EE", "#94A3B8"]

METRICS = {
    "judge_quality": "Judge quality score (1-5)",
    "json_bare": "Bare JSON rate (no code fences)",
    "json_valid": "Valid JSON rate",
    "latency_ms": "Latency (ms)",
    "cost_usd": "Cost per request (USD)",
    "output_tokens": "Output tokens",
    "input_tokens": "Input tokens",
}
METRIC_ALIASES = {"quality": "judge_quality"}
VERDICT_TEXT = {
    "pending": "pending",
    "better": "better",
    "worse": "worse",
    "no_difference": "no difference",
    "insufficient_data": "not enough data",
}
BADGES = {
    "draft": ("#8B98A9", "rgba(139,152,169,0.16)"),
    "running": (TEAL, "rgba(45,212,191,0.16)"),
    "completed": (TEAL, "rgba(45,212,191,0.16)"),
    "cancelled": (AMBER, "rgba(251,191,36,0.16)"),
}
BANNERS = {
    "winner": (GREEN, "rgba(52,211,153,0.11)"),  # green is reserved for a declared winner
    "interim": (AMBER, "rgba(251,191,36,0.10)"),
    "neutral": ("#8B98A9", "rgba(139,152,169,0.10)"),
}

CSS = """
<style>
.card { background:#1A212B; border:1px solid #2A3441; border-radius:12px; padding:14px 16px; }
.card .label { color:#8B98A9; font-size:0.8rem; }
.card .value { font-size:1.7rem; font-weight:600; line-height:1.35; }
.card .sub { color:#8B98A9; font-size:0.78rem; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; }
.banner { border-radius:12px; padding:14px 18px; border-left:5px solid; margin:8px 0 16px; }
.banner .title { display:block; font-size:1.05rem; font-weight:600; margin-bottom:2px; }
.banner .body { font-size:0.92rem; }
</style>
"""


class ApiError(Exception):
    """The API could not be reached, or it answered with an error."""


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------


def _request(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
    except httpx.HTTPError as exc:
        raise ApiError(
            f"Cannot reach the API at {API_URL}. Is the server running? ({exc})"
        ) from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"{response.status_code}: {detail}")
    return response.json()


@st.cache_data(ttl=5, show_spinner=False)
def api_get(path: str, params: tuple = ()):
    """GET with a short cache, so moving around the dashboard does not hammer the API."""
    return _request("GET", path, params=dict(params))


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------


def esc(value: object) -> str:
    return html.escape(str(value))


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.4g}" if signed else f"{value:.4g}"


def when(value: str) -> str:
    return value[:19].replace("T", " ")


def color_for(index: int) -> str:
    return VARIANT_COLORS[index % len(VARIANT_COLORS)]


def badge(status: str) -> str:
    fg, bg = BADGES.get(status, BADGES["draft"])
    return f'<span class="badge" style="color:{fg};background:{bg}">{esc(status)}</span>'


def card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="card"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
    )


def banner(kind: str, title: str, body: str) -> None:
    accent, background = BANNERS[kind]
    st.markdown(
        f'<div class="banner" style="border-color:{accent};background:{background}">'
        f'<span class="title" style="color:{accent}">{esc(title)}</span>'
        f'<span class="body">{esc(body)}</span></div>',
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


def ci_half_width(std: float | None, n: int) -> float:
    """Half-width of a 95% confidence interval for one variant's mean."""
    if std is None or n < 2:
        return 0.0
    return float(sps.t.ppf(0.975, n - 1)) * std / math.sqrt(n)


def mean_bar_chart(variants: list[dict]) -> go.Figure:
    labels, means, errors, colors = [], [], [], []
    for i, v in enumerate(variants):
        if v["mean"] is None:
            continue
        labels.append(v["label"])
        means.append(v["mean"])
        errors.append(ci_half_width(v["std"], v["n"]))
        colors.append(color_for(i))
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=means,
            marker_color=colors,
            error_y=dict(type="data", array=errors, color=TEXT, thickness=1.5, width=6),
        )
    )
    return style_figure(fig)


def running_mean_chart(series: dict) -> go.Figure:
    fig = go.Figure()
    for i, variant in enumerate(series["variants"]):
        points = variant["points"]
        fig.add_trace(
            go.Scatter(
                x=[p["n"] for p in points],
                y=[p["running_mean"] for p in points],
                mode="lines",
                name=variant["label"],
                line=dict(color=color_for(i), width=2),
            )
        )
    fig.update_xaxes(title_text="Observations per variant")
    return style_figure(fig)


def variants_table(results: dict) -> pd.DataFrame:
    comparison = {c["variant_id"]: c for c in results["comparisons"]}
    rows = []
    for i, v in enumerate(results["variants"]):
        c = comparison.get(v["variant_id"])
        low_high = "-" if c is None or c["ci_low"] is None else f"{fmt(c['ci_low'])} to {fmt(c['ci_high'])}"
        rows.append(
            {
                "Variant": v["label"] + (" (control)" if i == 0 else ""),
                "Samples": v["n"],
                "Mean": fmt(v["mean"]),
                "Std": fmt(v["std"]),
                "Still needed": v["samples_needed"],
                "Verdict": "control" if c is None else VERDICT_TEXT.get(c["verdict"], c["verdict"]),
                "Difference": "-" if c is None else fmt(c["diff"], signed=True),
                "95% CI of difference": low_high,
                "p (adjusted)": "-" if c is None else fmt(c["p_adjusted"]),
            }
        )
    return pd.DataFrame(rows)


def operations_table(ops: dict) -> pd.DataFrame:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.0f}%"

    return pd.DataFrame(
        [
            {
                "Variant": v["label"],
                "Requests": v["requests"],
                "Errors": v["errors"],
                "Mean latency (ms)": fmt(v["mean_latency_ms"]),
                "p95 latency (ms)": fmt(v["p95_latency_ms"]),
                "Mean cost (USD)": fmt(v["mean_cost_usd"]),
                "Total cost (USD)": fmt(v["total_cost_usd"]),
                "Valid JSON": pct(v["json_valid_rate"]),
                "Bare JSON": pct(v["json_bare_rate"]),
                "Judged": v["judge_n"],
                "Judge mean": fmt(v["judge_mean"]),
                "Judge failures": v["judge_failures"],
                "Judge cost (USD)": fmt(v["judge_cost_usd"]),
            }
            for v in ops["variants"]
        ]
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def open_experiment(experiment_id: int) -> None:
    # A button callback runs before the next rerun, so it may set widget state safely.
    st.session_state["page"] = "Experiment detail"
    st.session_state["exp_id"] = experiment_id


def experiments_page() -> None:
    st.title("Experiments")
    experiments = api_get("/experiments")
    if not experiments:
        st.info(
            "No experiments yet. Create one with POST /experiments on the API docs page "
            f"({API_URL}/docs)."
        )
        return
    for e in experiments:
        assigned = [v["assigned_users"] for v in e["variants"]]
        target = e["target_sample_size"]
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([3, 2, 4, 1])
            c1.markdown(f"**#{e['id']} {e['name']}**")
            c1.caption(f"Prompt {e['prompt_id']}: " + ", ".join(v["label"] for v in e["variants"]))
            c2.markdown(badge(e["status"]), unsafe_allow_html=True)
            c2.caption(f"Created {when(e['created_at'])}")
            smallest = min(assigned) if assigned else 0
            c3.progress(
                min(1.0, smallest / target) if target else 0.0,
                text=f"{smallest} / {target} users in the smallest variant",
            )
            c4.button("Open", key=f"open_{e['id']}", on_click=open_experiment, args=(e["id"],))


def detail_page() -> None:
    st.title("Experiment detail")
    experiments = api_get("/experiments")
    if not experiments:
        st.info("No experiments yet.")
        return
    by_id = {e["id"]: e for e in experiments}
    exp_id = st.selectbox(
        "Experiment",
        list(by_id),
        key="exp_id",
        format_func=lambda i: f"#{i} {by_id[i]['name']}",
    )
    experiment = by_id[exp_id]

    primary = METRIC_ALIASES.get(experiment["primary_metric"], experiment["primary_metric"])
    options = list(METRICS)
    metric = st.selectbox(
        "Metric",
        options,
        index=options.index(primary) if primary in options else 0,
        key=f"metric_{exp_id}",
        format_func=lambda m: METRICS[m],
    )
    params = (("metric", metric),)
    results = api_get(f"/experiments/{exp_id}/results", params)
    ops = api_get(f"/experiments/{exp_id}/metrics")
    series = api_get(f"/experiments/{exp_id}/series", params)

    st.markdown(
        f"{badge(experiment['status'])} &nbsp; Prompt {experiment['prompt_id']} &nbsp;"
        f"({'higher' if results['higher_is_better'] else 'lower'} is better, "
        f"decided by {results['decision_test'].replace('_', '-')} test)",
        unsafe_allow_html=True,
    )

    decision = results["decision"]
    if decision == "winner":
        winner = next(c for c in results["comparisons"] if c["label"] == results["winner"])
        banner(
            "winner",
            f"{results['winner']} wins on {METRICS[metric].lower()}",
            f"Adjusted p = {fmt(winner['p_adjusted'])}. Difference {fmt(winner['diff'], signed=True)} "
            f"(95% CI {fmt(winner['ci_low'])} to {fmt(winner['ci_high'])}). {results['note']}",
        )
    elif decision == "not_ready":
        need = max(v["samples_needed"] for v in results["variants"])
        banner("interim", f"Interim result: {need} more samples needed", results["note"])
    elif decision == "control_holds":
        banner("neutral", "The control holds", results["note"])
    else:
        banner("neutral", "No significant difference", results["note"])

    total_requests = sum(v["requests"] for v in ops["variants"])
    total_errors = sum(v["errors"] for v in ops["variants"])
    generator_cost = sum(v["total_cost_usd"] for v in ops["variants"])
    judge_cost = sum(v["judge_cost_usd"] for v in ops["variants"])
    cards = [
        ("Requests", f"{total_requests}", f"target {results['target_per_variant']} per variant"),
        ("Error rate", f"{(total_errors / total_requests * 100) if total_requests else 0:.1f}%", f"{total_errors} failed"),
        ("Generator cost", f"${generator_cost:.4f}", "all variants"),
        ("Judge cost", f"${judge_cost:.4f}", "scoring only"),
    ]
    for column, (label, value, sub) in zip(st.columns(4), cards):
        column.markdown(card(label, value, sub), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Mean by variant")
        st.caption("Bars show the mean, whiskers the 95% confidence interval.")
        st.plotly_chart(mean_bar_chart(results["variants"]))
    with right:
        st.subheader("Running mean")
        st.caption("Curves settle as observations accumulate.")
        if any(v["points"] for v in series["variants"]):
            st.plotly_chart(running_mean_chart(series))
        else:
            st.caption("No data yet.")

    st.subheader("Sample size progress")
    target = results["target_per_variant"]
    for v in results["variants"]:
        st.progress(min(1.0, v["n"] / target) if target else 0.0, text=f"{v['label']}: {v['n']} / {target}")

    st.subheader("Comparison with the control")
    st.dataframe(variants_table(results), hide_index=True)
    with st.expander("Operational metrics (latency, cost, JSON validity, judge)"):
        st.dataframe(operations_table(ops), hide_index=True)

    if decision == "winner":
        promote_section(experiment, results, metric)


def promote_section(experiment: dict, results: dict, metric: str) -> None:
    with st.expander("Promote the winner to production"):
        versions = api_get(f"/prompts/{experiment['prompt_id']}/versions")
        number_for_id = {v["id"]: v["version"] for v in versions}
        variant = next(v for v in experiment["variants"] if v["label"] == results["winner"])
        number = number_for_id[variant["version_id"]]
        st.write(
            f"This makes version {number} the active version of prompt {experiment['prompt_id']}, "
            "so every user gets it."
        )
        key = f"promote_{experiment['id']}"
        actor = st.text_input("Your name", key=f"{key}_actor")
        confirmed = st.checkbox("I understand this changes production traffic", key=f"{key}_confirm")
        if st.button("Promote", key=f"{key}_button", disabled=not (actor.strip() and confirmed)):
            _request(
                "POST",
                f"/prompts/{experiment['prompt_id']}/activate",
                json={
                    "version": number,
                    "actor": actor.strip(),
                    "reason": f"Won experiment #{experiment['id']} on {metric}",
                },
            )
            st.cache_data.clear()
            st.success(f"Version {number} is now active.")


def registry_page() -> None:
    st.title("Prompt registry")
    prompts = api_get("/prompts")
    if not prompts:
        st.info("No prompts yet. Create one with POST /prompts on the API docs page.")
        return
    by_id = {p["id"]: p for p in prompts}
    prompt_id = st.selectbox("Prompt", list(by_id), format_func=lambda i: f"#{i} {by_id[i]['name']}")
    prompt = by_id[prompt_id]
    versions = api_get(f"/prompts/{prompt_id}/versions")
    active = prompt["active_version"]
    st.markdown(
        f"{prompt['versions']} versions. Active: "
        + (f"version {active}" if active else "none (activate one below)"),
    )
    if not versions:
        st.info("This prompt has no versions yet.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Version": f"{v['version']} (active)" if v["version"] == active else str(v["version"]),
                    "Model": v["model"],
                    "Temperature": v["temperature"],
                    "Commit message": v["commit_message"],
                    "Created": when(v["created_at"]),
                }
                for v in versions
            ]
        ),
        hide_index=True,
    )

    numbers = [v["version"] for v in versions]
    text_for = {v["version"]: v["system_prompt"] for v in versions}
    st.subheader("Prompt text")
    shown = st.selectbox("Version", numbers, index=len(numbers) - 1, key="shown_version")
    st.code(text_for[shown], language="text", wrap_lines=True)

    st.subheader("Compare two versions")
    col_a, col_b = st.columns(2)
    older = col_a.selectbox("From", numbers, index=max(0, len(numbers) - 2), key="diff_from")
    newer = col_b.selectbox("To", numbers, index=len(numbers) - 1, key="diff_to")
    diff = "\n".join(
        difflib.unified_diff(
            text_for[older].splitlines(),
            text_for[newer].splitlines(),
            fromfile=f"version {older}",
            tofile=f"version {newer}",
            lineterm="",
        )
    )
    st.code(diff or "No differences.", language="diff", wrap_lines=True)

    st.subheader("Activate a version")
    st.caption("Rolling back is just activating an older version. Nothing is deleted.")
    target = st.selectbox("Version to activate", numbers, index=len(numbers) - 1, key="activate_version")
    actor = st.text_input("Your name", key="activate_actor")
    reason = st.text_input("Reason", key="activate_reason")
    if st.button("Activate", disabled=not (actor.strip() and reason.strip())):
        _request(
            "POST",
            f"/prompts/{prompt_id}/activate",
            json={"version": target, "actor": actor.strip(), "reason": reason.strip()},
        )
        st.cache_data.clear()
        st.success(f"Version {target} is now active. Refresh to see it.")


def audit_page() -> None:
    st.title("Audit log")
    rows = api_get("/audit-log")
    if not rows:
        st.info("No changes recorded yet.")
        return
    frame = pd.DataFrame(
        [
            {
                "When": when(r["created_at"]),
                "Who": r["actor"],
                "Action": r["action"],
                "Prompt": r["prompt_id"],
                "Details": ", ".join(f"{k}: {v}" for k, v in r["details"].items()),
            }
            for r in rows
        ]
    )
    chosen = st.multiselect("Action", sorted(frame["Action"].unique()))
    if chosen:
        frame = frame[frame["Action"].isin(chosen)]
    st.caption(f"{len(frame)} entries, newest first")
    st.dataframe(frame, hide_index=True)


PAGES = {
    "Experiments": experiments_page,
    "Experiment detail": detail_page,
    "Prompt registry": registry_page,
    "Audit log": audit_page,
}


def main() -> None:
    st.set_page_config(page_title="Prompt A/B", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("Prompt A/B")
    page = st.sidebar.radio("Page", list(PAGES), key="page", label_visibility="collapsed")
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(f"API: {API_URL}")
    try:
        PAGES[page]()
    except ApiError as exc:
        st.error(str(exc))


main()