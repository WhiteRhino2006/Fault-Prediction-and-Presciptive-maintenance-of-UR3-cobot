"""Live user interface for the closed loop.  Run:  streamlit run dashboard.py"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import runlock
from src.data import N_JOINTS, ROOT

LIVE = ROOT / "outputs" / "live"
FIG = ROOT / "outputs" / "figures"
COLORS = {"CONTINUE": "#2ca02c", "MONITOR": "#b8a03a", "DERATE_JOINT": "#e6b400", "SLOW_DOWN": "#ff7f0e", "SAFE_STOP": "#d62728",
          "COOL_DOWN": "#1f77b4"}

st.set_page_config(page_title="Cobot Prescriptive Maintenance", layout="wide")

# ---- Demo control panel: starts/stops loop.py so the presenter never needs the terminal ----
with st.sidebar:
    BIG = st.toggle("Projector mode (large text)", value=True)
    st.header("Demo control")
    use_sim = st.toggle("Drive UR3 in CoppeliaSim", value=True, help="Off = run the pipeline without the simulator")
    mode = st.radio("Scenario", ["Real faults (from test data)", "Inject a fault into a joint"])
    joint = st.selectbox("Joint to inject", range(N_JOINTS), format_func=lambda j: f"J{j}",
                         disabled=mode.startswith("Real"))
    secs = st.slider("Duration (s)", 30, 300, 150, step=10)
    ctl = st.radio("Speed controller", ["PI (reacts)", "MPC (plans 20 s ahead)"],
                   help="PI slows down when risk/heat rises and pauses to cool. MPC predicts heating and slows "
                        "down early so the joint never reaches the limit (no cool-down stops needed).")

    @st.fragment(run_every=1)  # re-checks every second, so the buttons unlock as soon as a run ends
    def run_controls(use_sim, mode, joint, secs, ctl):
        proc = st.session_state.get("proc")
        # a run counts as active if this page started it OR any loop.py is registered in the PID file
        # (e.g. started before the page was refreshed) - so Start can never launch a second run
        running = (proc is not None and proc.poll() is None) or runlock.running_pid() is not None
        if st.button("▶ Start demo", type="primary", disabled=running, width="stretch"):
            runlock.stop_running()  # belt and braces: never two runs driving the robot
            cmd = [sys.executable, "-u", str(ROOT / "loop.py"), "--steps", str(secs),
                   "--controller", "mpc" if ctl.startswith("MPC") else "pi"]
            if not use_sim:
                cmd.append("--nosim")
            if mode.startswith("Inject"):
                cmd += ["--inject", str(joint)]
            (LIVE / "state.json").unlink(missing_ok=True)
            st.session_state.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(LIVE / "loop.log", "w"),
                                                     stderr=subprocess.STDOUT)
            st.rerun()
        if st.button("■ Stop", disabled=not running, width="stretch"):
            # polite stop: the demo stops the arm + simulation itself (also works after a page refresh)
            runlock.stop_running()
            if proc is not None and proc.poll() is None:  # fallback if it never registered / didn't respond
                proc.terminate()
            st.rerun()
        if running:
            st.caption("Status: 🟢 running")
        elif proc is not None and proc.returncode in (0, 1):
            st.success("✅ Run finished - press ▶ Start demo for a new run")
        else:
            st.caption("Status: ⚪ idle")
        if proc is not None and not running and proc.returncode not in (0, None, 1) and (LIVE / "loop.log").exists():
            st.error((LIVE / "loop.log").read_text()[-600:])

    run_controls(use_sim, mode, joint, secs, ctl)

FS = 20 if BIG else 13  # chart font size
LW = 4 if BIG else 2    # chart line width
if BIG:
    st.markdown("""<style>
    html { font-size: 21px; }
    h1 { font-size: 2.4rem !important; }
    [data-testid="stMetricValue"] { font-size: 2.3rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.15rem !important; }
    .stTabs [data-baseweb="tab"] p { font-size: 1.35rem !important; font-weight: 600; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { font-size: 1.1rem !important; }
    [data-testid="stDataFrame"] { font-size: 1.1rem; }
    </style>""", unsafe_allow_html=True)

st.title("Causal + Game-Theoretic Prescriptive Maintenance - UR3 Cobot")


def styled(fig, height):
    """Apply projector-friendly sizes to a chart."""
    k = 1.3 if BIG else 1.0
    fig.update_layout(height=int(height * k), font=dict(size=FS), title_font_size=FS + 4,
                      margin=dict(l=10, r=10, t=50 if BIG else 30, b=10), legend=dict(font=dict(size=FS)))
    return fig


def causal_figure(edges, root, comp):
    ang = np.linspace(0, 2 * np.pi, N_JOINTS, endpoint=False) + np.pi / 2
    xy = np.c_[np.cos(ang), np.sin(ang)]
    fig = go.Figure()
    for i, j, w in edges:
        fig.add_annotation(x=xy[j, 0], y=xy[j, 1], ax=xy[i, 0], ay=xy[i, 1], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2 + 4 * min(w, 1), arrowcolor="#aaa",
                           standoff=32 if BIG else 22, startstandoff=32 if BIG else 22)
    col = ["#d62728" if j == root else "#ff7f0e" if j in comp else "#1f77b4" for j in range(N_JOINTS)]
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=[f"J{j}" for j in range(N_JOINTS)],
                             marker=dict(size=64 if BIG else 44, color=col),
                             textfont=dict(color="white", size=24 if BIG else 14), hoverinfo="text"))
    fig.update_layout(showlegend=False, xaxis=dict(visible=False, range=[-1.4, 1.4]),
                      yaxis=dict(visible=False, range=[-1.4, 1.4]),
                      title="2. Causal graph")
    return styled(fig, 420)


@st.fragment(run_every=0.5)  # refresh twice a second so the screen keeps pace with the arm
def live():
    try:
        s = json.loads((LIVE / "state.json").read_text())
        h = pd.read_csv(LIVE / "history.csv")
    except Exception:
        proc = st.session_state.get("proc")
        if (proc is not None and proc.poll() is None) or runlock.running_pid() is not None:
            st.info("⏳ Starting… loading the model and connecting to CoppeliaSim (about 5–10 s).")
        else:
            st.info("Choose a scenario in the left panel and press ▶ Start demo.")
        return

    c = COLORS[s["action"]]
    st.markdown(
        f"<div style='background:{c};color:white;padding:16px 22px;border-radius:10px;margin-bottom:6px;font-size:{32 if BIG else 22}px'>"
        f"<b>{s['action'].replace('_', ' ')}</b> &nbsp;|&nbsp; fault risk {s['p_fault']:.0%} &nbsp;|&nbsp; "
        f"arm pace {s['pace']:.0%} &nbsp;|&nbsp; {s['advice']}</div>", unsafe_allow_html=True)
    st.caption("🟢 CONTINUE – normal  ·  🟡 MONITOR – risk elevated, locating the cause  ·  🟡 DERATE – a joint is protected, speed limited  ·  "
               "🟠 SLOW DOWN – high fault risk  ·  🔵 COOL DOWN – paused so a hot joint can cool  ·  "
               "🔴 SAFE STOP – very high risk, arm stopped")
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    m = st.columns(5)
    m[0].metric(f"Time (of {s.get('steps', s['t'] + 1)} s)", f"{s['t'] + 1}")  # t counts from 0
    reg = s.get("latched") or []
    m[1].metric("Faulty", ", ".join(f"J{j}" for j in reg) if reg else "none",
                help="Joints CONFIRMED as root cause (counterfactual check 3x in a row, or had to be cooled). "
                     "Remembered until maintenance - this is what the banner acts on.")
    m[2].metric("Suspect", "-" if s["root_joint"] is None else f"J{s['root_joint']}",
                help="The causal layer's live best guess from the last 60 s of sensor data. "
                     "It can move around until it is confirmed.")
    inj = s.get("injected")
    m[3].metric("Injected" if inj is not None else "Truth",
                f"J{inj}" if inj is not None else ("FAULT" if s["true_fault"] else "OK"),
                help="Hidden from the pipeline - shown only for comparison")
    m[4].metric("Robot", "SIM" if s["sim_connected"] else "OFF",
                help=f"UR3 in CoppeliaSim connected? Pipeline latency {s['latency_ms']:.0f} ms per step")

    # Self-healing control status (what the robot is fixing by itself right now)
    ctrl = s.get("controller")
    if ctrl:
        cool = s.get("cooling")
        hottest = int(np.argmax(s["temp_rise"]))
        mpc = s.get("mpc")
        if mpc:
            q = mpc["heat_power"]
            jq = int(np.argmax(q))
            ctl_txt = (f"MPC (plans 20 s ahead): estimated heating power J{jq} {q[jq] * 60:.1f}°C/min at full speed → "
                       f"pace **{s['pace']:.0%}**")
        else:
            ctl_txt = (f"PI speed controller: risk {s['p_fault']:.0%} vs target {ctrl['target']:.0%} → "
                       f"error {ctrl['error']:+.2f}, integral {ctrl['integral']:.2f} → pace **{s['pace']:.0%}**")
        st.markdown(
            f"**🛠 Self-healing control** (pipeline latency {s['latency_ms']:.0f} ms) &nbsp;·&nbsp; {ctl_txt}"
            f" &nbsp;·&nbsp; Thermal guard: "
            + (f"**J{cool} cooling down (paused)**" if cool is not None else
               f"OK (hottest J{hottest} +{s['temp_rise'][hottest]:.1f}°C, pause at +2.0°C)")
            + (f" &nbsp;·&nbsp; Kalman heating rate J{hottest}: {s['heat_rate'][hottest] * 60:+.2f}°C/min"
               if s.get("heat_rate") else "")
            + (" &nbsp;·&nbsp; 🔒 **Fault memory: " + ", ".join(f"J{j}" for j in s["latched"])
               + " registered faulty** (speed limited until maintenance)" if s.get("latched") else ""))

    cf = s.get("counterfactual")
    if cf and s["root_joint"] is not None and cf["root"] != s["root_joint"]:
        # the suspect changed since the last check (runs every 5 s): don't show a result about the old suspect
        st.markdown(f"**🔁 Counterfactual check of the current suspect** &nbsp;·&nbsp; checking new suspect "
                    f"**J{s['root_joint']}**… (result within 5 s)")
        cf = None
    if cf:
        parts = ", ".join(f"J{d['victim']} −{max(d['drop'], 0):.0%}" for d in cf["per_victim"])
        st.markdown(
            f"**🔁 Counterfactual check of the current suspect** (every 5 s) &nbsp;·&nbsp; *If J{cf['root']} had been healthy, the victims' "
            f"abnormal behaviour would drop by:* {parts} → "
            + ("**root cause confirmed ✅** (explains the others better than any other joint)" if cf["confirmed"]
               else f"**not confirmed ⚠️** (J{cf.get('rival', '?')} explains the others as well or better)"
               if cf.get("scores") and cf["scores"][cf.get("rival", 0)] >= cf["scores"][cf["root"]]
               else "**not confirmed ⚠️** (influence too weak)"))

    # Charts stacked one below the other, each full width (less crowded on a projector)
    st.divider()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.t, y=h.p_fault, name="predicted risk", line=dict(color="#ff4b4b", width=LW)))
    fig.add_trace(go.Scatter(x=h.t, y=h.true_fault, name="actual fault", fill="tozeroy",
                             line=dict(color="rgba(0,0,0,0)"), fillcolor="rgba(160,160,160,0.4)"))
    fig.add_trace(go.Scatter(x=h.t, y=h.pace, name="arm pace", line=dict(color="#4da3ff", dash="dot", width=LW)))
    if s.get("controller"):
        fig.add_hline(y=s["controller"]["target"], line=dict(color="#ff9f9f", dash="dash", width=1),
                      annotation_text="risk target (PI controller)", annotation_position="top left")
    fig.update_layout(title="1. Prediction vs reality", yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h"))
    st.plotly_chart(styled(fig, 360), width="stretch")

    st.divider()
    _, mid, _ = st.columns([1, 2, 1])  # keep the joint circle round and centred
    with mid:
        st.plotly_chart(causal_figure(s["edges"], s["root_joint"], s["compensators"]), width="stretch")
        st.markdown("🔴 **current suspect (root cause)** &nbsp;&nbsp; 🟠 **compensating** &nbsp;&nbsp; 🔵 not involved &nbsp;&nbsp; "
                    "arrow A → B = *A influences B*")

    st.divider()
    sig = st.radio("Joint signal", ["current", "temp", "speed"], horizontal=True, key="sig")
    tail = h.tail(120)
    fig = go.Figure([go.Scatter(x=tail.t, y=tail[f"{sig}_J{j}"], name=f"J{j}", line=dict(width=LW - 1))
                     for j in range(N_JOINTS)])
    fig.update_layout(title=f"3. Sensor layer: {sig} per joint")
    st.plotly_chart(styled(fig, 340), width="stretch")

    st.divider()
    j = [f"J{k}" for k in range(N_JOINTS)]
    fig = go.Figure([go.Bar(x=j, y=s["effort_nash"], name="Nash (selfish joints)"),
                     go.Bar(x=j, y=s["effort_social"], name="Cooperative optimum")])
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=-0.15),
                      title=f"4. Game layer: effort per joint (PoA = {s['poa']:.2f})")
    st.plotly_chart(styled(fig, 340), width="stretch")
    st.divider()

    st.subheader("5. Prescriptive action log")
    lg = pd.DataFrame(s["log"][::-1])
    st.dataframe(lg[[c for c in ("t", "event", "advice") if c in lg.columns]], width="stretch", hide_index=True)


@st.dialog("📊 Model validation (offline results)", width="large")
def offline_results():
    """Evidence for reviewers - opened from the sidebar, hidden from the operator screen."""
    try:
        r = json.loads((ROOT / "outputs" / "results.json").read_text())
        df = pd.DataFrame(r["results"])
        # Deep sequence baselines (LSTM/GRU) are research comparisons: kept in outputs/results.csv and the
        # paper, not shown in the operator interface
        df = df[~df["model"].isin(["LSTM", "GRU", "SVM (RBF)", "1D-CNN", "Transformer"])]
        st.subheader("Fault prediction (held-out test set, 10 s horizon)")
        st.dataframe(df[["features", "model", "precision", "recall", "f1", "roc_auc", "pr_auc"]].round(3),
                     hide_index=True, width="stretch")
        st.subheader("Root-cause identification (injected faults)")
        st.json(r["root_cause_experiment"])
    except Exception:
        st.info("Run `run_all_training.bat` first.")
    for f in sorted(FIG.glob("*.png")):  # one figure per row
        st.divider()
        st.image(str(f), caption=f.stem.replace("_", " "), width="stretch")


with st.sidebar:
    st.divider()
    if st.button("📊 Model validation", width="stretch", help="Offline test results and paper figures"):
        offline_results()

live()
