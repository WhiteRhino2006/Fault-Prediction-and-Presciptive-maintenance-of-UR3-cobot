"""Experiment: closed-loop ablation of the self-healing control (offline, headless; no simulator needed).

Same injected faults, five control strategies (all implemented by src.control.Supervisor):
  No action           the arm always runs at full pace (prediction/diagnosis only, nobody acts)
  Rule-based          fixed risk thresholds -> fixed speed levels (the first version of the framework)
  PI only             PI risk controller
  PI + thermal guard  PI + Kalman-filtered thermal duty-cycle guard
  Full (ours)         PI + thermal guard + fault memory (a confirmed faulty joint stays latched)
  Full + Markov game  as Full, but effort sharing comes from the Markov (dynamic) game
  MPC + fault memory  model predictive pace control (plans 20 s ahead) + thermal guard + fault memory
Each run replays real UR3 data through the digital-twin plant; the faulty joint's friction heat
depends on the pace actually applied, so the strategies really change the outcome.

Metrics (from fault injection to end of run):
  peak_rise_C     highest temperature rise of the faulty joint caused by the fault
  overheat_s      seconds the faulty joint spent above +2.0 degC (the safe limit)
  work_cycles     pick-and-place cycles completed (throughput)
  mean_pace       average commanded speed fraction
  false_all_clear_s  seconds the operator was told "CONTINUE - no maintenance needed" after the faulty joint
                  had already been diagnosed (what fault memory is designed to prevent)

Run:  python exp_closed_loop.py
"""
import numpy as np
import pandas as pd

import loop as L
from src.control import ArmPlant, Supervisor
from src.data import N_JOINTS, ROOT, SENSOR_COLS, load_clean
from src.pipeline import Pipeline

STEPS, INJECT_AT = 120, 10
STARTS_FROM_TEST = [0.05, 0.25, 0.45, 0.65, 0.85]  # 5 positions x 6 joints = 30 runs per strategy (held-out test part)
STRATEGIES = {"No action": "none", "Rule-based": "rule", "PI only": "pi",
              "PI + thermal guard": "pi_thermal", "Full (ours)": "full",
              "Full + Markov game": "full_markov", "MPC + fault memory": "mpc_full"}
LIMIT = 2.0
WORK_CYCLE_S = 12.0  # reference cycle time for the work metric (fixed, independent of the demo's visual speed)


def run(df, start, k, strategy, seed):
    rng = np.random.default_rng(seed)
    L.inject_fault.hist = []
    pipe, plant, sup = Pipeline(), ArmPlant(), Supervisor(STRATEGIES[strategy])
    pace, phase = 1.0, 0.0
    peak, over, paces, diagnosed, all_clear = 0.0, 0, [], False, 0
    for t in range(start, start + STEPS):
        reading = df.iloc[t][SENSOR_COLS].to_dict()
        injecting = t - start >= INJECT_AT
        if injecting:
            L.inject_fault(reading, k, t - start - INJECT_AT, pipe.baseline, rng)
        reading = plant.step(reading, pace, k if injecting else None)
        out = pipe.step(reading)
        pace, rx, _ = sup.step(out, reading, pd.DataFrame(pipe.buffer), pipe.baseline)
        phase += pace / WORK_CYCLE_S
        if injecting:
            peak = max(peak, plant.dT[k])
            over += int(plant.dT[k] > LIMIT)
            paces.append(pace)
            diagnosed = diagnosed or rx["root_joint"] == k
            # operator told "CONTINUE - no maintenance needed" although the faulty joint was already diagnosed
            all_clear += int(diagnosed and rx["action"] == "CONTINUE")
    return {"peak_rise_C": peak, "overheat_s": over, "work_cycles": phase, "mean_pace": float(np.mean(paces)),
            "false_all_clear_s": all_clear}


def main():
    import sys
    only = [a for a in sys.argv[1:]]  # optional: re-run only these strategy names, keep the others from the CSV
    df = load_clean()
    n = len(df)
    rows = []
    for si, frac in enumerate(STARTS_FROM_TEST):
        start = int(n * 0.70) + int((n * 0.30 - STEPS) * frac)
        for k in range(N_JOINTS):
            for strat in (only or STRATEGIES):
                r = run(df, start, k, strat, seed=100 * si + k)  # same fault noise for every strategy
                r.update(strategy=strat, joint=k, start=start)
                rows.append(r)
                print(f"start={start} J{k} {strat:12s} peak={r['peak_rise_C']:.2f}C over={r['overheat_s']:3d}s "
                      f"work={r['work_cycles']:.2f} pace={r['mean_pace']:.2f}", flush=True)
    res = pd.DataFrame(rows)
    csv = ROOT / "outputs" / "exp_closed_loop.csv"
    if only and csv.exists():
        old = pd.read_csv(csv)
        res = pd.concat([old[~old.strategy.isin(only)], res], ignore_index=True)
    res.to_csv(csv, index=False)
    summ = res.groupby("strategy")[["peak_rise_C", "overheat_s", "false_all_clear_s", "work_cycles", "mean_pace"]].agg(["mean", "std"])
    print("\n", summ.reindex(list(STRATEGIES)).round(2).to_string())
    plot(res)


def plot(res):
    """Draw the ablation figure (also callable on the saved CSV: python -c "import exp_closed_loop as e, pandas as pd;
    e.plot(pd.read_csv('outputs/exp_closed_loop.csv'))")."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for a, (m, lab) in zip(ax, [("peak_rise_C", "Peak temperature rise of faulty joint (°C)"),
                                ("overheat_s", "Time above +2 °C limit (s)"),
                                ("false_all_clear_s", "False 'no maintenance needed' (s)"),
                                ("work_cycles", "Work completed (cycles)")]):
        g = res.groupby("strategy")[m]
        a.bar(list(STRATEGIES), g.mean().reindex(list(STRATEGIES)), yerr=g.std().reindex(list(STRATEGIES)), capsize=4,
              color=["#999", "#e6b400", "#4da3ff", "#9467bd", "#2ca02c", "#17becf", "#8c564b"])
        a.set_title(lab, fontsize=10)
        a.tick_params(axis="x", rotation=20)
    ax[0].axhline(LIMIT, color="r", ls="--", lw=0.8)
    fig.suptitle(f"Closed-loop ablation: {len(res) // len(STRATEGIES)} injected-fault runs per strategy (mean ± sd)")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "figures" / "closed_loop_ablation.png", dpi=150)


if __name__ == "__main__":
    main()
