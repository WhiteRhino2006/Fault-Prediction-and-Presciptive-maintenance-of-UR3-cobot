"""Closed loop: sensors -> pipeline -> action on the UR3 in CoppeliaSim -> dashboard.

Digital-twin setup: the UR3 in CoppeliaSim executes a pick-and-place cycle, while its
joint sensor readings (current, temperature, speed) are replayed from the real UR3
CobotOps recordings, including real fault episodes. Every second the pipeline reads the
sensors, decides an action, and the action changes how the simulated arm moves.

Run (CoppeliaSim open with the UR3 scene):  python loop.py
Run without CoppeliaSim (dashboard only):    python loop.py --nosim
"""
import argparse
import json
import math
import threading
import time

import numpy as np
import pandas as pd

from src.data import N_JOINTS, ROOT, SENSOR_COLS, load_clean
from src.control import ArmPlant, Supervisor
from src import runlock
from src.pipeline import Pipeline

LIVE = ROOT / "outputs" / "live"
LIVE.mkdir(parents=True, exist_ok=True)
STATE_FILE, HIST_FILE = LIVE / "state.json", LIVE / "history.csv"

# Pick-and-place cycle for the UR3 (joint angles in rad)
WAYPOINTS = np.array([
    [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],      # home
    [0.9, -1.10, 1.30, -1.80, -1.57, 0.0],   # above pick
    [0.9, -0.85, 1.45, -2.20, -1.57, 0.0],   # pick
    [-0.9, -1.10, 1.30, -1.80, -1.57, 1.2],  # place
])
CYCLE_S = 24.0  # seconds per full cycle at nominal speed (slow enough to stay within the UR3's joint speed limits,
                # so that slowing down to 50% is clearly visible)


def trajectory(phase):
    """Smooth closed path through the waypoints (periodic Catmull-Rom spline); phase in [0, 1).
    Unlike segment-by-segment easing, the arm flows through each waypoint without stopping."""
    k = len(WAYPOINTS)
    x = (phase % 1.0) * k
    i = int(x)
    u = x - i
    p0, p1, p2, p3 = (WAYPOINTS[(i + d) % k] for d in (-1, 0, 1, 2))
    return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


# Colours used to show on the robot itself what the system believes
SUSPECT_RGB, FAULTY_RGB = [1.0, 0.55, 0.0], [0.9, 0.05, 0.05]


class Robot:
    def __init__(self, enabled):
        self.enabled = enabled
        if not enabled:
            return
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        print("Connecting to CoppeliaSim (make sure it is open) ...")
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        sim = self.sim
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        base = sim.getObject("/UR3")
        self._remove_demo_script(base)
        self.joints = [h for h in sim.getObjectsInTree(base, sim.object_joint_type)][:N_JOINTS]
        self.link_shapes = [self._link_shapes(h) for h in self.joints]
        self.markers, self.colored = {}, {}
        print(f"Found UR3 with {len(self.joints)} joints")
        # Stepped mode: the simulation advances exactly one step per motion command -> no jitter
        self.stepped = False
        for target in (sim, self.client):
            try:
                target.setStepping(True)
                self._stepper, self.stepped = target, True
                break
            except Exception:
                continue
        sim.startSimulation()

    def _link_shapes(self, joint):
        """Shapes moved directly by this joint (up to the next joint) - used to colour the joint's link."""
        sim, out = self.sim, []
        try:
            for h in sim.getObjectsInTree(joint, sim.object_shape_type):
                p = sim.getObjectParent(h)
                while p != -1 and p != joint and sim.getObjectType(p) != sim.object_joint_type:
                    p = sim.getObjectParent(p)
                if p == joint:
                    out.append(h)
        except Exception:
            pass
        return out

    def color_joint(self, j, rgb):
        """Show a coloured marker ball on joint j (rgb=None removes it). The robot's own shapes are never
        recoloured - each UR3 link has several coloured parts that a single recolour would destroy."""
        key = tuple(rgb) if rgb else None
        if not self.enabled or self.colored.get(j) == key:
            return
        sim = self.sim
        try:
            old = self.markers.pop(j, None)
            if old is not None:
                sim.removeObjects([old])
            if rgb:
                m = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [0.07, 0.07, 0.07], 0)
                sim.setObjectPosition(m, sim.getObjectPosition(self.joints[j], -1), -1)
                sim.setObjectParent(m, self.joints[j], True)  # moves with the joint
                sim.setObjectInt32Param(m, sim.shapeintparam_static, 1)
                sim.setObjectInt32Param(m, sim.shapeintparam_respondable, 0)
                sim.setShapeColor(m, None, sim.colorcomponent_ambient_diffuse, list(rgb))
                self.markers[j] = m
        except Exception as e:
            print("marker error:", e)
        self.colored[j] = key

    def step(self):
        if self.enabled and self.stepped:
            self._stepper.step()

    def _remove_demo_script(self, base):
        """The stock UR3 model ships with a demo script that moves the arm; remove it so we are in control."""
        sim = self.sim
        try:  # CoppeliaSim >= 4.6: scripts are scene objects
            for h in sim.getObjectsInTree(base):
                if sim.getObjectType(h) == sim.sceneobject_script:
                    sim.removeObjects([h])
                    print("Removed UR3 demo script")
            return
        except Exception:
            pass
        try:  # older versions
            sh = sim.getScript(sim.scripttype_childscript, base)
            if sh != -1:
                sim.removeScript(sh)
                print("Removed UR3 demo script")
        except Exception as e:
            print("Could not remove demo script automatically:", e)

    # Real UR convention -> CoppeliaSim model (whose zero pose is the arm pointing straight up)
    SIM_OFFSET = np.array([0.0, math.pi / 2, 0.0, math.pi / 2, 0.0, 0.0])

    def command(self, q):
        if not self.enabled:
            return
        for h, qi in zip(self.joints, np.asarray(q) + self.SIM_OFFSET):
            self.sim.setJointTargetPosition(h, float(qi))

    def read(self):
        if not self.enabled:
            return None
        return [self.sim.getJointPosition(h) - o for h, o in zip(self.joints, self.SIM_OFFSET)]

    def stop(self):
        if self.enabled:
            for j in list(self.colored):  # leave the scene as we found it
                self.color_joint(j, None)
            if self.stepped:
                try:
                    self._stepper.setStepping(False)
                except Exception:
                    pass
            self.sim.stopSimulation()


class MotionThread(threading.Thread):
    """Moves the arm on its own steady 20 Hz clock, independent of how long the pipeline takes to think.
    The main loop only updates `pace` once per second; this thread turns it into smooth joint motion
    (previously the motion commands were squeezed into whatever time was left after the pipeline,
    which made the arm stutter when the pipeline was slow). It is the only thread talking to the robot."""
    HZ = 20
    ACCEL = 1.0         # max change of pace per second (smooth speed-ups and slow-downs, no jerks)
    TREMOR_RAD = 0.05   # friction tremor amplitude of a fully developed fault at full speed
    SUSPECT_HOLD = 3    # seconds a suspect must stay the same before it gets an orange marker

    def __init__(self, robot, rate=1.0):
        super().__init__(daemon=True)
        self.robot, self.rate = robot, rate
        self.pace, self.phase, self.q = 1.0, 0.0, None
        self.fault_joint, self.fault_level = None, 0.0  # physical fault (from the injection)
        self.suspect, self.faulty = None, []              # what the SYSTEM believes (colours)
        self._cur_pace, self._tremor = 1.0, 0.0
        self._last_suspect, self._suspect_for = None, 0
        self._rng = np.random.default_rng()
        self._halt = threading.Event()

    def run(self):
        dt, nxt, k = 1.0 / self.HZ, time.time(), 0
        while not self._halt.is_set():
            # ease towards the commanded pace instead of jumping
            self._cur_pace += float(np.clip(self.pace - self._cur_pace, -self.ACCEL * dt, self.ACCEL * dt))
            self.phase += self._cur_pace * (dt * self.rate) / CYCLE_S
            q = trajectory(self.phase)
            if self.fault_joint is not None:
                # stick-slip friction of a worn joint: a jittery tremor that grows with the fault and with speed
                self._tremor = 0.7 * self._tremor + 0.3 * self._rng.normal()
                q[self.fault_joint] += self.TREMOR_RAD * self.fault_level * self._cur_pace * self._tremor
            try:
                self.robot.command(q)
                self.robot.step()
                if k % self.HZ == 0:  # once per second: read joint angles, update joint markers
                    self.q = self.robot.read()
                    # orange only once the same suspect has held for SUSPECT_HOLD s (no flicker)
                    self._suspect_for = self._suspect_for + 1 if self.suspect == self._last_suspect else 0
                    self._last_suspect = self.suspect
                    steady = self.suspect if self._suspect_for >= self.SUSPECT_HOLD else None
                    for j in range(N_JOINTS):
                        rgb = FAULTY_RGB if j in self.faulty else SUSPECT_RGB if j == steady else None
                        self.robot.color_joint(j, rgb)
            except Exception:
                if self._halt.is_set():
                    break
            k += 1
            nxt += dt
            time.sleep(max(0.0, nxt - time.time()))

    def stop(self):
        self._halt.set()
        self.join(timeout=2.0)


def inject_fault(reading, k, age, baseline, rng):
    """Live fault injection on joint k: friction bursts + drift in current and heating.
    The controller then propagates the disturbance: the next joint reacts 1 s later and the
    previous joint 2 s later, so THREE joints look abnormal while only k is faulty.
    The pipeline is not told k."""
    c = f"current_J{k}"
    sd = baseline["std"][c]
    burst = rng.normal(0, 1.5 * sd) + min(age, 60) / 30 * sd
    hist = inject_fault.hist
    for m, lag, gain in [((k + 1) % N_JOINTS, 1, 0.8), ((k - 1) % N_JOINTS, 2, 0.6)]:
        past = hist[-lag] if len(hist) >= lag else 0.0
        reading[f"current_J{m}"] += gain * past * baseline["std"][f"current_J{m}"] / sd
    reading[c] += burst
    hist.append(burst)  # the fault's friction heat is produced by the ArmPlant (it depends on speed)


inject_fault.hist = []


def pick_segment(df, start):
    """Default start: shortly before a fault-rich stretch in the held-out test part."""
    if start is not None:
        return start
    n = len(df)
    test = df.iloc[int(n * 0.70):]
    onsets = test.index[(test["fault"] == 1) & (test["fault"].shift(1) == 0)]
    return max(int(onsets[0]) - 90, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosim", action="store_true", help="run without CoppeliaSim")
    ap.add_argument("--start", type=int, default=None, help="dataset row to start replaying from")
    ap.add_argument("--steps", type=int, default=60, help="seconds of data to replay")
    ap.add_argument("--rate", type=float, default=1.0, help="replay speed-up (2 = twice real time)")
    ap.add_argument("--inject", type=int, default=None, help="joint index (0-5) to inject a degradation fault into")
    ap.add_argument("--inject-at", type=int, default=10, help="seconds into the run when the injected fault starts")
    ap.add_argument("--controller", choices=["pi", "mpc"], default="pi", help="pace controller: PI (reactive) or MPC (plans ahead)")
    args = ap.parse_args()
    rng = np.random.default_rng()

    df = load_clean()
    start = pick_segment(df, args.start)
    pipe = Pipeline()
    runlock.claim()  # only one demo may drive the robot: stop any other running loop first
    robot = Robot(not args.nosim)
    HIST_FILE.unlink(missing_ok=True)

    motion = MotionThread(robot, args.rate)
    motion.start()
    log = []
    plant, sup = ArmPlant(), Supervisor("mpc_full" if args.controller == "mpc" else "full")
    logged_reg, last_q = [], None
    pace = 1.0  # pace applied during the previous second (the plant responds to it)
    print(f"Replaying rows {start}..{start + args.steps}. Open the dashboard: streamlit run dashboard.py")
    try:
        for t in range(start, min(start + args.steps, len(df))):
            row = df.iloc[t]
            reading = row[SENSOR_COLS].to_dict()
            tic = time.time()
            injecting = args.inject is not None and t - start >= args.inject_at
            if injecting:
                inject_fault(reading, args.inject, t - start - args.inject_at, pipe.baseline, rng)
            # Digital-twin plant: the arm's own speed changes its current and temperature
            reading = plant.step(reading, pace, args.inject if injecting else None)
            # Only sensor readings go into the pipeline; the ground-truth label is never an input
            out = pipe.step(reading)
            latency_ms = (time.time() - tic) * 1000  # sensing -> decision time of the pipeline

            # ---- Self-healing control: PI speed controller + thermal guard + fault memory ----
            pace, rx, info = sup.step(out, reading, pd.DataFrame(pipe.buffer), pipe.baseline)
            cf = info["counterfactual"]
            q = last_q  # joint angles measured at the end of the previous second

            cs, gm = out["causal"], out["game"]
            # Log only meaningful events: the operator banner changes, or a joint gets registered as faulty
            new_reg = [j for j in info["latched"] if j not in logged_reg]
            if new_reg or (log and log[-1]["action"] != rx["action"]) or (not log and rx["level"] > 0):
                event = ("REGISTERED FAULTY: " + ", ".join(f"J{j}" for j in new_reg)) if new_reg else rx["action"]
                log.append({"t": int(t - start), "event": event, "action": rx["action"], "advice": rx["advice"]})
                logged_reg += new_reg
            state = {
                "t": int(t - start), "steps": int(min(args.steps, len(df) - start)), "row": int(t), "timestamp": str(row["timestamp"]),
                "p_fault": out["p_fault"], "action": rx["action"], "level": rx["level"],
                "speed_scale": rx["speed_scale"], "pace": pace, "advice": rx["advice"],
                "root_joint": rx["root_joint"], "compensators": rx["compensators"],
                "first_signal": cs["first_signal"],
                "edges": cs["edges"], "root_score": list(map(float, cs["root_score"])),
                "effort_nash": list(map(float, gm["effort_nash"])), "effort_social": list(map(float, gm["effort_social"])),
                "costs": list(map(float, gm["costs"])), "poa": gm["price_of_anarchy"],
                "true_fault": int(row["fault"]), "true_fault_type": int(row["fault_type"]),
                "injected": args.inject if (args.inject is not None and t - start >= args.inject_at) else None,
                "sim_connected": robot.enabled, "joint_pos": q, "latency_ms": max(latency_ms, 0.0),
                "log": log[-12:],
                **info,
            }
            STATE_FILE.write_text(json.dumps(state, default=float))
            h = {"t": state["t"], "p_fault": out["p_fault"], "level": rx["level"], "pace": pace,
                 "true_fault": state["true_fault"], **{k: float(v) for k, v in reading.items()}}
            pd.DataFrame([h]).to_csv(HIST_FILE, mode="a", header=not HIST_FILE.exists(), index=False)
            print(f"t={state['t']:4d}  p={out['p_fault']:.2f}  {rx['action']:12s} pace={pace:.2f} "
                  f"root={rx['root_joint']}  truth={'FAULT' if state['true_fault'] else 'ok'}"
                  + (f"  injected=J{state['injected']}" if state["injected"] is not None else "")
                  + (f"  cf={'OK' if cf['confirmed'] else 'no'}:{cf['best']['drop']:.2f}" if cf else "")
                  + (f"  latched={info['latched']}" if info["latched"] else ""))

            # ---- Act on the robot AFTER publishing the decision, so screen and arm change together ----
            # The motion thread moves the arm smoothly at 20 Hz; here we only hand it the new pace.
            motion.pace = pace
            motion.fault_joint = args.inject if injecting else None
            motion.fault_level = min(1.0, (t - start - args.inject_at) / 30.0) if injecting else 0.0
            motion.suspect, motion.faulty = rx["root_joint"], list(info["latched"])
            last_q = motion.q
            if runlock.stop_requested():  # Stop pressed on the dashboard: finish cleanly
                print("Stop requested - stopping the arm.")
                break
            # Keep real time: each second of data takes 1/rate seconds INCLUDING the pipeline
            time.sleep(max(0.0, tic + 1.0 / args.rate - time.time()))
    except KeyboardInterrupt:
        pass
    finally:
        # each clean-up step runs even if an earlier one fails: the arm/simulation must always be stopped
        for cleanup in (motion.stop, robot.stop, runlock.release):
            try:
                cleanup()
            except Exception as e:
                print("cleanup error:", e)


if __name__ == "__main__":
    main()
