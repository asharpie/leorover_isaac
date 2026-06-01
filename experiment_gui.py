"""
Tkinter launcher GUI for the Isaac Lab Leo Rover stack.

Isaac-Lab analogue of the PyBullet `experiment_gui.py`. Pick a task, set the
common knobs, override any `config.py` constant, and launch `scripts/train.py`
(or the comparison script) in a new console — without editing files between runs.

How overrides reach the sim: the GUI serializes a dict to the
`EXPERIMENT_OVERRIDES` environment variable. `config.py` (carried over verbatim
from the PyBullet repo) reads that JSON at import and applies it to its module
globals BEFORE the Isaac env/agent configs are built — so any reward weight,
residual bound, ADR threshold, or terrain range you set here takes effect for
that run. The free-form "Extra overrides (JSON)" box lets you change ANY config
constant, not just the ones with form fields.

Launch command (default): `<isaac launcher> -p scripts/train.py --task ... --num_envs ...`
Set the launcher to your install's `isaaclab` / `isaaclab.sh` (uses `-p` to run
inside Isaac's python). Untick "use isaaclab -p" to call a plain python instead.
"""

import os
import sys
import json
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import config  # carried-over single source of truth for defaults


TASKS = {
    "Train PPO (Mars, pure)":      ("train", "Isaac-LeoRover-Mars-v0"),
    "Train Hybrid (LQR+residual)": ("train", "Isaac-LeoRover-Mars-Hybrid-v0"),
    "Flat smoke test":             ("train", "Isaac-LeoRover-Flat-v0"),
    "Compare: Hybrid vs LQR":      ("compare", "Isaac-LeoRover-Mars-Hybrid-v0"),
}


class IsaacExperimentGUI:
    def __init__(self, root):
        self.root = root
        root.title("Leo Rover — Isaac Lab Experiment Launcher")
        root.resizable(True, True)
        self._build()
        self._on_task_change()

    # ------------------------------------------------------------------ #
    def _spin(self, parent, r, c, lo, hi, default, flt=False, inc=0.01):
        var = (tk.DoubleVar if flt else tk.IntVar)(value=default)
        sb = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=11,
                         **({"increment": inc} if flt else {}))
        sb.grid(row=r, column=c, sticky="w", padx=4, pady=2)
        var.widget = sb
        return var

    def _build(self):
        pad = {"padx": 8, "pady": 3}
        main = ttk.Frame(self.root, padding=10); main.pack(fill="both", expand=True)

        # Task
        tf = ttk.LabelFrame(main, text="Task", padding=8); tf.pack(fill="x", **pad)
        self.task_var = tk.StringVar(value="Train PPO (Mars, pure)")
        for i, name in enumerate(TASKS):
            ttk.Radiobutton(tf, text=name, variable=self.task_var, value=name,
                            command=self._on_task_change
                            ).grid(row=i // 2, column=i % 2, sticky="w", padx=12, pady=2)

        # Isaac launch
        lf = ttk.LabelFrame(main, text="Isaac launch", padding=8); lf.pack(fill="x", **pad)
        ttk.Label(lf, text="Launcher:").grid(row=0, column=0, sticky="e")
        self.launcher = tk.StringVar(value="isaaclab")
        ttk.Entry(lf, textvariable=self.launcher, width=28).grid(row=0, column=1, sticky="w", padx=4)
        self.use_dash_p = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="use `isaaclab -p`", variable=self.use_dash_p).grid(row=0, column=2, sticky="w")
        self.headless = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="headless", variable=self.headless).grid(row=0, column=3, sticky="w")
        ttk.Label(lf, text="num_envs:").grid(row=1, column=0, sticky="e")
        self.num_envs = self._spin(lf, 1, 1, 1, 65536, 4096)
        ttk.Label(lf, text="max_iterations:").grid(row=1, column=2, sticky="e")
        self.max_iters = self._spin(lf, 1, 3, 1, 1000000, 30000)
        ttk.Label(lf, text="seed:").grid(row=2, column=0, sticky="e")
        self.seed = self._spin(lf, 2, 1, 0, 2**31 - 1, 42)

        # Checkpoint (compare)
        self.ckpt_frame = ttk.LabelFrame(main, text="Checkpoint (for Compare)", padding=8)
        self.ckpt_frame.pack(fill="x", **pad)
        self.ckpt = tk.StringVar(value="")
        ttk.Entry(self.ckpt_frame, textvariable=self.ckpt, width=60).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(self.ckpt_frame, text="Browse…", command=self._browse).grid(row=0, column=1, padx=4)
        self.ckpt_frame.columnconfigure(0, weight=1)

        # Environment overrides
        ef = ttk.LabelFrame(main, text="Environment (config overrides)", padding=8); ef.pack(fill="x", **pad)
        ttk.Label(ef, text="Terrain min %:").grid(row=0, column=0, sticky="e")
        self.terr_min = self._spin(ef, 0, 1, 0, 100, int(config.TRAINING_TERRAIN_MIN))
        ttk.Label(ef, text="Terrain max %:").grid(row=0, column=2, sticky="e")
        self.terr_max = self._spin(ef, 0, 3, 0, 100, int(config.ADR_TERRAIN_MAX_LIMIT))
        ttk.Label(ef, text="Friction min %:").grid(row=1, column=0, sticky="e")
        self.fric_min = self._spin(ef, 1, 1, 0, 100, int(config.TRAINING_FRICTION_MIN))
        ttk.Label(ef, text="Friction max %:").grid(row=1, column=2, sticky="e")
        self.fric_max = self._spin(ef, 1, 3, 0, 100, int(config.TRAINING_FRICTION_MAX))
        self.camera = tk.BooleanVar(value=bool(config.USE_CAMERA_LOOKAHEAD))
        ttk.Checkbutton(ef, text="camera lookahead", variable=self.camera).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(ef, text="Residual v (hybrid):").grid(row=3, column=0, sticky="e")
        self.res_v = self._spin(ef, 3, 1, 0, 2, config.MAX_RESIDUAL_VELOCITY, flt=True)
        ttk.Label(ef, text="Residual ω (hybrid):").grid(row=3, column=2, sticky="e")
        self.res_w = self._spin(ef, 3, 3, 0, 5, config.MAX_RESIDUAL_OMEGA, flt=True)

        # Reward overrides
        rf = ttk.LabelFrame(main, text="Reward weights (config overrides)", padding=8); rf.pack(fill="x", **pad)
        ttk.Label(rf, text="W_CTE:").grid(row=0, column=0, sticky="e")
        self.w_cte = self._spin(rf, 0, 1, 0, 100, config.PPO_W_CTE, flt=True, inc=0.5)
        ttk.Label(rf, text="W_PROGRESS:").grid(row=0, column=2, sticky="e")
        self.w_prog = self._spin(rf, 0, 3, 0, 100, config.PPO_W_PROGRESS, flt=True, inc=0.5)
        ttk.Label(rf, text="W_EFFORT:").grid(row=1, column=0, sticky="e")
        self.w_eff = self._spin(rf, 1, 1, 0, 10, config.PPO_W_EFFORT, flt=True)
        ttk.Label(rf, text="SUCCESS_BONUS:").grid(row=1, column=2, sticky="e")
        self.succ = self._spin(rf, 1, 3, 0, 1000, config.PPO_SUCCESS_BONUS, flt=True, inc=10)
        ttk.Label(rf, text="FAILURE_PENALTY:").grid(row=2, column=0, sticky="e")
        self.fail = self._spin(rf, 2, 1, 0, 1000, config.PPO_FAILURE_PENALTY, flt=True, inc=10)

        # ADR overrides
        af = ttk.LabelFrame(main, text="ADR curriculum (config overrides)", padding=8); af.pack(fill="x", **pad)
        ttk.Label(af, text="terrain_max_start %:").grid(row=0, column=0, sticky="e")
        self.adr_start = self._spin(af, 0, 1, 0, 100, config.ADR_TERRAIN_MAX_START, flt=True)
        ttk.Label(af, text="success_thresh:").grid(row=0, column=2, sticky="e")
        self.adr_succ = self._spin(af, 0, 3, 0, 1, config.ADR_SUCCESS_THRESHOLD, flt=True)
        ttk.Label(af, text="cte_thresh (m):").grid(row=1, column=0, sticky="e")
        self.adr_cte = self._spin(af, 1, 1, 0, 1, config.ADR_CTE_THRESHOLD, flt=True)
        ttk.Label(af, text="step_up %:").grid(row=1, column=2, sticky="e")
        self.adr_step = self._spin(af, 1, 3, 0, 50, config.ADR_STEP_UP, flt=True)

        # Free-form overrides
        xf = ttk.LabelFrame(main, text="Extra overrides (JSON — any config constant)", padding=8)
        xf.pack(fill="both", **pad)
        self.extra = tk.Text(xf, height=4, width=70)
        self.extra.insert("1.0", "{}")
        self.extra.pack(fill="both", expand=True)

        # Launch
        bf = ttk.Frame(main); bf.pack(fill="x", pady=10)
        ttk.Button(bf, text="▶  Launch", command=self._launch).pack(side="right", padx=8)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(bf, textvariable=self.status, foreground="gray").pack(side="left", padx=8)

    # ------------------------------------------------------------------ #
    def _browse(self):
        p = filedialog.askopenfilename(title="Select checkpoint (.pt)",
                                       filetypes=[("Checkpoint", "*.pt"), ("All", "*.*")])
        if p:
            self.ckpt.set(p)

    def _on_task_change(self, *_):
        kind, _id = TASKS[self.task_var.get()]
        if kind == "compare":
            self.ckpt_frame.pack(fill="x", padx=8, pady=3)
        else:
            self.ckpt_frame.pack_forget()

    # ------------------------------------------------------------------ #
    def _launch(self):
        kind, task_id = TASKS[self.task_var.get()]
        if self.terr_min.get() > self.terr_max.get():
            messagebox.showerror("Invalid", "Terrain min > max"); return
        if self.fric_min.get() > self.fric_max.get():
            messagebox.showerror("Invalid", "Friction min > max"); return

        overrides = {
            "TRAINING_TERRAIN_MIN": self.terr_min.get(),
            "TRAINING_TERRAIN_MAX": self.terr_max.get(),
            "ADR_TERRAIN_MAX_LIMIT": float(self.terr_max.get()),
            "TRAINING_FRICTION_MIN": self.fric_min.get(),
            "TRAINING_FRICTION_MAX": self.fric_max.get(),
            "USE_CAMERA_LOOKAHEAD": bool(self.camera.get()),
            "MAX_RESIDUAL_VELOCITY": self.res_v.get(),
            "MAX_RESIDUAL_OMEGA": self.res_w.get(),
            "PPO_W_CTE": self.w_cte.get(),
            "PPO_W_PROGRESS": self.w_prog.get(),
            "PPO_W_EFFORT": self.w_eff.get(),
            "PPO_SUCCESS_BONUS": self.succ.get(),
            "PPO_FAILURE_PENALTY": self.fail.get(),
            "ADR_TERRAIN_MAX_START": self.adr_start.get(),
            "ADR_SUCCESS_THRESHOLD": self.adr_succ.get(),
            "ADR_CTE_THRESHOLD": self.adr_cte.get(),
            "ADR_STEP_UP": self.adr_step.get(),
        }
        # merge free-form JSON
        try:
            extra = json.loads(self.extra.get("1.0", "end").strip() or "{}")
            overrides.update(extra)
        except json.JSONDecodeError as e:
            messagebox.showerror("Bad JSON", f"Extra overrides not valid JSON:\n{e}"); return

        here = os.path.dirname(os.path.abspath(__file__))
        if kind == "compare":
            if not self.ckpt.get().strip():
                messagebox.showerror("Missing", "Compare needs a checkpoint."); return
            script = os.path.join(here, "scripts", "compare_hybrid_vs_lqr.py")
            args = ["--checkpoint", self.ckpt.get().strip(), "--num_envs", str(self.num_envs.get())]
        else:
            script = os.path.join(here, "scripts", "train.py")
            args = ["--task", task_id, "--num_envs", str(self.num_envs.get()),
                    "--max_iterations", str(self.max_iters.get()), "--seed", str(self.seed.get())]
            if self.headless.get():
                args.append("--headless")

        # build command (isaaclab -p <script>  OR  <launcher> <script>)
        launcher = self.launcher.get().strip() or "isaaclab"
        if self.use_dash_p.get():
            cmd = [launcher, "-p", script] + args
        else:
            cmd = [launcher, script] + args

        env = os.environ.copy()
        env["EXPERIMENT_OVERRIDES"] = json.dumps(overrides)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        self.status.set(f"Launching {task_id} …"); self.root.update()
        try:
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/k"] + cmd, env=env, cwd=here,
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd, env=env, cwd=here)
            self.status.set(f"{task_id} launched (overrides applied)")
        except FileNotFoundError:
            messagebox.showerror("Launch failed",
                                 f"Could not run '{launcher}'. Set the Launcher field to your "
                                 f"Isaac Lab launcher (e.g. the full path to isaaclab / isaaclab.sh).")
            self.status.set("Launch failed")


def launch_gui():
    root = tk.Tk()
    IsaacExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
