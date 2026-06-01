"""
Tkinter launcher GUI for the Isaac Lab Leo Rover stack.

Runs on your LOCAL Windows desktop (Tk works natively — no X server / X11
forwarding needed) and, on Launch, **SSHes the training command to the pod**.
You pick a task, tweak any config, hit Launch, and a console pops up streaming
the pod's training output. Untick "Run on remote pod" to instead launch locally
(for a local Isaac install).

How config overrides reach the sim: the GUI serializes a dict to the
`EXPERIMENT_OVERRIDES` environment variable on the pod. `config.py` (carried
over verbatim from the PyBullet repo) reads that JSON at import and applies it to
its module globals BEFORE the Isaac env/agent configs are built — so any reward
weight, residual bound, ADR threshold, or terrain range you set here takes
effect for that run. The free-form "Extra overrides (JSON)" box lets you change
ANY config constant, not just the ones with form fields.

Remote delivery is robust to quoting: the whole remote command is base64-encoded
and decoded on the pod (`echo <b64> | base64 -d | bash -l`), so the JSON's quotes
never get mangled by Windows/SSH. The new console uses `cmd /k`, so it stays open
after training ends/crashes for you to read the output.
"""

import os
import sys
import json
import base64
import shlex
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Defaults come from config.py, but that imports torch (present on the pod, not
# necessarily on your Windows desktop). Fall back to hardcoded defaults so the
# GUI runs anywhere with just standard Python + tkinter. The POD's config.py is
# the real source of truth at run time; these only seed the form fields.
try:
    import config
except Exception:
    from types import SimpleNamespace
    config = SimpleNamespace(
        TRAINING_TERRAIN_MIN=10.0, ADR_TERRAIN_MAX_LIMIT=100.0,
        TRAINING_FRICTION_MIN=50.0, TRAINING_FRICTION_MAX=90.0,
        USE_CAMERA_LOOKAHEAD=True, MAX_RESIDUAL_VELOCITY=0.15, MAX_RESIDUAL_OMEGA=0.30,
        PPO_W_CTE=5.0, PPO_W_PROGRESS=10.0, PPO_W_EFFORT=0.5,
        PPO_SUCCESS_BONUS=200.0, PPO_FAILURE_PENALTY=50.0,
        ADR_TERRAIN_MAX_START=10.0, ADR_SUCCESS_THRESHOLD=0.70,
        ADR_CTE_THRESHOLD=0.10, ADR_STEP_UP=3.0,
    )


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
        self._on_run_mode_change()

    # ------------------------------------------------------------------ #
    def _spin(self, parent, r, c, lo, hi, default, flt=False, inc=0.01):
        var = (tk.DoubleVar if flt else tk.IntVar)(value=default)
        sb = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=11,
                         **({"increment": inc} if flt else {}))
        sb.grid(row=r, column=c, sticky="w", padx=4, pady=2)
        var.widget = sb
        return var

    def _entry(self, parent, r, c, default, width=22):
        var = tk.StringVar(value=default)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=r, column=c, sticky="w", padx=4, pady=2)
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

        # ── Run location: Remote pod (SSH) vs Local ──
        rf = ttk.LabelFrame(main, text="Run location", padding=8); rf.pack(fill="x", **pad)
        self.run_remote = tk.BooleanVar(value=True)
        ttk.Checkbutton(rf, text="Run on remote pod via SSH (uncheck = run locally)",
                        variable=self.run_remote, command=self._on_run_mode_change
                        ).grid(row=0, column=0, columnspan=4, sticky="w")

        # SSH target (prefilled with your current pod — update IP/port after each migration)
        self.ssh_frame = ttk.LabelFrame(main, text="SSH target (pod)", padding=8)
        self.ssh_frame.pack(fill="x", **pad)
        ttk.Label(self.ssh_frame, text="Host:").grid(row=0, column=0, sticky="e")
        self.ssh_host = self._entry(self.ssh_frame, 0, 1, "69.30.85.81")
        ttk.Label(self.ssh_frame, text="Port:").grid(row=0, column=2, sticky="e")
        self.ssh_port = self._entry(self.ssh_frame, 0, 3, "22027", width=10)
        ttk.Label(self.ssh_frame, text="User:").grid(row=1, column=0, sticky="e")
        self.ssh_user = self._entry(self.ssh_frame, 1, 1, "root")
        ttk.Label(self.ssh_frame, text="SSH key:").grid(row=1, column=2, sticky="e")
        self.ssh_key = self._entry(self.ssh_frame, 1, 3, "~/.ssh/id_ed25519", width=28)
        ttk.Label(self.ssh_frame, text="Remote project dir:").grid(row=2, column=0, sticky="e")
        self.remote_proj = self._entry(self.ssh_frame, 2, 1, "/workspace/leorover_isaac", width=34)
        ttk.Label(self.ssh_frame, text="Remote python:").grid(row=2, column=2, sticky="e")
        self.remote_py = self._entry(self.ssh_frame, 2, 3, "python", width=28)
        ttk.Label(self.ssh_frame, text="Pre-command (optional):").grid(row=3, column=0, sticky="e")
        self.remote_pre = self._entry(self.ssh_frame, 3, 1, "", width=60)
        ttk.Label(self.ssh_frame, text="e.g. conda activate ...", foreground="gray"
                  ).grid(row=3, column=3, sticky="w")

        # Isaac launch (timing + LOCAL launcher)
        lf = ttk.LabelFrame(main, text="Run settings", padding=8); lf.pack(fill="x", **pad)
        ttk.Label(lf, text="num_envs:").grid(row=0, column=0, sticky="e")
        self.num_envs = self._spin(lf, 0, 1, 1, 65536, 4096)
        ttk.Label(lf, text="max_iterations:").grid(row=0, column=2, sticky="e")
        self.max_iters = self._spin(lf, 0, 3, 1, 1000000, 30000)
        ttk.Label(lf, text="seed:").grid(row=1, column=0, sticky="e")
        self.seed = self._spin(lf, 1, 1, 0, 2**31 - 1, 42)
        self.headless = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="headless", variable=self.headless).grid(row=1, column=2, sticky="w")
        # Local-only launcher fields
        self.local_launcher = ttk.Label(lf, text="Local launcher:")
        self.local_launcher.grid(row=2, column=0, sticky="e")
        self.launcher = self._entry(lf, 2, 1, "isaaclab")
        self.use_dash_p = tk.BooleanVar(value=True)
        self.use_dash_p_cb = ttk.Checkbutton(lf, text="use `isaaclab -p` (local)", variable=self.use_dash_p)
        self.use_dash_p_cb.grid(row=2, column=2, sticky="w")

        # Checkpoint (compare)
        self.ckpt_frame = ttk.LabelFrame(main, text="Checkpoint (for Compare — pod path if remote)", padding=8)
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
        rwf = ttk.LabelFrame(main, text="Reward weights (config overrides)", padding=8); rwf.pack(fill="x", **pad)
        ttk.Label(rwf, text="W_CTE:").grid(row=0, column=0, sticky="e")
        self.w_cte = self._spin(rwf, 0, 1, 0, 100, config.PPO_W_CTE, flt=True, inc=0.5)
        ttk.Label(rwf, text="W_PROGRESS:").grid(row=0, column=2, sticky="e")
        self.w_prog = self._spin(rwf, 0, 3, 0, 100, config.PPO_W_PROGRESS, flt=True, inc=0.5)
        ttk.Label(rwf, text="W_EFFORT:").grid(row=1, column=0, sticky="e")
        self.w_eff = self._spin(rwf, 1, 1, 0, 10, config.PPO_W_EFFORT, flt=True)
        ttk.Label(rwf, text="SUCCESS_BONUS:").grid(row=1, column=2, sticky="e")
        self.succ = self._spin(rwf, 1, 3, 0, 1000, config.PPO_SUCCESS_BONUS, flt=True, inc=10)
        ttk.Label(rwf, text="FAILURE_PENALTY:").grid(row=2, column=0, sticky="e")
        self.fail = self._spin(rwf, 2, 1, 0, 1000, config.PPO_FAILURE_PENALTY, flt=True, inc=10)

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

        # Buttons
        bf = ttk.Frame(main); bf.pack(fill="x", pady=10)
        ttk.Button(bf, text="▶  Launch", command=self._launch).pack(side="right", padx=8)
        ttk.Button(bf, text="Show command", command=self._show_command).pack(side="right", padx=4)
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

    def _on_run_mode_change(self, *_):
        # SSH fields stay visible (harmless in local mode); just grey out the
        # local-only launcher toggle when running remotely.
        state = "disabled" if self.run_remote.get() else "normal"
        try:
            self.use_dash_p_cb.configure(state=state)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _collect_overrides(self):
        if self.terr_min.get() > self.terr_max.get():
            messagebox.showerror("Invalid", "Terrain min > max"); return None
        if self.fric_min.get() > self.fric_max.get():
            messagebox.showerror("Invalid", "Friction min > max"); return None
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
        try:
            extra = json.loads(self.extra.get("1.0", "end").strip() or "{}")
            overrides.update(extra)
        except json.JSONDecodeError as e:
            messagebox.showerror("Bad JSON", f"Extra overrides not valid JSON:\n{e}"); return None
        return overrides

    def _script_and_args(self, kind, task_id, proj):
        """Return (script_path, args_string) for the remote/local command.

        `proj` is the project root (remote path if running on the pod).
        """
        if kind == "compare":
            if not self.ckpt.get().strip():
                messagebox.showerror("Missing", "Compare needs a checkpoint path."); return None, None
            script = f"{proj}/scripts/compare_hybrid_vs_lqr.py"
            args = f"--checkpoint {shlex.quote(self.ckpt.get().strip())} --num_envs {self.num_envs.get()}"
        else:
            script = f"{proj}/scripts/train.py"
            args = (f"--task {task_id} --num_envs {self.num_envs.get()} "
                    f"--max_iterations {self.max_iters.get()} --seed {self.seed.get()}")
            if self.headless.get():
                args += " --headless"
        return script, args

    def _remote_inner(self, overrides, kind, task_id):
        """The bash command that runs ON THE POD."""
        proj = self.remote_proj.get().strip().rstrip("/")
        py = self.remote_py.get().strip() or "python"
        pre = self.remote_pre.get().strip()
        script, args = self._script_and_args(kind, task_id, proj)
        if script is None:
            return None
        j = json.dumps(overrides).replace("'", "'\\''")   # safe inside single quotes
        prefix = (pre + " && ") if pre else ""
        return (f"cd {proj} && {prefix}"
                f"EXPERIMENT_OVERRIDES='{j}' PYTHONPATH={proj} "
                f"{py} {script} {args}")

    def _remote_ssh_argv(self, inner):
        """Wrap the inner bash command for robust delivery over SSH (base64)."""
        b64 = base64.b64encode(inner.encode()).decode()
        key = os.path.expanduser(self.ssh_key.get().strip())
        host = self.ssh_host.get().strip()
        port = str(self.ssh_port.get()).strip()
        user = self.ssh_user.get().strip() or "root"
        remote = f"echo {b64} | base64 -d | bash -l"
        return ["ssh", "-tt", "-i", key, "-p", port, f"{user}@{host}", remote]

    # ------------------------------------------------------------------ #
    def _show_command(self):
        ov = self._collect_overrides()
        if ov is None:
            return
        kind, task_id = TASKS[self.task_var.get()]
        if self.run_remote.get():
            inner = self._remote_inner(ov, kind, task_id)
            if inner is None:
                return
            messagebox.showinfo(
                "Remote command (runs on the pod)",
                inner + "\n\n(Sent base64-encoded over SSH so quoting can't break.)")
        else:
            here = os.path.dirname(os.path.abspath(__file__))
            script, args = self._script_and_args(kind, task_id, here)
            if script is None:
                return
            launcher = self.launcher.get().strip() or "isaaclab"
            dash = "-p " if self.use_dash_p.get() else ""
            messagebox.showinfo("Local command", f"{launcher} {dash}{script} {args}")

    def _launch(self):
        ov = self._collect_overrides()
        if ov is None:
            return
        kind, task_id = TASKS[self.task_var.get()]

        if self.run_remote.get():
            # ---- remote: SSH the command to the pod ----
            inner = self._remote_inner(ov, kind, task_id)
            if inner is None:
                return
            ssh_argv = self._remote_ssh_argv(inner)
            self.status.set(f"SSH-launching {task_id} on {self.ssh_host.get()} …"); self.root.update()
            try:
                if sys.platform == "win32":
                    subprocess.Popen(["cmd", "/k"] + ssh_argv,
                                     creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen(ssh_argv)
                self.status.set(f"{task_id} launched on pod (new console). Close it to stop watching.")
            except FileNotFoundError:
                messagebox.showerror("Launch failed", "Could not run 'ssh'. Is OpenSSH installed/on PATH?")
                self.status.set("Launch failed")
            return

        # ---- local: run on this machine (needs a local Isaac install) ----
        here = os.path.dirname(os.path.abspath(__file__))
        script, args_str = self._script_and_args(kind, task_id, here)
        if script is None:
            return
        launcher = self.launcher.get().strip() or "isaaclab"
        argv = ([launcher, "-p", script] if self.use_dash_p.get() else [launcher, script]) + shlex.split(args_str)
        env = os.environ.copy()
        env["EXPERIMENT_OVERRIDES"] = json.dumps(ov)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.status.set(f"Launching {task_id} locally …"); self.root.update()
        try:
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/k"] + argv, env=env, cwd=here,
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(argv, env=env, cwd=here)
            self.status.set(f"{task_id} launched locally (overrides applied)")
        except FileNotFoundError:
            messagebox.showerror("Launch failed",
                                 f"Could not run '{launcher}'. Set the Local launcher field to your "
                                 f"Isaac Lab launcher (e.g. full path to isaaclab / isaaclab.sh).")
            self.status.set("Launch failed")


def launch_gui():
    root = tk.Tk()
    IsaacExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
