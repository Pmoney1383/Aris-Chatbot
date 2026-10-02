"""
Phase 1 -> Phase 1.5 data pipeline, run unattended in its own console window.

Waits for Phase 1 training to finish, then runs everything Phase 1.5 needs as
early and as parallel as the machine allows. It never starts Phase 1.5 training.

  T0  (Phase 1 exits at its last step), at once and in parallel:
      - collector:  gather_images.py --phase15 --no-megalith --expand
      - captions:   caption_images.py uncaptioned   (images already on disk; doesn't wait for the collector)
      - flips:      preprocess.py flips             (mirrored latents for Phase 1's existing shards)
      - the prompt tests in PROMPT_TESTS, if present (quick look at the final Phase 1 model)
  collection ends when the collector finishes or at the collection deadline
  (COLLECT_UNTIL, local time, at least COLLECT_MIN_HOURS after T0), whichever is first,
  via the graceful STOP_COLLECTION file. Then:
      - interim:    export-csv + latents + flips for the newly collected (already captioned) images
      - captions 2: uncaptioned again (the new Phase 1.5 images) + export-food-batch
  gate: waits for dataset/vehicle_probe/probe.pt and the P15_CODE_READY marker (written after the
        multi-length caption code is swapped into training/ and tested), at most GATE_HOURS after
        the collection deadline; past that it continues without whichever is missing.
  final: vehicle_probe features+apply (if probe.pt), export-csv, preprocess text, latents, flips, verify.
  -> writes 'ready' to dataset/phase15_pipeline_state.txt.

Every step is resumable, so this script can be restarted at any time: completed steps are
recorded in dataset/phase15_pipeline.json. Long jobs run in their own minimized consoles
with logs in logs/. Create dataset/PIPELINE_NO_RESTART to stop it from restarting a
Phase 1 run that exited early (e.g. you stopped it on purpose).

Usage (from image-generator-v2/scripts):
    ../.venv/Scripts/python.exe phase15_pipeline.py
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import config

R = config.ROOT
P = R / ".venv" / "Scripts" / "python.exe"
LOGS = R / "logs"
STATE_TXT = R / "dataset" / "phase15_pipeline_state.txt"
STATE_JSON = R / "dataset" / "phase15_pipeline.json"
CODE_READY = R / "dataset" / "P15_CODE_READY"
NO_RESTART = R / "dataset" / "PIPELINE_NO_RESTART"
PROBE = R / "dataset" / "vehicle_probe" / "probe.pt"
PHASE1_LOG = LOGS / "dit_phase1.csv"
PHASE1_STEPS = 40_000
COLLECT_UNTIL = (10, 30)       # local time the collector is asked to stop (the morning after T0)
COLLECT_MIN_HOURS = 4.0
CAPTION_UNTIL = None           # "HH:MM" stops captioning early (rest waits for a training pause); None = caption all
GATE_HOURS = 3.0
PROMPT_TESTS = Path(r"C:\Users\amirp\AppData\Local\Temp\claude\C--Users-amirp-Documents-Git-Projects-Aris-Chatbot"
                    r"\cc245f93-3336-4f28-83b4-c74c7c2e194b\scratchpad")
PROMPT_TEST_FILES = ["prompt_test2.py", "prompt_test3.py", "prompt_test4.py", "prompt_test_cars.py"]
GEN = "set PREP_GENERAL_ONLY=1&& set PYTHONUNBUFFERED=1&& "   # no space before && so the value has none


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOGS / "phase15_pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE_JSON.read_text()) if STATE_JSON.exists() else {"done": []}


def save_state(s: dict) -> None:
    STATE_JSON.write_text(json.dumps(s, indent=1))


def set_stage(name: str) -> None:
    STATE_TXT.write_text(name)
    log(f"stage -> {name}")


def python_cmdlines() -> list[str]:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object CommandLine"],
                         capture_output=True, text=True, timeout=120).stdout
    return [l.strip() for l in out.splitlines() if l.strip()]


def running(fragment: str) -> bool:
    return any(fragment in c for c in python_cmdlines())


def stop_processes(fragment: str) -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object CommandLine -like "
                    f"'*{fragment}*' | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"], timeout=120)
    while running(fragment):
        time.sleep(5)


def launch(title: str, workdir: Path, command: str, logname: str | None, visible: bool = False) -> subprocess.Popen:
    """Run `command` (cmd.exe syntax) in its own console. Output goes to logs/<logname> when given."""
    inner = f"title {title}&& " + (f'({command}) >> "{LOGS / logname}" 2>&1' if logname else command)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 1 if visible else 7   # SW_SHOWNORMAL / SW_SHOWMINNOACTIVE
    log(f"launch [{title}] {command}")
    return subprocess.Popen(f'cmd.exe /s {"/k" if visible else "/c"} "{inner}"', cwd=workdir,
                            creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=si)


def last_step(path: Path) -> int:
    if not path.exists():
        return 0
    step = 0
    for line in path.read_text().splitlines()[1:]:
        try:
            step = max(step, int(line.split(",", 1)[0]))
        except ValueError:
            pass
    return step


def tail(path: Path, n: int = 4000) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - n))
        return f.read().decode("utf-8", "replace")


class Job:
    """A named, resumable step: launched once, retried (it resumes) if it exits non-zero.
    If a process matching `match` is already running (this script was restarted mid-step),
    the job waits for it to exit and then runs once more, which resumes and confirms it."""

    def __init__(self, state, name, title, workdir, command, logname, retries=2, match=None):
        self.state, self.name, self.title, self.workdir = state, name, title, workdir
        self.command, self.logname, self.retries, self.match = command, logname, retries, match
        self.proc: subprocess.Popen | None = None
        self.waiting = False

    @property
    def done(self) -> bool:
        return self.name in self.state["done"]

    def start(self):
        if self.done or self.proc is not None or self.waiting:
            return
        if self.match and running(self.match):
            log(f"{self.name}: a '{self.match}' process is already running; will rerun it once that exits")
            self.waiting = True
            return
        self.proc = launch(self.title, self.workdir, self.command, self.logname)

    def poll(self) -> bool:
        """True once the job has finished successfully (or was already done)."""
        if self.done:
            return True
        if self.waiting:
            if not running(self.match):
                self.waiting = False
                self.proc = launch(self.title, self.workdir, self.command, self.logname)
            return False
        if self.proc is None:
            return False
        rc = self.proc.poll()
        if rc is None:
            return False
        if rc == 0:
            log(f"{self.name}: finished")
            self.state["done"].append(self.name)
            save_state(self.state)
            self.proc = None
            return True
        if self.retries > 0:
            self.retries -= 1
            log(f"{self.name}: exited with {rc}; restarting (it resumes), {self.retries} retries left")
            self.proc = launch(self.title, self.workdir, self.command, self.logname)
        else:
            log(f"{self.name}: FAILED with {rc}; see logs/{self.logname}. Continuing without it.")
            self.state["done"].append(self.name)
            self.state.setdefault("failed", []).append(self.name)
            save_state(self.state)
            self.proc = None
            return True
        return False


def wait_for_phase1():
    absent = 0
    while True:
        step = last_step(PHASE1_LOG)
        alive = running("train.py --phase 1")
        if not alive and step >= PHASE1_STEPS:
            log(f"Phase 1 finished at step {step}.")
            return
        if alive:
            absent = 0
        else:
            absent += 1
            if absent >= 3 and not NO_RESTART.exists():
                log(f"Phase 1 is not running (last step {step} < {PHASE1_STEPS}); restarting it.")
                launch("DiT phase 1 training", R / "training", f'"{P}" train.py --phase 1', None, visible=True)
                absent = -10   # give it time to appear
        time.sleep(60)


def collect_deadline(t0: float) -> float:
    start = dt.datetime.fromtimestamp(t0)
    d = start.replace(hour=COLLECT_UNTIL[0], minute=COLLECT_UNTIL[1], second=0, microsecond=0)
    if d <= start:
        d += dt.timedelta(days=1)
    return max(d.timestamp(), t0 + COLLECT_MIN_HOURS * 3600)


def main():
    LOGS.mkdir(exist_ok=True)
    s = load_state()
    log(f"pipeline started (done so far: {', '.join(s['done']) or 'nothing'})")
    S, T = R / "scripts", R / "training"

    if "t0" not in s:
        set_stage("train")
        wait_for_phase1()
        s["t0"] = time.time()
        save_state(s)
    t0 = s["t0"]
    deadline = collect_deadline(t0)
    log(f"collection deadline {dt.datetime.fromtimestamp(deadline):%Y-%m-%d %H:%M}")

    collect = Job(s, "collect", "Phase 1.5 collection", S,
                  f'set PYTHONUNBUFFERED=1&& "{P}" gather_images.py --phase15 --no-megalith --expand',
                  "collect_expand.log", retries=5, match="gather_images.py")
    caption1 = Job(s, "caption1", "Florence captions (existing images)", S,
                   f'set PYTHONUNBUFFERED=1&& "{P}" caption_images.py uncaptioned', "caption.log",
                   match="caption_images.py")
    flips1 = Job(s, "flips_existing", "Flips (Phase 1 shards)", T, GEN + f'"{P}" preprocess.py flips --res 256',
                 "flips_existing.log", match="preprocess.py")
    tests = [Job(s, f"test_{f}", "Prompt test", T, f'"{P}" "{PROMPT_TESTS / f}"', "prompt_tests_final.log", retries=0)
             for f in PROMPT_TEST_FILES if (PROMPT_TESTS / f).exists()]
    interim = Job(s, "interim", "Phase 1.5 interim latents", T,
                  GEN + f'"{P}" ..\\scripts\\caption_images.py export-csv && "{P}" preprocess.py latents --res 256 '
                        f'&& "{P}" preprocess.py flips --res 256', "prep_interim.log", match="preprocess.py")
    caption2 = Job(s, "caption2", "Florence captions (new images)", S,
                   f'set PYTHONUNBUFFERED=1&& "{P}" caption_images.py uncaptioned'
                   + (f' --until {CAPTION_UNTIL}' if CAPTION_UNTIL else '') + ' && '
                   f'"{P}" caption_images.py export-food-batch', "caption2.log", match="caption_images.py")

    if "collect" not in s["done"]:
        config.STOP_COLLECTION_FILE.unlink(missing_ok=True)
        set_stage("collect")
    for j in (collect, caption1, flips1, *tests):
        j.start()

    # -- collection, with a deadline ------------------------------------------------
    stop_sent = False
    while not collect.poll():
        if not stop_sent and time.time() >= deadline:
            log("collection deadline reached: asking the collector to stop (it saves everything first)")
            config.STOP_COLLECTION_FILE.touch()
            stop_sent = True
            collect.retries = 0
        caption1.poll(), flips1.poll(), [t.poll() for t in tests]
        time.sleep(30)
    config.STOP_COLLECTION_FILE.unlink(missing_ok=True)

    # -- second caption pass (new Phase 1.5 images first) + interim latents --------------
    set_stage("caption")
    if not caption1.done:
        # Restart captioning so the newly collected labeled images go first (the first
        # pass's list was fixed before they existed). Mark it done before stopping it so
        # it isn't retried; caption2 resumes exactly where it left off.
        s["done"].append("caption1")
        save_state(s)
        caption1.proc = None
        stop_processes("caption_images.py uncaptioned")
        log("caption1: stopped at collection end; caption2 continues (Phase 1.5 images first)")
    caption2.start()
    while not flips1.poll():        # never two flips runs at once
        caption2.poll()
        time.sleep(30)
    interim.start()
    while not (caption2.poll() & interim.poll()):
        time.sleep(30)

    # -- gate: vehicle probe + multi-length caption code ----------------------------
    gate_until = deadline + GATE_HOURS * 3600
    while not (PROBE.exists() and CODE_READY.exists()) and time.time() < gate_until:
        if "gate_wait_logged" not in s:
            log(f"waiting for {'' if PROBE.exists() else 'probe.pt '}{'' if CODE_READY.exists() else 'P15_CODE_READY'} "
                f"(until {dt.datetime.fromtimestamp(gate_until):%H:%M})")
            s["gate_wait_logged"] = True
        time.sleep(60)
    probe, code = PROBE.exists(), CODE_READY.exists()
    log(f"gate passed: vehicle probe {'yes' if probe else 'NO (nothing dropped)'}, "
        f"multi-length code {'yes' if code else 'NO (single caption store)'}")

    # -- final preprocessing ------------------------------------------------------------
    set_stage("preprocess")
    steps = []
    if probe:
        steps += [f'cd /d "{S}"', f'"{P}" vehicle_probe.py features', f'"{P}" vehicle_probe.py apply']
    steps += [f'cd /d "{S}"', f'"{P}" caption_images.py export-csv', f'cd /d "{T}"',
              f'"{P}" preprocess.py text' + (" --variant all" if code else ""),
              f'"{P}" preprocess.py latents --res 256', f'"{P}" preprocess.py flips --res 256',
              f'"{P}" preprocess.py verify --res 256']
    final = Job(s, "final", "Phase 1.5 final preprocessing", T, GEN + " && ".join(steps), "preprocess_p15.log",
                match="preprocess.py")
    final.start()
    while not final.poll():
        time.sleep(30)

    failed = s.get("failed", [])
    set_stage("ready" if "final" not in failed else "preprocess-failed")
    log("Phase 1.5 data is ready: start it with  train.py --phase 1.5" if "final" not in failed
        else "final preprocessing failed; see logs/preprocess_p15.log")
    if failed:
        log(f"steps that failed: {', '.join(failed)}")
    log(tail(LOGS / "preprocess_p15.log", 1500))


if __name__ == "__main__":
    main()
