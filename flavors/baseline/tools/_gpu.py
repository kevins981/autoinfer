"""Shared GPU helpers for the baseline flavor's tools (experiment, gpujob): no queue (one agent),
but the same safety and accounting as v1: refuse when someone else is on the GPUs or the run's
limit (config.md) is reached, kill everything the job started, and log GPU-minutes to $AUTOINFER_RUN/gpu/ledger.jsonl (read by resources.py)."""
import json, os, re, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

EXIT_FOREIGN, EXIT_BUDGET, EXIT_TIMEOUT = 75, 76, 124   # 76 = the run's limit is reached
FOREIGN_MEM_MIB = 1024


def die(tool, msg, code=1):
    print(f"{tool}: {msg}", file=sys.stderr)
    sys.exit(code)


def run_dir(tool):
    r = os.environ.get("AUTOINFER_RUN")
    if not r:
        die(tool, "AUTOINFER_RUN is not set (the run folder)")
    return Path(r).resolve()


def now_iso(t=None):
    return datetime.fromtimestamp(t or time.time(), timezone.utc).isoformat(timespec="seconds")


def foreign_usage():
    """Anything on the GPUs while we're not running = someone else. Returns a description or None."""
    try:
        mem = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"nvidia-smi failed: {e}"
    procs = [l.strip() for l in apps.strip().splitlines() if l.strip()]
    if procs:
        return "; ".join(f"pid/MiB {p}" for p in procs)
    busy = []
    for l in mem.strip().splitlines():
        i, m = [int(x) for x in l.split(",")]
        if m > FOREIGN_MEM_MIB:
            busy.append(f"GPU {i}: {m} MiB used")
    return "; ".join(busy) or None


def limits(run):
    """The run's stopping rules from config.md: 'max experiments: N', 'max hours: H' (wall clock) and
    'max gpu hours: G' (GPU time of all jobs, experiments and gpujobs); any may be 'none'.
    Returns a dict with the limits and the amounts used so far."""
    cfg = (run / "config.md").read_text()
    def val(name):
        m = re.search(rf"^\s*-?\s*{name}:\s*([0-9.]+)", cfg, re.M | re.I)
        return float(m.group(1)) if m else None
    done, gpu_min = 0, 0.0
    led = run / "gpu" / "ledger.jsonl"
    if led.exists():
        for l in led.read_text().splitlines():
            if l.strip():
                e = json.loads(l)
                gpu_min += e.get("gpu_minutes", 0)
                done += e.get("label", "").startswith("experiment:")
    started = json.loads((run / "run.json").read_text()).get("started")
    hours = (time.time() - datetime.fromisoformat(started).timestamp()) / 3600 if started else 0.0
    return {"max_experiments": val("max experiments"), "max_hours": val("max hours"),
            "max_gpu_hours": val("max gpu hours"), "experiments": done, "hours": hours,
            "gpu_hours": gpu_min / 60}


def limits_line(run):
    L = limits(run)
    of = lambda k: f" of max {L[k]:g}" if L[k] is not None else ""
    return (f"{L['experiments']} experiments done{of('max_experiments')}; "
            f"{L['gpu_hours']:.2f} GPU-hours used{of('max_gpu_hours')}; "
            f"{L['hours']:.2f} h elapsed{of('max_hours')}")


def check_start(tool, run, experiment=False):
    L = limits(run)
    over = ((L["max_hours"] is not None and L["hours"] >= L["max_hours"]) or
            (L["max_gpu_hours"] is not None and L["gpu_hours"] >= L["max_gpu_hours"]) or
            (experiment and L["max_experiments"] is not None and L["experiments"] >= L["max_experiments"]))
    if over:
        die(tool, f"the run's limit is reached ({limits_line(run)}). Stop experimenting: write "
                  f"$AUTOINFER_RUN/final.md (program.md, 'When the run ends') and stop.", EXIT_BUDGET)
    foreign = foreign_usage()
    if foreign:
        die(tool, f"the GPUs are in use by someone else ({foreign}). Nothing was run. Wait and retry "
                  f"later (e.g. in 10 min); never compete for the GPUs.", EXIT_FOREIGN)


def kill_group(proc, grace=10):
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    for _ in range(grace):
        if proc.poll() is not None:
            break
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_job(tool, run, cmd, log_path, timeout_min, label):
    """Run cmd in its own process group with a timeout; log to the ledger. Returns the exit code."""
    g = run / "gpu"
    g.mkdir(exist_ok=True)
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        def stop(*_):
            kill_group(proc, 5)
            raise KeyboardInterrupt
        for s in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, stop)
        rc, timed_out = None, False
        try:
            rc = proc.wait(timeout=timeout_min * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            log.write(f"\n[{tool}] timeout after {timeout_min:g} min: killing the job\n")
            rc = EXIT_TIMEOUT
        except KeyboardInterrupt:
            rc = 130
        finally:
            kill_group(proc, 5)                      # anything the job left running
    end = time.time()
    n = sum(1 for _ in open(g / "ledger.jsonl")) + 1 if (g / "ledger.jsonl").exists() else 1
    entry = {"job": n, "round": None, "team": None, "agent": os.environ.get("AUTOINFER_AGENT", "researcher"),
             "label": label, "start": now_iso(start), "end": now_iso(end),
             "gpu_minutes": round((end - start) / 60, 3), "exit_code": rc, "timed_out": timed_out,
             "cancelled": rc == 130, "charged": True}
    with open(g / "ledger.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{tool}] {label}: exit {rc} after {(end - start) / 60:.1f} min; {limits_line(run)}", flush=True)
    return rc
