#!/usr/bin/env python3
"""Top source lines by warp-stall samples for one kernel in an ncu report (needs a -lineinfo build + --import-source).
Usage: python ncu_hotlines.py report.ncu-rep [kernel_index=0] [top=15]   (kernel_index = n-th kernel matching 'Marlin'; edit -k for others)"""
import csv, io, subprocess, sys, os
rep = sys.argv[1]; skip = sys.argv[2] if len(sys.argv) > 2 else "0"; top = int(sys.argv[3]) if len(sys.argv) > 3 else 15
ncu = os.path.join(os.environ["ENV_PREFIX"], "bin", "ncu")   # source the task env.sh first
out = subprocess.run([ncu, "--import", rep, "--page", "source", "--csv", "--print-source", "cuda,sass", "--kernel-name-base", "demangled",
                      "-k", "regex:Marlin", "--launch-skip", skip, "--launch-count", "1"], capture_output=True, text=True).stdout
rows = list(csv.reader(io.StringIO(out)))
lines, hdr, fname, srcs = [], None, None, {}
for r in rows:
    if not r: continue
    if r[0] == "File Path": fname = r[1]; continue
    if r[0] == "Function Name": continue
    if r[0] == "Line No": hdr = r; continue
    if hdr is None or len(r) < len(hdr): continue
    if r[0] and r[0].isdigit():
        d = dict(zip(hdr, r)); d["Source"] = r[1]; d["file"] = os.path.basename(fname or "?"); lines.append(d)
def num(d, k):
    try: return float(d.get(k, "0"))
    except ValueError: return 0.0
tot = sum(num(d, "Warp Stall Sampling (All Samples)") for d in lines)
print(f"{len(lines)} source lines, total stall samples {tot:.0f}")
stalls = [k for k in lines[0] if k.startswith("stall_") and "Not Issued" not in k] if lines else []
for d in sorted(lines, key=lambda d: -num(d, "Warp Stall Sampling (All Samples)"))[:top]:
    s = num(d, "Warp Stall Sampling (All Samples)")
    why = sorted(((num(d, k), k[6:]) for k in stalls), reverse=True)[:2]
    print(f"{100*s/tot:5.1f}%  {d['file']}:{d['Line No']:>4}  instr={num(d,'Instructions Executed'):.0f}  [{', '.join(f'{n}:{int(v)}' for v,n in why if v)}]  {d['Source'].strip()[:90]}")
