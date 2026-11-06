"""Kernel time by category from an nsys `cuda_gpu_kern_sum` CSV (summed over both GPUs).
usage: python kernel_breakdown.py kern_sum.csv [top=25]
Categories are regexes on kernel names (first match wins); adjust CATS if a new kernel shows up
under "other". Percentages are of total kernel time, not wall time."""
import csv, re, sys

CATS = [
    ("FP8 weight GEMM (Marlin)", r"marlin"),
    ("all-reduce / all-gather (NCCL)", r"nccl"),
    ("other GEMM (cutlass/cublas: lm_head, small bf16 GEMMs)", r"gemm|cutlass::Kernel2|cublas|gemv|splitKreduce"),
    ("GDN / linear attention (delta rule, conv1d)", r"gdn|delta_rule|conv1d|chunk_|recompute_w_u|merge_16x16|mamba|_fused_post_conv|kkt|cumsum"),
    ("attention (full) + KV cache write", r"flash|attention|attn|reshape_and_cache|paged|batch_decode|prefill"),
    ("norms, activations, rope, gating", r"norm|act_and_mul|silu|rope|gate|sigmoid"),
    ("sampling / spec-decode bookkeeping", r"argmax|rejection|resample|logits|sampl|slot_mapping|block_tables|positions"),
]


def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    rows = list(csv.reader(open(path)))
    h = next(i for i, r in enumerate(rows) if r and r[0] == "Time (%)")
    hdr = rows[h]
    ti, ni, name_i = hdr.index("Total Time (ns)"), hdr.index("Instances"), hdr.index("Name")
    kern = [(float(r[ti]), int(r[ni]), r[name_i]) for r in rows[h + 1:] if len(r) > name_i]
    total = sum(t for t, _, _ in kern)
    by = {}
    for t, n, name in kern:
        cat = next((c for c, rx in CATS if re.search(rx, name, re.I)), "other")
        a = by.setdefault(cat, [0.0, 0])
        a[0] += t; a[1] += n
    print(f"total kernel time {total / 1e9:.2f} s (both GPUs), {len(kern)} kernels\n")
    print(f"{'share':>6}  {'time s':>7}  {'calls':>8}  category")
    for cat, (t, n) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"{100 * t / total:5.1f}%  {t / 1e9:7.2f}  {n:8d}  {cat}")
    print(f"\ntop {top} kernels:")
    print(f"{'share':>6}  {'avg us':>8}  {'calls':>8}  name")
    for t, n, name in sorted(kern, reverse=True)[:top]:
        print(f"{100 * t / total:5.1f}%  {t / n / 1e3:8.1f}  {n:8d}  {name[:110]}")


if __name__ == "__main__":
    main()
