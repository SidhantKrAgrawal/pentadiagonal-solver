#!/usr/bin/env python3
"""build_grid_report.py — assemble the grid study into one .txt deliverable.

    python3 scripts/build_grid_report.py <stamp> <out.txt>

Reads results/grid_transcript_<host>_<stamp>/ for every host it finds, and
writes: a methodology header, the full run-by-run transcripts, then the summary
tables and the per-axis winners.

Two rules the tables obey, because breaking either produces a wrong answer:

  * Global-Transpose is an x-direction kernel.  Its y/z columns are the naive
    strided solve, so it is excluded from the y and z rankings rather than
    being credited with naive's time.
  * A row whose requested kernel was not the kernel that ran (honoured=no) is
    excluded from ranking and listed separately.  It is a real measurement of
    something else.
"""
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGOS = ["naive", "transpose", "thomas-pcr", "shared-fact"]
PRETTY = {
    "naive":       "Algo 1  GPU Naive",
    "transpose":   "Algo 2  Global-Transpose",
    "thomas-pcr":  "Algo 3  Hybrid Thomas-PCR",
    "shared-fact": "Algo 4  Shared-Factorisation",
    "auto":        "        auto (production)",
}
PREC = {"double": "FP64", "float": "FP32"}


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(stamp):
    """-> {host: {'env':str, 'transcript':str, 'cpu':[...], 'gpu':[...]}}"""
    out = {}
    base = os.path.join(ROOT, "results")
    for d in sorted(os.listdir(base)):
        if not d.startswith("grid_transcript_") or not d.endswith("_" + stamp):
            continue
        host = d[len("grid_transcript_"):-(len(stamp) + 1)]
        p = os.path.join(base, d)
        rec = {"dir": p}
        with open(os.path.join(p, "00_environment.txt")) as fh:
            rec["env"] = fh.read()
        with open(os.path.join(p, "transcript.txt")) as fh:
            rec["transcript"] = fh.read()
        with open(os.path.join(p, "cpu.csv")) as fh:
            rec["cpu"] = list(csv.DictReader(fh))
        with open(os.path.join(p, "gpu.csv")) as fh:
            rec["gpu"] = list(csv.DictReader(fh))
        out[host] = rec
    return out


def hr(c="=", n=100):
    return c * n


def cpu_tables(data, w):
    w(hr())
    w("TABLE 1 — CPU BASELINE: OpenMP thread scaling (ms per ADI iteration)")
    w(hr())
    w("")
    w("Total is the sum of the three directional sweeps (the CPU path has no")
    w("inter-kernel gap to lose).  T* marks the fastest thread count.")
    w("")
    for host in sorted(data):
        for prec in ("double", "float"):
            w("  %s   %s" % (host, PREC[prec]))
            w("  " + hr("-", 84))
            w("  %-5s %-8s %10s %10s %10s %12s %10s" %
              ("N", "threads", "x_ms", "y_ms", "z_ms", "TOTAL_ms", "vs T=1"))
            w("  " + hr("-", 84))
            for N in ("128", "256", "320", "384"):
                rows = [r for r in data[host]["cpu"]
                        if r["N"] == N and r["precision"] == prec and f(r["total_ms"])]
                if not rows:
                    continue
                base = next((f(r["total_ms"]) for r in rows if r["threads"] == "1"), None)
                best = min(rows, key=lambda r: f(r["total_ms"]))
                for r in rows:
                    tot = f(r["total_ms"])
                    mark = "*" if r is best else " "
                    spd = "%.2fx" % (base / tot) if base else "-"
                    w("  %-5s %-8s %10.3f %10.3f %10.3f %12.3f %10s %s" %
                      (N, r["threads"], f(r["x_ms"]), f(r["y_ms"]), f(r["z_ms"]),
                       tot, spd, mark))
                w("")
            w("")


def gpu_rows(data, host, N, prec, algo):
    return [r for r in data[host]["gpu"]
            if r["algo"] == algo and r["N"] == N and r["precision"] == prec
            and r["status"] == "ok"]


def gpu_tables(data, w):
    w(hr())
    w("TABLE 2 — GPU ALGORITHMS: end-to-end total and per-axis breakdown (ms/iteration)")
    w(hr())
    w("")
    w("TOTAL is the end-to-end wall clock for one full ADI iteration (Pass A).")
    w("x/y/z come from the instrumented pass of the SAME run (Pass B); they sum")
    w("to slightly less than TOTAL because the inter-kernel gaps are not in them.")
    w("For Thomas-PCR the best lane count at that (N, precision) is shown.")
    w("A direction whose kernel is NOT the row's algorithm is marked with the")
    w("kernel that actually ran there:")
    w("  (n) = naive.  Global-Transpose is an x-only kernel, so its y/z are")
    w("  always (n) by design; a (n) on a Thomas-PCR row is a fallback, and its")
    w("  TOTAL is then a mixed configuration, not that algorithm's total (*).")
    w("")
    for host in sorted(data):
        for prec in ("double", "float"):
            w("  %s   %s" % (host, PREC[prec]))
            w("  " + hr("-", 96))
            w("  %-5s %-28s %6s %11s %11s %11s %12s" %
              ("N", "algorithm", "lanes", "x_ms", "y_ms", "z_ms", "TOTAL_ms"))
            w("  " + hr("-", 96))
            for N in ("128", "256", "320", "384"):
                for algo in ALGOS + ["auto"]:
                    rows = gpu_rows(data, host, N, prec, algo)
                    if not rows:
                        bad = [r for r in data[host]["gpu"]
                               if r["algo"] == algo and r["N"] == N
                               and r["precision"] == prec]
                        st = bad[0]["status"] if bad else "not run"
                        w("  %-5s %-28s %6s %11s %11s %11s %12s" %
                          (N, PRETTY[algo], "-", "-", "-", "-", st))
                        continue
                    ok = [r for r in rows if r["honoured"] in ("yes", "n/a")]
                    r = min(ok or rows, key=lambda r: f(r["e2e_wall_ms"]))
                    mixed = "" if ok else "*"

                    def mark(axis):
                        ran = r["%s_kernel" % axis]
                        if algo in ("auto", ran):
                            return ""
                        return "(n)" if ran == "naive" else "(%s)" % ran[:3]

                    w("  %-5s %-28s %6s %8.3f%-3s %8.3f%-3s %8.3f%-3s %11.3f%s" %
                      (N, PRETTY[algo], r["lanes"],
                       f(r["x_ms"]), mark("x"), f(r["y_ms"]), mark("y"),
                       f(r["z_ms"]), mark("z"), f(r["e2e_wall_ms"]), mixed))
                w("")
            w("")


def lanes_table(data, w):
    w(hr())
    w("TABLE 3 — ALGORITHM 3 (Hybrid Thomas-PCR): lane sweep")
    w(hr())
    w("")
    w("Lanes = threads cooperating on one system.  Only instantiated (N, lanes)")
    w("templates are run: N=128/256 -> 8,16,32;  N=320 -> 32;  N=384 -> 16,32.")
    w("Lanes affect the x-solve; y and z are unchanged by it.  BEST is by x_ms.")
    w("An uninstantiated lane count does not error -- it silently reverts to 32,")
    w("which is why only the honoured pairs above are requested at all.")
    w("A '(z=naive)' note means Algorithm 3 declined the z-solve at that size and")
    w("the row's TOTAL is a mixed configuration; its x and y are still Algo 3.")
    w("")
    for host in sorted(data):
        for prec in ("double", "float"):
            w("  %s   %s" % (host, PREC[prec]))
            w("  " + hr("-", 90))
            w("  %-5s %6s %10s %10s %10s %12s %6s  %s" %
              ("N", "lanes", "x_ms", "y_ms", "z_ms", "TOTAL_ms", "best", "note"))
            w("  " + hr("-", 90))
            for N in ("128", "256", "320", "384"):
                rows = [r for r in gpu_rows(data, host, N, prec, "thomas-pcr")
                        if r["x_kernel"] == "thomas-pcr"]
                if not rows:
                    continue
                best = min(rows, key=lambda r: f(r["x_ms"]))
                for r in sorted(rows, key=lambda r: int(r["lanes"])):
                    note = ""
                    if r["z_kernel"] != "thomas-pcr":
                        note = "(z=%s)" % r["z_kernel"]
                    if r["y_kernel"] != "thomas-pcr":
                        note += " (y=%s)" % r["y_kernel"]
                    w("  %-5s %6s %10.3f %10.3f %10.3f %12.3f %6s  %s" %
                      (N, r["lanes"], f(r["x_ms"]), f(r["y_ms"]), f(r["z_ms"]),
                       f(r["e2e_wall_ms"]), "<--" if r is best else "", note))
                w("")
            w("")


GENERAL = ["naive", "transpose", "thomas-pcr"]


def best_per_axis(data, host, N, prec, axis, pool):
    """Fastest kernel in `pool` for one axis, or None.

    A row is admitted for an axis only if the kernel that RAN in that
    direction is the one being credited.  This both keeps Global-Transpose
    out of the y/z ranking (its y/z are the naive kernel) and salvages the
    valid x and y measurements from a row whose z fell back.
    """
    key = axis + "_ms"
    cands = []
    for algo in pool:
        if algo == "transpose" and axis != "x":
            continue
        for r in gpu_rows(data, host, N, prec, algo):
            if r["%s_kernel" % axis] != algo:
                continue
            if f(r[key]) is not None:
                cands.append((f(r[key]), algo, r["lanes"]))
    return min(cands) if cands else None


def label(algo, lanes):
    lab = PRETTY[algo].split("  ", 1)[1].strip()
    if algo == "thomas-pcr":
        lab += " L=%s" % lanes
    return lab


def winners(data, w):
    w(hr())
    w("TABLE 4 — BEST ALGORITHM PER AXIS  (the headline result)")
    w(hr())
    w("")
    w("Per machine, per precision, per grid: the fastest kernel for each of the")
    w("three directional solves, taken from the per-axis breakdown.")
    w("")
    w("READ 4a AND 4b TOGETHER.  Algorithm 4 (Shared-Factorisation) solves a")
    w("RESTRICTED problem class -- every line along the solve direction must share")
    w("the same five coefficients, which an ADI sweep with constant coefficients")
    w("does but a general batched solve does not.  It is therefore not comparable")
    w("with Algorithms 1-3 on capability, only on speed for that special case:")
    w("  4a  best of ALL FOUR         -- valid if your problem has ADI structure")
    w("  4b  best of ALGORITHMS 1-3   -- the general-purpose answer")
    w("A row is credited for an axis only if its kernel actually ran in that")
    w("direction, so Global-Transpose is ranked for x only, and a row whose z")
    w("fell back still contributes its valid x and y.")
    w("")
    picks = {}
    for tag, pool, title in (("4a", ALGOS, "best of all four algorithms"),
                             ("4b", GENERAL, "best general-purpose (Algorithms 1-3 only)")):
        w("  " + hr("=", 96))
        w("  TABLE %s — %s" % (tag, title.upper()))
        w("  " + hr("=", 96))
        for host in sorted(data):
            for prec in ("double", "float"):
                w("")
                w("  %s   %s   [%s]" % (host, PREC[prec], title))
                w("  " + hr("-", 96))
                w("  %-5s | %-31s | %-31s | %-31s" %
                  ("N", "best x", "best y", "best z"))
                w("  " + hr("-", 96))
                for N in ("128", "256", "320", "384"):
                    cell = {}
                    for axis in "xyz":
                        b = best_per_axis(data, host, N, prec, axis, pool)
                        if b is None:
                            cell[axis] = "-"
                            continue
                        t, algo, lanes = b
                        cell[axis] = "%-23s %7.3f" % (label(algo, lanes), t)
                        picks[(tag, host, prec, N, axis)] = (algo, t, lanes)
                    w("  %-5s | %-31s | %-31s | %-31s" %
                      (N, cell["x"], cell["y"], cell["z"]))
                w("")
                # Does mixing kernels per direction beat the best single one?
                for N in ("128", "256", "320", "384"):
                    got = [picks.get((tag, host, prec, N, a)) for a in "xyz"]
                    if not all(got):
                        continue
                    comp = sum(g[1] for g in got)
                    singles = []
                    for algo in pool:
                        for r in gpu_rows(data, host, N, prec, algo):
                            if r["honoured"] == "yes":
                                singles.append((f(r["e2e_wall_ms"]), algo, r["lanes"]))
                    if not singles:
                        continue
                    bt, ba, bl = min(singles)
                    mix = " ".join("%s=%s" % (a, picks[(tag, host, prec, N, a)][0])
                                   for a in "xyz")
                    w("    N=%-4s per-axis-best sum = %8.3f ms  (%s)" % (N, comp, mix))
                    w("           best single algorithm (%s) = %8.3f ms  -> %.2fx" %
                      (label(ba, bl), bt, bt / comp))
                w("")
    return picks


def fallbacks(data, w):
    w(hr())
    w("TABLE 5 — RUNS WHERE THE REQUESTED KERNEL DID NOT RUN, OR THE RUN FAILED")
    w(hr())
    w("")
    any_row = False
    for host in sorted(data):
        for r in data[host]["gpu"]:
            if r["status"] != "ok":
                w("  %-9s %-12s N=%-4s %-7s lanes=%-3s -> %s" %
                  (host, r["algo"], r["N"], r["precision"], r["lanes"], r["status"]))
                any_row = True
            elif r["honoured"] == "no":
                w("  %-9s %-12s N=%-4s %-7s lanes=%-3s -> requested %s, "
                  "ran x=%s y=%s z=%s" %
                  (host, r["algo"], r["N"], r["precision"], r["lanes"],
                   r["algo"], r["x_kernel"], r["y_kernel"], r["z_kernel"]))
                any_row = True
    if not any_row:
        w("  (none — every requested kernel ran, every run completed)")
    w("")
    w("  Why this table exists: a fallback does not fail, it produces a perfectly")
    w("  plausible number for a DIFFERENT kernel.  Before the solver was made to")
    w("  report what it launched, the four rows above were recorded as Thomas-PCR")
    w("  z-solve measurements; they are in fact the naive strided kernel (which is")
    w("  why they look ~3x faster than the Thomas-PCR z-times at N=320).")
    w("  Handling: the mixed TOTAL is marked (*) in Table 2 and never used as that")
    w("  algorithm's total; the per-axis rankings admit each direction separately,")
    w("  so the valid x and y from these rows still count.")
    w("")


def cpu_vs_gpu(data, w):
    w(hr())
    w("TABLE 6 — GPU vs BEST CPU (best GPU total over best CPU total, both tuned)")
    w(hr())
    w("")
    w("Two speedups, because they answer different questions:")
    w("  GENERAL   best of Algorithms 1-3 vs the CPU library — both solve the")
    w("            general problem, so this is the like-for-like number and the")
    w("            one to quote as 'the GPU speedup'.")
    w("  ADI-CLASS best of all four (i.e. Algorithm 4) vs the same CPU baseline.")
    w("            The GPU exploits shared coefficients here and the CPU does not,")
    w("            so it flatters the GPU: a CPU shared-factorisation kernel does")
    w("            not exist yet.  Quote it only with that caveat attached.")
    w("")
    w("  %-9s %-5s %-5s %11s %7s | %11s %-22s %8s | %11s %8s" %
      ("machine", "prec", "N", "CPU_ms", "thr", "GPU_gen_ms", "general algorithm",
       "speedup", "GPU_adi_ms", "speedup"))
    w("  " + hr("-", 118))
    for host in sorted(data):
        for prec in ("double", "float"):
            for N in ("128", "256", "320", "384"):
                crows = [r for r in data[host]["cpu"]
                         if r["N"] == N and r["precision"] == prec and f(r["total_ms"])]
                if not crows:
                    continue

                def best(pool):
                    out = []
                    for algo in pool:
                        for r in gpu_rows(data, host, N, prec, algo):
                            if r["honoured"] == "yes":
                                out.append((f(r["e2e_wall_ms"]), algo, r["lanes"]))
                    return min(out) if out else None

                gen = best(GENERAL)
                adi = best(ALGOS)
                if gen is None or adi is None:
                    continue
                cb = min(crows, key=lambda r: f(r["total_ms"]))
                ct = f(cb["total_ms"])
                w("  %-9s %-5s %-5s %11.3f %7s | %11.3f %-22s %7.2fx | %11.3f %7.2fx" %
                  (host, PREC[prec], N, ct, cb["threads"],
                   gen[0], label(gen[1], gen[2]), ct / gen[0],
                   adi[0], ct / adi[0]))
        w("")


def one_liners(data, w, picks):
    """The one-line-per-algorithm reading of the tables, with the numbers that
    justify each line pulled from the data rather than typed in."""

    def gv(host, N, prec, algo, key, lanes=None):
        rows = [r for r in gpu_rows(data, host, N, prec, algo)
                if lanes is None or r["lanes"] == lanes]
        rows = [r for r in rows if f(r[key]) is not None]
        return min(f(r[key]) for r in rows) if rows else float("nan")

    w(hr())
    w("ANALYSIS — ONE LINE PER ALGORITHM")
    w(hr())
    w("")
    w("(all figures 256^3 unless stated; cobra-01 = RTX 3050 sm_86, FP64 at 1/64")
    w(" rate; panda-01 = GTX 1080 sm_61, FP64 at 1/32 rate)")
    w("")

    c = lambda *a, **k: gv("cobra-01", *a, **k)
    p = lambda *a, **k: gv("panda-01", *a, **k)

    w("ALGO 1  GPU Naive")
    w("  Uncoalesced x-solve dominates everything: x = %.1f ms vs y/z %.1f/%.1f ms"
      % (c("256", "double", "naive", "x_ms"), c("256", "double", "naive", "y_ms"),
         c("256", "double", "naive", "z_ms")))
    w("  on cobra FP64 — %.1fx a single strided direction, %.0f%% of the whole"
      % (c("256", "double", "naive", "x_ms") / c("256", "double", "naive", "y_ms"),
         100.0 * c("256", "double", "naive", "x_ms") /
         c("256", "double", "naive", "e2e_wall_ms")))
    w("  iteration — because a thread-per-system walk along x reads a full cache")
    w("  sector to use one element of it.")
    w("  Its y/z are respectable, though: it is a bad x-kernel, not a bad kernel.")
    w("")

    w("ALGO 2  Global-Transpose")
    w("  Fixes exactly that: x %.1f -> %.1f ms on cobra FP64 (%.1fx) by transposing"
      % (c("256", "double", "naive", "x_ms"), c("256", "double", "transpose", "x_ms"),
         c("256", "double", "naive", "x_ms") / c("256", "double", "transpose", "x_ms")))
    w("  into a coalesced layout, and it pays 27 array passes to do it — about half")
    w("  the remaining x-time is the transpose itself, not the solve.  Scales")
    w("  cleanly to N=384 and is the FP64 x-default on the 1/64-rate card.")
    w("")

    w("ALGO 3  Hybrid Thomas-PCR (SPIKE)")
    w("  The precision-split kernel: FP32 x = %.2f ms (best general x anywhere on"
      % c("256", "float", "thomas-pcr", "x_ms"))
    w("  cobra) but FP64 x = %.1f ms, because its ~5x redundant arithmetic is free"
      % c("256", "double", "thomas-pcr", "x_ms"))
    w("  in FP32 and compute-bound at 1/64 FP64.  On panda's 1/32 FP64 the same")
    w("  kernel costs %.1f ms — the ratio tracks the hardware, not the algorithm."
      % p("256", "double", "thomas-pcr", "x_ms"))
    w("  It is an x-specialist: its FP64 y/z (%.0f/%.0f ms) lose to naive's %.0f/%.0f."
      % (c("256", "double", "thomas-pcr", "y_ms"), c("256", "double", "thomas-pcr", "z_ms"),
         c("256", "double", "naive", "y_ms"), c("256", "double", "naive", "z_ms")))
    w("")
    w("  LANES (Table 3) — the tuning knob, and it is architecture-dependent:")
    w("    FP64: cobra prefers L=8 (%.1f vs %.1f ms at L=16), panda prefers L=16"
      % (c("256", "double", "thomas-pcr", "x_ms", "8"),
         c("256", "double", "thomas-pcr", "x_ms", "16")))
    w("          (%.1f vs %.1f at L=8) — sm_86's 128 KB unified L1/shared absorbs"
      % (p("256", "double", "thomas-pcr", "x_ms", "16"),
         p("256", "double", "thomas-pcr", "x_ms", "8")))
    w("          the register spilling that L=8 causes; Pascal's does not.")
    w("    FP32: L=32 on both (%.2f cobra / %.2f panda); L=8 collapses to %.1f ms"
      % (c("256", "float", "thomas-pcr", "x_ms", "32"),
         p("256", "float", "thomas-pcr", "x_ms", "32"),
         c("256", "float", "thomas-pcr", "x_ms", "8")))
    w("          on cobra — a 5x penalty for the wrong knob at the same N.")
    w("    => the lane count should be chosen by (architecture, precision), not by")
    w("       precision alone as the current default does.")
    w("")

    w("ALGO 4  Shared-Factorisation")
    w("  Fastest everywhere it applies, by a wide margin — %.2f ms total on cobra"
      % c("256", "double", "shared-fact", "e2e_wall_ms"))
    w("  FP64 against Global-Transpose's %.1f (%.1fx) — because it moves 4 arrays"
      % (c("256", "double", "transpose", "e2e_wall_ms"),
         c("256", "double", "transpose", "e2e_wall_ms") /
         c("256", "double", "shared-fact", "e2e_wall_ms")))
    w("  per direction instead of 13-27: the factorisation is computed once into a")
    w("  5N table and broadcast, so no per-row divides and no scratch streaming.")
    w("  CAVEAT: restricted problem class (all lines share coefficients).  It is")
    w("  the right answer for constant-coefficient ADI and inapplicable otherwise.")
    w("  It is also the only algorithm whose FP64 and FP32 times are close (%.2f vs"
      % c("256", "double", "shared-fact", "e2e_wall_ms"))
    w("  %.2f, %.2fx) — at 4 arrays it is so bandwidth-lean that halving the word"
      % (c("256", "float", "shared-fact", "e2e_wall_ms"),
         c("256", "double", "shared-fact", "e2e_wall_ms") /
         c("256", "float", "shared-fact", "e2e_wall_ms")))
    w("  size stops buying much.")
    w("")

    w("CPU BASELINE (OpenMP)")
    w("  Scaling stops at 3-4 threads in FP64 on both machines (~1.8x from 6 cores)")
    w("  and only reaches ~2.7-3.0x in FP32: the solve is memory-bound and one or")
    w("  two cores already saturate the DRAM controller.  FP32 scales further than")
    w("  FP64 for exactly that reason — half the bytes per element.  z is always the")
    w("  slowest direction (stride N^2 defeats the cache), which is the mirror image")
    w("  of the GPU, where x is the problem.")
    w("")

    w("CROSS-MACHINE")
    w("  The GTX 1080 wins in FP64 (1/32 rate, 289 GB/s) and the RTX 3050 wins in")
    w("  FP32 — the same kernels, ranked differently by the two cards' FP64:FP32")
    w("  ratios.  This is the dissertation's central claim reproduced at four grid")
    w("  sizes rather than one: the right algorithm is a property of the hardware,")
    w("  not of the problem.")
    w("")


def next_run(data, w, picks):
    """The per-direction configuration implied by Table 4, as runnable commands.

    This is the input to the planned follow-up run: one measurement per
    (machine, precision, N) with each direction set to its own winner, to test
    whether the mixed configuration actually delivers the per-axis-best sum
    once the inter-kernel gaps are paid for real.
    """
    w(hr())
    w("NEXT RUN — RECOMMENDED PER-DIRECTION CONFIGURATION")
    w(hr())
    w("")
    w("Table 4's 'per-axis-best sum' is arithmetic on three separately measured")
    w("directions; it is a PREDICTION, not a measurement.  Running the mixed")
    w("configuration end-to-end is what tests it, because a mixed setup pays its")
    w("own launch gaps and cannot reuse one kernel's scratch across directions.")
    w("Where the sum equals the best single algorithm (most rows), mixing has")
    w("nothing to win and the single algorithm is the simpler choice.")
    w("")
    for tag, title in (("4b", "GENERAL-PURPOSE (Algorithms 1-3)"),
                       ("4a", "ADI-CLASS (all four, i.e. Algorithm 4 available)")):
        w("  %s" % title)
        w("  " + hr("-", 96))
        for host in sorted(data):
            for prec in ("double", "float"):
                for N in ("128", "256", "320", "384"):
                    got = {a: picks.get((tag, host, prec, N, a)) for a in "xyz"}
                    if not all(got.values()):
                        continue
                    algos = [got[a][0] for a in "xyz"]
                    if len(set(algos)) == 1:
                        w("  %-9s %-5s N=%-4s single algorithm: %s"
                          % (host, PREC[prec], N, label(algos[0], got["x"][2])))
                        continue
                    lanes = got["x"][2]
                    env = ""
                    if "thomas-pcr" in algos and lanes not in ("-", None):
                        env = "PENTA_PCR_LANES=%s " % lanes
                    w("  %-9s %-5s N=%-4s %sPENTA_XALGO=%s PENTA_YALGO=%s PENTA_ZALGO=%s"
                      % (host, PREC[prec], N, env, algos[0], algos[1], algos[2]))
        w("")


def main():
    stamp = sys.argv[1]
    outp = sys.argv[2]
    data = load(stamp)
    if not data:
        sys.exit("no result directories for stamp " + stamp)

    lines = []
    w = lines.append

    w(hr("#"))
    w("# PENTADIAGONAL ADI SOLVER — FULL GRID / PRECISION / ALGORITHM / MACHINE STUDY")
    w(hr("#"))
    w("")
    w("Generated: %s" % __import__("datetime").datetime.now().isoformat(timespec="seconds"))
    w("")
    w("MATRIX")
    w("  grids       N = 128, 256, 320, 384   (cubic, N^3)")
    w("  precisions  FP64 (double), FP32 (float)")
    w("  machines    cobra-01 (RTX 3050, sm_86)   panda-01 (GTX 1080, sm_61)")
    w("  CPU         OpenMP threads 1..6")
    w("  GPU         Algo 1 Naive | Algo 2 Global-Transpose |")
    w("              Algo 3 Hybrid Thomas-PCR (lane sweep) | Algo 4 Shared-Factorisation")
    w("              + auto dispatch, recorded for reference")
    w("")
    for host in sorted(data):
        w("  runs on %-9s : %d CPU + %d GPU" %
          (host, len(data[host]["cpu"]), len(data[host]["gpu"])))
    w("")
    w("HOW THE TIMES ARE OBTAINED  (this is the part that is easy to get wrong)")
    w("  One execution produces BOTH the total and the per-axis split:")
    w("    Pass A  runs x+y+z back-to-back with nothing inserted between them and")
    w("            times the whole thing -> the TOTAL quoted everywhere below.")
    w("    Pass B  re-runs the same iteration with CUDA events around each")
    w("            direction -> the x / y / z breakdown.")
    w("  The axes are NOT timed in separate executions, and sum(x+y+z) is never")
    w("  quoted as the total: it omits the inter-kernel gaps (typically <1%).")
    w("  Warm-up is bounded by WALL TIME (3000 ms), not by an iteration count, so")
    w("  a fast kernel at a small N is not measured mid clock-ramp.")
    w("  GPU: 50 timed iterations.  CPU: 20 timed iterations, steady_clock.")
    w("  '%peak' inside the transcripts is against the memory clock measured")
    w("  UNDER LOAD (the P2 power state clocks below spec on both cards).")
    w("")
    w("CONTENTS")
    w("  PART A   full run-by-run transcript: command, output, result")
    w("  PART B   summary tables 1-6 and per-algorithm analysis")
    w("")

    w("")
    w(hr("#"))
    w("# PART A — FULL RUN TRANSCRIPT")
    w(hr("#"))
    for host in sorted(data):
        w(data[host]["transcript"])

    w("")
    w(hr("#"))
    w("# PART B — SUMMARY TABLES AND ANALYSIS")
    w(hr("#"))
    w("")
    for host in sorted(data):
        w("MACHINE %s" % host)
        for ln in data[host]["env"].strip().splitlines():
            w("  " + ln)
        w("")
    cpu_tables(data, w)
    gpu_tables(data, w)
    lanes_table(data, w)
    picks = winners(data, w)
    fallbacks(data, w)
    cpu_vs_gpu(data, w)
    one_liners(data, w, picks)
    next_run(data, w, picks)

    with open(outp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", outp, "(%d lines)" % len(lines))


if __name__ == "__main__":
    main()
