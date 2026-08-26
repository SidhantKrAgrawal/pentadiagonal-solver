import csv, os
R="/dcs/pg25/u5736287/Documents/Dissertation-20260616T113352Z-3-001/Dissertation/pentadsolver-gpu/results"
G={"cobra":"grid_sweep_cobra-01_20260811_000719","panda":"grid_sweep_panda-01_20260811_000719"}
NS=[128,256,320,384]
HOSTS=["cobra","panda"]
def rd(h,f):
    with open(os.path.join(R,G[h],f)) as fh: return list(csv.DictReader(fh))
gpu={h:rd(h,"gpu.csv") for h in G}
cpu={h:rd(h,"cpu.csv") for h in G}

def cpu_best(h,N,prec):
    rows=[r for r in cpu[h] if int(r["N"])==N and r["precision"]==prec and r["total_ms"]!="fail"]
    if not rows: return None
    b=min(rows,key=lambda r:float(r["total_ms"]))
    return dict(tag=f"{b['threads']}T", tot=float(b["total_ms"]),
                x=float(b["x_ms"]), y=float(b["y_ms"]), z=float(b["z_ms"]))

def gpu_row(h,algo,N,prec,pick_best_lane=False):
    rows=[r for r in gpu[h] if r["algo"]==algo and int(r["N"])==N
          and r["precision"]==prec and r["status"]=="ok"]
    if not rows: return None
    if pick_best_lane:
        b=min(rows,key=lambda r:float(r["x_ms"]))          # lanes only affect x
        tag=f"L={b['lanes']}"
    else:
        b=rows[0]; tag=""
    return dict(tag=tag, tot=float(b["e2e_wall_ms"]), x=float(b["x_ms"]),
                y=float(b["y_ms"]), z=float(b["z_ms"]))

SPECS=[("CPU OpenMP (best threads)", lambda h,N,p: cpu_best(h,N,p)),
       ("GPU Naive",                 lambda h,N,p: gpu_row(h,"naive",N,p)),
       ("Global-Transpose",          lambda h,N,p: gpu_row(h,"transpose",N,p)),
       ("Hybrid Thomas-PCR (best L)",lambda h,N,p: gpu_row(h,"thomas-pcr",N,p,True)),
       ("Shared-Factorisation *",    lambda h,N,p: gpu_row(h,"shared-fact",N,p))]

O=[]
def w(s=""): O.append(s)

w("---")
w("title: Pentadiagonal ADI solver — per-grid timing tables")
w("subtitle: 'Four grids x two precisions x five algorithms x two GPUs'")
w("date: 11 August 2026")
w("geometry: landscape, margin=1.6cm")
w("fontsize: 9pt")
w("mainfont: DejaVu Serif")
w("monofont: DejaVu Sans Mono")
w("---")
w()
w("All times in **milliseconds per ADI iteration**. `x`, `y`, `z` are the "
  "per-direction solve times; `total` is the full iteration.")
w()
w("For the CPU the thread count giving the lowest total is used, and named in the "
  "row. For Hybrid Thomas-PCR the lane count giving the lowest x-solve is used, "
  "and named in the row — lanes affect only the x-direction.")
w()
w("**Two things to read carefully.** GPU `total` is measured end-to-end and "
  "includes inter-kernel gaps, so it is slightly larger than x+y+z; the CPU total "
  "is exactly x+y+z. And the Thomas-PCR row drives that algorithm in **all three** "
  "directions, which is not how it would be deployed — its y and z figures are "
  "therefore poor in FP64 and its total reflects that.")
w()
w("\\* Shared-Factorisation is a **restricted** solver: it assumes every line in a "
  "direction shares the same coefficients. Shown for reference; excluded from any "
  "general-purpose claim.")
w()

for N in NS:
    for prec,plab in (("double","FP64"),("float","FP32")):
        w(f"\n## {N}³ — {plab}\n")
        w("| algorithm | cobra total | cobra x | cobra y | cobra z | "
          "panda total | panda x | panda y | panda z |")
        w("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label,fn in SPECS:
            cells=[]; tags=[]
            for h in HOSTS:
                d=fn(h,N,prec)
                if d is None:
                    cells += ["—"]*4
                else:
                    cells += [f"**{d['tot']:.2f}**", f"{d['x']:.2f}",
                              f"{d['y']:.2f}", f"{d['z']:.2f}"]
                    if d["tag"]: tags.append(f"{h} {d['tag']}")
            name=label
            if tags: name=f"{label.split(' (')[0]} ({', '.join(tags)})"
            w(f"| {name} | " + " | ".join(cells) + " |")
        w()
print("\n".join(O))
