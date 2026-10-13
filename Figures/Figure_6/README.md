# Figure 6 — Cost per update and occupation-layer scaling (one H100)

Main text, Section 4.6.

| Panel | Content |
|---|---|
| (a) | Wall time per 1000 steps versus number of bands (Al 4-atom cell, 2³ mesh, 48³ FFT, 100 Ha, 12 electrons); labels = ratio this work / QR-Adam. Gather implementation. |
| (b) | Occupation-layer latency versus number of k points (b = 52, 52 electrons): global QR versus FD charge root, forward and forward+reverse; medians of 31 evaluations after 3 warm-ups. |

## data/
- `B1_NC_main.csv` — panel (a): `arm`, `bands`, `seconds_per1000`.
- `timings.csv` — panel (b): `K`, `QR_forward`, `QR_forward_and_reverse`, `FD_forward`, `FD_forward_and_reverse` (seconds).
- `slopes.json` — fitted log–log slopes shown in panel (b).
- `SOURCES.json`.

## script/
`python script/plot_figure_5.py` → `Figure_6.pdf/png`.
