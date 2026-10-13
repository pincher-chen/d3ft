# Figure4 — Posterior bands of five systems versus QE

Main text,Section4.4.Accuracy and robustness across materials.

|Panel|Content|
|---|---|
|(a)|Al,4atom conventional fcc cell,14bands;folded cubic path.|
|(b)|Na,2atom conventional bcc cell,9bands;folded cubic path,NLCC corrected.|
|(c)|Si,2atom diamond primitive cell,12bands.|
|(d)|SiC,2atom zincblende primitive cell,12bands.|
|(e)|GaAs,2atom zincblende primitive cell,17bands including Ga3d.|
|(f)|Aligned absolute-error distributions for all compared eigenvalues,log energy axis.|

## data/
- `<system>/band_comparison.csv`: `point,band,distance_inverse_A,candidate_Ha,QE_Ha,aligned_delta_eV,label`. Point and band indices are0-based. Labels occur at high-symmetry nodes. Values are copies,not rounded or refitted.
- `<system>/path.json`: fractional coordinates,reciprocal distances,node indices and branch boundaries. Si/SiC/GaAs:Gamma-X-W-L-Gamma-K,56points. Al/Na:Gamma-X-M-Gamma-R-X|M-R,62points; `|` marks disconnected branches. These conventional-cell bands are folded; no primitive-cell unfolding is claimed.
- `summary.csv`: per-system MAE,maximum deviation,RMSE,original-grid overlaps,Ritz residual and empty-state buffer.
- `QA.json`: full completed computation QA,per-system error groups,SCF chemical potentials,source checksums,Ritz residuals and Na oldFigureS1 overlap checks. GaAs is a byte-preserving reuse of gaas_bands_20261013.
- `PROTOCOL.json`: exact Table1 physics,accepted state paths and seeds. Al/Si/SiC/GaAs seed301 at4000;Na repaired seed13101 at4000. No parameters retuned. Forty states were solved independently; only Table1 b are plotted.
- `SOURCES.json`: each local file,original absolute path andSHA256. The script verifies local copies; no original HPC directory is required for plotting.

## Energy zero and alignment
Each system uses exactly one global mean shift over all path points and plotted bands. Al/Na curves are relative to their own charge-consistent chemical potentials from the SCF grid; applying the same shift to candidate energies and chemical potential preserves their difference. Semiconductors are relative to the QE path VBM,with the global shift applied to candidate energies. Panel(f) uses the archived globally aligned delta column. Zero-valued differences are displayed at1e-12eV only on the log axis; data/statistics are unmodified.

## script/
Run from any directory:

    python /absolute/path/to/Figure_4/script/plot_figure_4.py

Produces `Figure_4.pdf`, `Figure_4.png` and `Figure_4_QA.json`. Requires only Python,numpy,matplotlib. No GPU,Slurm,QE or scientific solver is invoked. Style follows the other Figures packages:Liberation Sans with fallback,8pt labels/titles,7pt ticks,0.8pt axes and inward ticks. Canvas7x5in; PDF embedded fonts,PNG350dpi. All labels must lie inside the canvas.

Computations are preserved in paper/jctc/bands_all_20261013; Na's failed Davidson NSCF and successful same-density CG recovery remain there. The independent posterior eigsh and QE results passed original-grid checks; Na additionally agrees with old FigureS1 at shared points.

The release MANIFEST verifies immutable inputs and plotting code. PDF/PNG/QA are regenerated products; original render hashes are retained in provenance/checks/Figure4_original_manifest.json.
