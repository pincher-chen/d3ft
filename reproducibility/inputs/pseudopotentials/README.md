# Exact pseudopotential inputs

These are the UPF bytes used by the manuscript runs (not downloaded replacements). Source metadata and embedded creator/author information are retained unchanged. SG15/PBE supplies Si, C, Ga and As; the Ga pseudopotential includes 3d. Al and Na use von Barth–Car norm-conserving LDA files. Mg is an inherited auxiliary input outside the six manuscript systems.

The archive's structure/element SHA256 mapping is in `provenance/environment/pseudopotential_hashes.json`; the actual-byte NLCC inspection is in `reproducibility/analysis/NLCC_scope_20261010/scope.csv`. Only Na among the six manuscript systems contains NLCC. PP_RAB integration, interpolation, and the explicit density cutoff are used by the repaired Na kernel.

Pseudopotential authorship and any embedded UPF terms remain with their original providers; the root software license does not replace those terms. Replacing a UPF changes the Hamiltonian and invalidates direct reference reuse.
