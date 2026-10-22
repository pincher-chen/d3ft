# Provenance scopes

`RESPONSE_DESIGN_MANIFEST.json` is the current `response-design` branch inventory,
including both scientific figure collections, source code, examples and checks.
Use `python scripts/verify_response_release.py` (or `scripts/verify_release.py`).

`FIGURE_LAYOUT_MIGRATION.json` maps former numbered figure paths to scientific
function names and records byte preservation for all scientific data and rendered
assets. `Figures/catalogue.json` identifies each package's collection.

`RELEASE_MANIFEST.json`, `COPY_SOURCES.json` and dependency/source records inherited
from the earlier release are historical snapshots: their original paths/hashes
are intentionally retained. They describe the `v1.0` tree, not the current branch.
Check out the `v1.0` tag to run that release's original verifier. They are not a
claim that archived scripts/data belong to the response-design experiments.
