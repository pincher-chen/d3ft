"""Create a local source-data/code review snapshot without copying huge arrays.

This is not a self-contained public release. Raw checkpoints stay in the
evidence tree and are explicitly indexed as external dependencies.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

P=Path(__file__).resolve()
while not (P/'validation').is_dir():
    P=P.parent
D=P/'paper_rewriting_output/final_paper'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(16*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(destination):
    destination.mkdir(exist_ok=False)
    entries=[]
    def copy(source,relative):
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        entries.append(dict(path=str(relative),original=str(source),sha256=sha(target),bytes=target.stat().st_size))
    # The current manuscripts and source data, not old backup manuscripts.
    for suffix in ['.tex','.bib','.bst','.csv']:
        for source in sorted(D.glob('*'+suffix)):
            copy(source,Path('paper')/source.name)
    for name in ['main.pdf','supplementary.pdf','README.md','response_design_to_do_list.md']:
        copy(D/name,Path('paper')/name)
    for source in sorted((D/'figures').rglob('*')):
        if source.is_file():
            copy(source,Path('paper/figures')/source.relative_to(D/'figures'))
    for source in sorted(D.glob('*.py')):
        copy(source,Path('paper_analysis')/source.name)
    for pattern in ['*provenance*.json','*build-check.json','*CHECK_20261006.json','TARGET_TOLERANCE_SCOPE_20261006.json']:
        for source in sorted(D.glob(pattern)):
            relative=Path('evidence_records')/source.name
            if not (destination/relative).exists():
                copy(source,relative)
    runtime=P/'validation/results/response_design_20261002_resource_fill/comparison_runtime'
    for source in sorted(runtime.rglob('*')):
        if source.is_file() and source.suffix in ['.py','.sh','.json']:
            copy(source,Path('code/frozen_comparison_runtime')/source.relative_to(runtime))
    inputs=P/'validation/inputs/response_design_20260930'
    for source in sorted(inputs.rglob('*')):
        if source.is_file():
            copy(source,Path('inputs')/source.relative_to(inputs))
    for name in ['LICENSE','requirements.txt','setup.py']:
        copy(P/name,Path('code/project_metadata')/name)
    for label,directory in [
        ('accepted_gamma_response',P/'validation/results/response_design_20261004_reverse_response_v5'),
        ('xc720_staged_geometry',P/'validation/results/response_design_20261005_numerical_verification/gradient_v7/drivers'),
        ('six_strain_design',P/'validation/results/response_design_20261006_six_strain_design_v2')]:
        for source in sorted(directory.glob('*.py')):
            copy(source,Path('code')/label/source.name)
        if label=='six_strain_design':
            for source in sorted((directory/'drivers').glob('*.py')):
                copy(source,Path('code')/label/'drivers'/source.name)
            for name in ['protocol.json','launch.json','environment_versions.json','optimizer_check.json','resource_preflight.json']:
                if (directory/name).exists():
                    copy(directory/name,Path('evidence_records/six_strain_protocol')/name)
    for source in sorted((P/'paper').glob('*20261006.md')):
        copy(source,Path('notes')/source.name)
    raw_index=dict(
        R1=str(P/'validation/results/response_design_20261004_reverse_response_v5'),
        R2_R3=str(P/'validation/results/response_design_20261004_response_design'),
        R3_dense=str(P/'validation/results/response_design_20261004_reverse_endpoint_k4'),
        R4=str(P/'validation/results/response_design_20261004_compact_training_v2'),
        numerical_refinement=str(P/'validation/results/response_design_20261005_numerical_verification'),
        band_structure=str(P/'validation/results/response_design_20261005_band_structure'),
        pending_six_strain=str(P/'validation/results/response_design_20261006_six_strain_design_v2'),
        third_party_jrystal='${D3FT_ROOT}/third_party/jrystal_dependency',
        third_party_AD_DFPT=str(P/'validation/third_party/ad-dfpt'))
    # Data remain qualified by the existing source-binding records; no claim
    # that omitted arrays have been embedded in this package.
    (destination/'external_raw_data.json').write_text(json.dumps(raw_index,indent=2)+'\n')
    readme='''# Local response_design review snapshot

This package contains the current paper/SI, figure source data, figures,
physical inputs/pseudopotentials, the frozen comparison runtime, accepted
response implementations and the pending six-strain protocol. Original
paths and hashes are preserved in `manifest.json`.

This is a review and source-data snapshot, not a self-contained public
release. Large densities, orbital coefficients, optimizer checkpoints and
third-party jrystal/AD–DFPT dependencies remain external. See
`external_raw_data.json` and archived source-binding records. Pending
six-strain trajectories are not included as accepted results.

Run `python verify.py` to verify files included in this snapshot. The
verification checks local snapshot hashes, not the omitted raw arrays.

To reproduce analyses in the current shared project: use the original
evidence-tree locations and scripts in `paper_analysis`; the existing
analysis scripts retain their original paths and were not silently rewritten
for relocation. `build_response_paper.py --directory paper --stem main` (or
`--stem supplementary`) rebuilds the LaTeX documents on the current system.
Physical reruns require the recorded JAX environment, CUDA/H100 resources,
original raw inputs and reference dependencies. Six-strain reruns should
use a NEW result root and freeze an available allocation/GPU via
`six_strain_design.py --prepare --root NEW_ROOT --jobid ALLOCATION --gpu 0`,
then invoke its frozen `controller.py --run --root NEW_ROOT`.

Before public release: add the raw arrays or a downloadable archive and
relocatable path configuration, identify third-party licenses/releases,
confirm the SiC lattice primary source, and provide an actual repository
version/URL and dataset identifier. Author/funding/contribution/competing-
interest fields remain for author confirmation. No public DOI is invented.
'''
    (destination/'README.md').write_text(readme)
    verifier='''import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for row in manifest['files']:
    path=root/row['path']
    assert path.is_file(), str(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'], str(path)
print(f"Verified {len(manifest['files'])} included files; external raw arrays were not embedded or verified.")
'''
    (destination/'verify.py').write_text(verifier)
    for name in ['external_raw_data.json','README.md','verify.py']:
        target=destination/name
        entries.append(dict(path=name,original='generated',sha256=sha(target),bytes=target.stat().st_size))
    manifest=dict(time=datetime.datetime.now().isoformat(),files=entries,file_count=len(entries),
        bytes=sum(row['bytes'] for row in entries),self_contained=False,public_release=False,
        pending_six_strain_results_included=False,external_raw_data_index='external_raw_data.json')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=prepare(args.output.resolve())
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
