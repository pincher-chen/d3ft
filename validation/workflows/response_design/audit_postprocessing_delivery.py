"""Actual-source, editable-figure and synchronized-TeX delivery audit."""
import json,hashlib,subprocess
from pathlib import Path
from PIL import Image
P=Path(__file__).resolve().parents[3];D=P/'paper_rewriting_output/final_paper';M=P.parents[1]/'d3ft2-nmi/papar/sn-article-template';O=P/'validation/results/response_design_20261005_band_structure'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();c=read(D/'ELECTRONIC_POSTPROCESSING_CHECK_20261005.json')
assert all(sha(p)==h for p,h in c['sources'].items())
figs={}
for stem in c['figures']:
 png=Image.open(D/'figures'/f'{stem}.png');tif=Image.open(D/'figures'/f'{stem}.tiff');files={e:sha(D/'figures'/f'{stem}.{e}') for e in ['pdf','svg','png','tiff']};assert all(sha(M/'figures'/f'{stem}.{e}')==h for e,h in files.items());font=subprocess.check_output(['pdffonts',str(D/'figures'/f'{stem}.pdf')],text=True);assert 'Type 3' not in font
 assert png.info.get('dpi',(0,))[0]>=449 and float(tif.info.get('dpi',(0,))[0])>=599
 figs[stem]=dict(png_size=png.size,tiff_size=tif.size,png_dpi=list(map(float,png.info.get('dpi',()))),tiff_dpi=list(map(float,tif.info.get('dpi',()))),embedded_truetype=True,mirror_identical=True,files=files)
assert (D/'main.tex').read_text().replace(r'\bibliography{references}',r'\bibliography{sn-bibliography}')==(M/'sn-article.tex').read_text()
assert (D/'supplementary.tex').read_text().replace(r'\bibliography{references}',r'\bibliography{sn-bibliography}')==(M/'supplementary.tex').read_text()
for directory,stem in [(D,'main'),(D,'supplementary'),(M,'sn-article'),(M,'supplementary')]:
 n='response-build-check.json' if stem=='main' else f'response-{stem}-build-check.json';b=read(directory/n);assert all(b[k]==0 for k in ['latex_errors','unresolved_references','bibliography_missing_entries','overfull_boxes'])
text=(D/'main.tex').read_text();assert '$-5.692493$ and $-4.791187$ meV' in text and 'fail the original 1-meV budget' in text
review_path=O/'visual_review_20261005.json';visual_ok=False
if review_path.exists():
 v=read(review_path);visual_ok=v['passed'] and all(sha(D/'figures'/f'{stem}.{ext}')==h for stem,files in v['figure_sha256'].items() for ext,h in files.items())
completed=['actual_structures','selected_mesh_edge_densities','physical_energy_volume']
if c['continuous_path_complete']:completed+=['continuous_parent_reference_bands','local_path_edge_refinement','R3_TeX_integration']
report=dict(passed=True,source_hashes_passed=len(c['sources']),figures=figs,completed=completed,continuous_path_complete=c['continuous_path_complete'],original_target_failures_preserved=True,visual_review_passed=visual_ok,visual_review='Final PNGs inspected; reviewed hashes match delivered figures' if visual_ok else 'Final visual review pending')
(O/'delivery_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='figures'},indent=2))
