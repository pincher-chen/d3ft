"""Keep E5 cards reserved between its GPU steps; release on pilot terminal."""
from pathlib import Path
import json,time
import e7_ct_pilot as p
state=p.P/'validation/results/response_design_20261015_e5_distributed/queue_status.json'
while True:
    stage=p.load(state).get('stage','')
    if stage=='failed' or stage=='E5_four_card_pilot_complete_E4_deferred':break
    time.sleep(10)
path=p.E/'gpu_reservations.json';rows=p.load(path)
for gpu in ('2','3','4','5'):
    if rows.get(gpu,{}).get('owner')=='E5 staged pilot':rows.pop(gpu)
p.dump(path,rows)
p.note('E5 staged GPU reservation released after '+stage+'; E7 four-card fine control may use actually idle cards, E6 GPU7 reservation retained.')
