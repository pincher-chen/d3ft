"""CPU-only incremental cost/fold report for the already running E7 pilot."""
import datetime, json, time
from e7_ct_pilot import E, P, report, load, dump

old = None
while True:
    status = load(E / 'status.json')
    terminal = status.get('stage') == 'pilot_finished_review_certificates'
    report(final=terminal)
    result = load(E / 'Ct_decision.json')
    key = (tuple(sorted(result['measured_costs_GPU_hours'])),
           tuple(sorted((m,d['passed']) for m,d in result['independent_gates'].items())),
           result['decision'], terminal)
    if key != old:
        text = (datetime.datetime.now().isoformat()+' E7 Ct逐步实测：'+
                json.dumps(result['measured_costs_GPU_hours'],ensure_ascii=False)+
                ' GPU·h；折数判断：'+result['decision']+
                ('（终态，预算及验收范围见Ct_decision.json）' if terminal else '（暂定，独立复核仍在队列）'))
        for path in (P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md'):
            with path.open('a') as f:
                f.write('\n- '+text+'\n')
        print(text,flush=True)
        old = key
    dump(E/'observer_status.json',dict(time=datetime.datetime.now().isoformat(),
        stage='finished' if terminal else 'watching',no_GPU_submitted=True))
    if terminal:
        break
    time.sleep(30)
