"""Keep GPU0 useful after Si cheap, without using the FD/E5 fixed devices."""
import json,time
import e7_ct_pilot as pilot
pilot.priority_pending=lambda: None
for material in ('Si','BN'):
    cheap=pilot.E/'runs'/(material+'_cheap')
    while not (cheap/'native_acceptance.json').exists():
        record=pilot.load(pilot.E/(material+'_cheap_launch.json'))
        if record.get('status')=='failed':break
        time.sleep(20)
    if not pilot.load(cheap/'native_acceptance.json').get('passed'):continue
    if (pilot.E/(material+'_fine_launch.json')).exists():continue
    pilot.run(material,'fine',[0],14400,pilot.load(cheap/'summary.json')['a0_A'])
    pilot.compare(material)
    pilot.report()
