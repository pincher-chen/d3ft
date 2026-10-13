"""Time synchronized updates without altering gradients or printing per-step logs."""
import json
import time
from d3ft.calc import ground_state as module


class TimingObserver:
    def __init__(self,result):
        self.result=result;self.original=module.create_convergence_checker
        self.events=[];self.started=time.perf_counter()

    def install(self):
        observer=self
        def factory(cfg):
            base=observer.original(cfg)
            class Checker:
                def check(self,value):
                    # Ground-state loop has already block_until_ready'ed the update.
                    observer.events.append(time.perf_counter()-observer.started)
                    return base.check(value)
            return Checker()
        module.create_convergence_checker=factory

    def finish(self):
        module.create_convergence_checker=self.original
        row=dict(synchronized_update_end_seconds=self.events,
                 setup_compile_first_update_seconds=self.events[0] if self.events else None,
                 scope='First duration includes setup, compilation and one update, not isolated compile time. Subsequent differences are synchronized update costs. Logical gradient count=iterations; complete forward-trial counts unavailable without full work logging.')
        (self.result/'update_timings.json').write_text(json.dumps(row,indent=2)+'\n')
