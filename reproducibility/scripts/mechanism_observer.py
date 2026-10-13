"""Record post-update physical parameters; analyze the corrected H in fresh processes.

This observer is absent from isolated timings and validation-seed runs.
"""
import inspect
import json
import numpy as np
from d3ft.calc import ground_state as module


class Observer:
    def __init__(self,result,cfg,init,mask):
        self.result=result;self.cfg=cfg;self.original=module.create_convergence_checker
        self.save(0,init,float(cfg.d3ft.free_energy.temp_init),float(cfg.d3ft.occupation.epsilon_init))

    def save(self,step,params,temp,eps):
        arrays={'params_pw':np.asarray(params['pw'])}
        def flatten(prefix,value):
            if isinstance(value,dict):
                for key,child in value.items():flatten(prefix+'__'+key,child)
            else:arrays[prefix]=np.asarray(value)
        flatten('occ',params['occ'])
        np.savez_compressed(self.result/f'mechanism_params_step{step}.npz',**arrays)
        (self.result/f'mechanism_step{step}.json').write_text(json.dumps(dict(step=step,temperature_Ha=temp,epsilon=eps))+'\n')

    def install(self):
        observer=self
        def factory(cfg):
            base=observer.original(cfg)
            class Checker:
                def check(self,value):
                    values=inspect.currentframe().f_back.f_locals
                    step=int(values['i'])+1
                    if step in {100,300,500,1000,2000,4000,5000,6000}:
                        observer.save(step,values['params'],float(values['temp_t']),float(values['eps_t']))
                    return base.check(value)
            return Checker()
        module.create_convergence_checker=factory

    def uninstall(self):module.create_convergence_checker=self.original
