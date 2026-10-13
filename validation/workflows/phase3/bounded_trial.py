"""Conditional trust-region decisions from residual and Jacobian error bounds.

All inputs use the same weighted residual coordinates. The caller must justify
the supplied error bounds; this module does not turn k-pair differences into
certified error bounds. Its conditional action alone never accepts DFT outputs.
"""
import math
from loss_intervals import loss_interval, reduction_interval


def decide(before, after, jacobian, step, before_errors, after_errors,
           jacobian_errors, eta=.1):
    n, p = len(before), len(step)
    if not n or not p or len(after)!=n or len(jacobian)!=n or len(jacobian_errors)!=n:
        raise ValueError('Inconsistent trial dimensions')
    if len(before_errors)!=n or len(after_errors)!=n:
        raise ValueError('Inconsistent residual error dimensions')
    if not math.isfinite(eta) or not 0 < eta < 1:
        raise ValueError('Acceptance ratio threshold must lie in (0,1)')
    if any(not math.isfinite(x) for x in step):
        raise ValueError('Nonfinite parameter step')
    for row, errors in zip(jacobian,jacobian_errors):
        if len(row)!=p or len(errors)!=p:
            raise ValueError('Inconsistent Jacobian columns')
        if any(not math.isfinite(x) for x in row+errors) or any(x<0 for x in errors):
            raise ValueError('Invalid Jacobian or error bounds')
    actual=reduction_interval(before,after,before_errors,after_errors)
    model=[r+math.fsum(j*s for j,s in zip(row,step)) for r,row in zip(before,jacobian)]
    model_errors=[e+math.fsum(d*abs(s) for d,s in zip(row,step))
                  for e,row in zip(before_errors,jacobian_errors)]
    base_lo,base_hi=loss_interval(before,before_errors)
    model_lo,model_hi=loss_interval(model,model_errors)
    # Reusing the same baseline in both terms is correlated. Independent
    # interval subtraction remains an outer bound, potentially conservative.
    pred_lo,pred_hi=base_lo-model_hi,base_hi-model_lo
    actual_lo,actual_hi=actual['reduction_interval']
    nominal_pred=math.fsum(x*x for x in before)-math.fsum(x*x for x in model)
    ratio=None
    if pred_lo>0:
        quotients=[a/b for a in (actual_lo,actual_hi) for b in (pred_lo,pred_hi)]
        ratio=[min(quotients),max(quotients)]
    if actual_hi<0:
        action,reason='reject','resolved_loss_increase'
    elif pred_hi<=0:
        action,reason='reject','model_has_no_possible_positive_reduction'
    elif pred_lo<=0:
        action,reason='refine','predicted_reduction_unresolved'
    elif actual_lo>0 and ratio[0]>eta:
        action,reason='accept','worst_case_reduction_and_ratio_pass'
    elif ratio[1]<=eta:
        action,reason='reject','best_case_ratio_fails'
    else:
        action,reason='refine','actual_reduction_or_ratio_unresolved'
    return dict(**actual, predicted_reduction_interval=[pred_lo,pred_hi],
                nominal_predicted_reduction=nominal_pred,rho_interval=ratio,
                eta=eta,conditional_action=action,reason=reason,
                production_accepted=False,
                scope='Conditional on supplied error bounds; physical gates and bound justification remain external')
