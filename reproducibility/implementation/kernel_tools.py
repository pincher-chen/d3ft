"""Audited, runtime-only variants of the captured frozen Python update.

No original module or source file is edited. Diagnostic outputs expose decisions;
the performance candidate removes only optional debug callbacks and uses scan.
"""
import ast
import hashlib
import inspect
import textwrap
import jax
import jax.numpy as jnp


def transformed_function(fun, diagnostics=False, no_callbacks=False):
    source=textwrap.dedent(inspect.getsource(fun))
    tree=ast.parse(source); definition=tree.body[0]; definition.decorator_list=[]
    changes=[]
    assigned={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    next_expression="next_rate" if "next_rate" in assigned else "config.optimizer_args.learning_rate"
    streak_expression="streak" if "streak" in assigned else "jnp.asarray(0)"
    class Rewrite(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if no_callbacks and node.name=='print_metrics':
                changes.append('remove optional print_metrics definition');return None
            return self.generic_visit(node)
        def visit_Expr(self,node):
            call=node.value
            if (no_callbacks and isinstance(call,ast.Call)
                and isinstance(call.func,ast.Attribute) and call.func.attr=='cond'
                and any(isinstance(n,ast.Name) and n.id=='print_metrics' for n in ast.walk(call))):
                changes.append('remove optional metric callback cond');return None
            return self.generic_visit(node)
        def visit_Return(self,node):
            if diagnostics and isinstance(node.value,ast.Tuple) and isinstance(node.value.elts[0],ast.Name) and node.value.elts[0].id=='params_out':
                extra=ast.parse(f"dict(F_before=loss, F_trial=trial_loss, rate=alpha, reductions=reductions, accepted=accepted, slope=slope, trial_count=reductions+1, next_rate={next_expression}, streak={streak_expression}, rounding=rounding)",mode='eval').body
                node.value.elts.append(extra);changes.append('expose existing scalar decisions')
            return node
    tree=Rewrite().visit(tree);ast.fix_missing_locations(tree)
    if no_callbacks:assert changes.count('remove optional metric callback cond')==1
    if diagnostics:assert changes.count('expose existing scalar decisions')==1
    # Prove that removing precisely the approved nodes recovers the same AST.
    expected=ast.parse(source);expected.body[0].decorator_list=[]
    if no_callbacks:
        class Drop(ast.NodeTransformer):
            def visit_FunctionDef(self,n):
                return None if n.name=='print_metrics' else self.generic_visit(n)
            def visit_Expr(self,n):
                return None if any(isinstance(x,ast.Name) and x.id=='print_metrics' for x in ast.walk(n)) else n
        expected=Drop().visit(expected)
    if not diagnostics:assert ast.dump(tree,include_attributes=False)==ast.dump(expected,include_attributes=False)
    namespace=dict(fun.__globals__);namespace.update(inspect.getclosurevars(fun).nonlocals)
    exec(compile(tree,fun.__code__.co_filename,'exec'),namespace)
    transformed=namespace[fun.__name__]
    return transformed,dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_source=source,generated_source=ast.unparse(tree),allowed_changes=changes,
        mathematical_update_expressions_changed=False,
        recycling_state_present="next_rate" in assigned,
        nonrecycling_diagnostics="next trial uses configured rate; streak0 is inapplicable sentinel")


def make_batch(fun):
    """Preserve the scalar update, state order, and all decisions in each step."""
    @jax.jit
    def batch(params,state,eps,temp,potential,indices):
        def step(carry,_):
            p,s=carry
            p,s,energy,diag=fun(p,s,eps,temp,potential)
            return (p,s),(energy,diag)
        return jax.lax.scan(step,(params,state),indices)
    return batch


def tree_error(left,right,rtol=2e-10,atol=2e-11):
    import numpy as np
    a=jax.tree.leaves(left);b=jax.tree.leaves(right);assert len(a)==len(b)
    maximum=0.;passed=True
    for x,y in zip(a,b):
        x=np.asarray(x);y=np.asarray(y)
        assert x.shape==y.shape and x.dtype==y.dtype
        if x.dtype.kind in 'biu':
            error=0. if np.array_equal(x,y) else float('inf')
        else:error=float(np.max(np.abs(x-y))) if x.size else 0.
        maximum=max(maximum,error)
        passed &= bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.allclose(x,y,rtol=rtol,atol=atol))
    return dict(passed=passed,max_absolute_error=maximum,rtol=rtol,atol=atol)


def decision_match(a,b):
    import numpy as np
    return all(np.array_equal(np.asarray(a[n]),np.asarray(b[n]))
               for n in ['accepted','reductions','trial_count','streak','rate','next_rate'])
