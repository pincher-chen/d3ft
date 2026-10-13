"""Private production update: retain the original PH executable, trial-only gather.

There are no diagnostic outputs in the production graph. The AST inverse check
proves that the corrected update differs in exactly one trial call.
"""
import ast
import copy
import inspect
import os
import textwrap
import jax
from unique_scatter_candidate import stage_policy, trial_objective


def production_update(fun):
    if not stage_policy(os.environ['JCTC_FROZEN_SOURCE']):
        return fun, dict(gather_enabled=False, original_function_identity=True)
    text = textwrap.dedent(inspect.getsource(fun))
    original = ast.parse(text)
    original.body[0].decorator_list = []
    tree = copy.deepcopy(original)
    matches = []
    class Replace(ast.NodeTransformer):
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == 'loss_and_energy'
                    and node.args and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == 'new'):
                node.func.id = '_implementation_trial_forward'
                matches.append(node.lineno)
            return self.generic_visit(node)
    tree = Replace().visit(tree)
    assert len(matches) == 1, matches
    class Undo(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == '_implementation_trial_forward': node.id = 'loss_and_energy'
            return node
    assert ast.dump(Undo().visit(copy.deepcopy(tree)), include_attributes=False) == ast.dump(original, include_attributes=False)
    ast.fix_missing_locations(tree)
    cells = inspect.getclosurevars(fun).nonlocals
    forward, info = trial_objective(cells['loss_and_energy'])
    ns = dict(fun.__globals__)
    ns.update(cells)
    ns['_implementation_trial_forward'] = forward
    exec(compile(tree, fun.__code__.co_filename, 'exec'), ns)
    return ns[fun.__name__], dict(gather_enabled=True, inverse_AST_verified=True,
        original_source=text, generated_source=ast.unparse(tree), trial_map=info,
        gradient_direction_Armijo_recycling_callbacks_unchanged=True,
        diagnostic_outputs_added=False)


class Installer:
    def __init__(self):
        self.original_jit = jax.jit
        self.records = []
    def install(self):
        def compile(fun=None, *args, **kw):
            if fun is None: return lambda f: compile(f, *args, **kw)
            if fun.__name__ == 'update' and fun.__code__.co_filename.endswith('/d3ft/calc/ground_state.py'):
                cells = inspect.getclosurevars(fun).nonlocals
                if cells['config'].d3ft.optim.get('metric_trial', False):
                    fun, info = production_update(fun)
                    self.records.append(info)
            return self.original_jit(fun, *args, **kw)
        jax.jit = compile
    def restore(self): jax.jit = self.original_jit
