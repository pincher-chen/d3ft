"""Private trial-objective clone: equivalent parallel gather of coefficients.

No frozen source/module globals are patched. Orbital gradients still use the
original objective. Only the Armijo trial forward callable is replaced.
"""
import ast
import os
import hashlib
import inspect
import types
import numpy as np
import jax.numpy as jnp


def clone_closure(fun,replacements):
    names=fun.__code__.co_freevars
    assert set(replacements)<=set(names),(set(replacements),names)
    def cell(value):return (lambda:value).__closure__[0]
    cells=tuple(cell(replacements[name]) if name in replacements else original
                for name,original in zip(names,fun.__closure__))
    result=types.FunctionType(fun.__code__,fun.__globals__,fun.__name__,fun.__defaults__,cells)
    result.__kwdefaults__=fun.__kwdefaults__
    return result


def lookup_from_mask(mask):
    mask=np.asarray(mask,dtype=bool)
    assert mask.ndim==3
    lookup=np.zeros(mask.size,dtype=np.int32)
    lookup[mask.ravel()]=np.arange(1,int(mask.sum())+1,dtype=np.int32)
    return lookup


def unique_expand(coeff_compact,mask,lookup=None):
    # mask's nonzero coordinates enumerate compact G in C/lexicographic order.
    # Every occupied voxel gathers exactly its original coefficient; all other
    # voxels gather a constant zero sentinel. No arithmetic or summation occurs.
    if lookup is None:lookup=lookup_from_mask(mask)
    coeff_compact=jnp.swapaxes(coeff_compact,-1,-2)
    coeff_shape=coeff_compact.shape[:-1]+mask.shape
    zero=jnp.zeros(coeff_compact.shape[:-1]+(1,),dtype=coeff_compact.dtype)
    packed=jnp.concatenate((zero,coeff_compact),axis=-1)
    return jnp.take(packed,jnp.asarray(lookup),axis=-1,mode='clip').reshape(coeff_shape)


def trial_objective(original):
    loss_cells=inspect.getclosurevars(original).nonlocals
    energy=loss_cells['total_energy']
    energy_cells=inspect.getclosurevars(energy).nonlocals
    make_coeff=energy_cells['make_coeff']
    make_cells=inspect.getclosurevars(make_coeff).nonlocals
    assert set(make_cells)=={'freq_mask'},'Only single-GPU Stiefel expansion is supported'
    source=inspect.getsource(make_coeff).strip()
    assert 'stiefel_coeff(p, freq_mask)' in source,source
    mask=np.array(make_cells['freq_mask'],dtype=bool,copy=True)
    assert mask.ndim==3 and np.any(mask)
    coordinates=np.column_stack(np.nonzero(mask))
    assert np.unique(coordinates,axis=0).shape[0]==len(coordinates)
    assert np.array_equal(coordinates,np.array(sorted(map(tuple,coordinates))))
    mask.setflags(write=False)
    lookup=lookup_from_mask(mask);lookup.setflags(write=False)
    def replacement(p):
        assert p.shape[-2]==len(coordinates)
        return unique_expand(p,mask,lookup)
    new_energy=clone_closure(energy,{'make_coeff':replacement})
    candidate=clone_closure(original,{'total_energy':new_energy})
    report=dict(change='Only trial coefficient expansion uses zero-padded gather instead of complex scatter-set',
        mask_shape=list(mask.shape),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        unique_coordinates=len(coordinates),coordinate_uniqueness_and_order_verified=True,
        lookup_sha256=hashlib.sha256(lookup.tobytes()).hexdigest(),
        zero_voxels_read_constant_zero=True,coefficient_values_are_copied_without_arithmetic=True,
        energy_arithmetic_changed=False,FD_map_changed=False,gradient_path_changed=False,
        frozen_files_modified=False,source_make_coeff=source)
    return candidate,report


def stage_policy(source):
    if source=='d3ft_fd_raw_v2_roundoff':return False
    if source=='d3ft_fd_raw_v2_stable_exact001':return True
    raise ValueError('Unapproved source for phase-scoped gather: '+source)

def candidate_update(fun):
    """Replace exactly one trial call in a diagnostic clone of frozen update."""
    from kernel_tools import transformed_function
    diagnostic,report=transformed_function(fun,diagnostics=True)
    enabled=stage_policy(os.environ['JCTC_FROZEN_SOURCE'])
    report.update(gather_enabled=enabled,phase_policy='Original compiled update in roundoff PH prefix; trial-only gather in stable_exact001 corrected phase',frozen_source=os.environ['JCTC_FROZEN_SOURCE'])
    if not enabled:
        report.update(allowed_changes=['expose existing scalar decisions for audit only'],production_prefix_executable_original=True)
        return diagnostic,report
    tree=ast.parse(report['generated_source'])
    unchanged=ast.parse(report['generated_source'])
    matches=[]
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self,node):
            if (isinstance(node.func,ast.Name) and node.func.id=='loss_and_energy'
                and isinstance(node.args[0],ast.Name) and node.args[0].id=='new'):
                node.func.id='_unique_scatter_trial_objective';matches.append(node.lineno)
            return self.generic_visit(node)
    tree=Rewrite().visit(tree);assert len(matches)==1,matches
    class Undo(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='_unique_scatter_trial_objective':node.id='loss_and_energy'
            return node
    import copy
    assert ast.dump(Undo().visit(copy.deepcopy(tree)),include_attributes=False)==ast.dump(unchanged,include_attributes=False)
    ast.fix_missing_locations(tree)
    original=inspect.getclosurevars(fun).nonlocals['loss_and_energy']
    candidate,scatter_report=trial_objective(original)
    namespace=dict(diagnostic.__globals__)
    namespace['_unique_scatter_trial_objective']=candidate
    exec(compile(tree,fun.__code__.co_filename,'exec'),namespace)
    report.update(generated_source=ast.unparse(tree),trial_scatter=scatter_report,
        allowed_changes=['expose existing scalar decisions','substitute gather trial forward only'],
        callbacks_preserved=True,scalar_update_order_preserved=True)
    return namespace[fun.__name__],report
