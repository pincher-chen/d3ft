"""Exact compact occupied k-pool terms, with bounded FFT working storage.

Experiment-local implementation. Fixed integer occupations and one spin only.
No changes to the plane-wave mask, k weights, projectors or quadrature.
"""
import numpy as np
import jax
import jax.numpy as jnp


def make_terms(mask, mesh=None, block=4):
    shape = tuple(mask.shape)
    index = jnp.asarray(np.flatnonzero(np.asarray(mask).reshape(-1)))
    n = int(np.prod(shape))

    def local(q, occ, g, k, pot, volume, coupling):
        if q.shape[0] != 1 or q.shape[-1] % block:
            raise ValueError('Restricted integer-occupied bands must divide the FFT block')
        sphere_g = g.reshape(-1, 3)[index]

        def one_k(carry, data):
            density, kinetic, nonlocal_energy = carry
            state, weight, projector, vector = data
            diag = jnp.sum((sphere_g+vector)**2, axis=-1)/2
            ek = jnp.sum(jnp.real(state*jnp.conj(state))*diag[None, :, None]*weight[:, None, :])
            overlap = jnp.einsum('sgb,amg->sbam', state, jnp.conj(projector))
            en = jnp.real(jnp.einsum('snam,ab,snbm,sn->',
                jnp.conj(overlap), coupling, overlap, weight))/volume
            states = jnp.transpose(state[0], (1, 0)).reshape(-1, block, len(index))
            weights = weight[0].reshape(-1, block)

            def one_block(rho, pair):
                coeff, occupation = pair
                grid = jnp.zeros((block, n), dtype=coeff.dtype).at[:, index].set(coeff)
                wave = jnp.fft.ifftn(grid.reshape((block,)+shape), axes=(-3, -2, -1))
                wave = wave*n/jnp.sqrt(volume)
                addition = jnp.einsum('b...,b->...', jnp.real(wave*jnp.conj(wave)), occupation)
                return rho+addition[None, ...], None

            density, _ = jax.lax.scan(jax.checkpoint(one_block), density, (states, weights))
            return (density, kinetic+ek, nonlocal_energy+en), None

        initial = (jnp.zeros((1,)+shape), jnp.array(0., dtype=jnp.float64),
                   jnp.array(0., dtype=jnp.float64))
        result, _ = jax.lax.scan(jax.checkpoint(one_k), initial,
            (jnp.swapaxes(q, 0, 1), jnp.swapaxes(occ, 0, 1), pot, k))
        return jax.tree_util.tree_map(lambda x: jax.lax.psum(x, 'k'), result) if mesh is not None else result

    if mesh is None:
        return local
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as S
    return shard_map(local, mesh=mesh,
        in_specs=(S(None, 'k', None, None), S(None, 'k', None),
                  S(), S('k', None), S('k', None, None, None), S(), S()),
        out_specs=(S(), S(), S()), check_rep=False)


def parity_check():
    """Compare with independent full FFT formulas and their complex gradients."""
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(7)
    shape = (8, 8, 8); mask = np.zeros(shape, bool); mask.flat[:32] = True
    q = jnp.asarray(rng.normal(size=(1, 2, 32, 4))+1j*rng.normal(size=(1, 2, 32, 4)))
    q = jax.vmap(jax.vmap(lambda x:jnp.linalg.qr(x)[0]))(q)
    occ = jnp.ones((1, 2, 4)); g = jnp.asarray(rng.normal(size=shape+(3,)))
    k = jnp.asarray([[0., 0., 0.], [.3, .2, -.1]])
    projector = jnp.asarray(rng.normal(size=(2, 3, 2, 32))+1j*rng.normal(size=(2, 3, 2, 32)))
    coupling = jnp.asarray([[.2, .1, 0.], [.1, .3, 0.], [0., 0., -.1]])
    volume = jnp.array(100.)
    terms = make_terms(mask, block=2)
    def full(state):
        coefficients = jnp.zeros((1, 2, 4, 512), dtype=state.dtype).at[..., :32].set(jnp.transpose(state, (0, 1, 3, 2)))
        coefficients = coefficients.reshape(1, 2, 4, *shape)
        wave = jnp.fft.ifftn(coefficients, axes=(-3, -2, -1))*512/jnp.sqrt(volume)
        rho = jnp.einsum('skbxyz,skb->sxyz', jnp.real(wave*jnp.conj(wave)), occ)
        diagonal = jnp.sum((g[None, ...]+k[:, None, None, None, :])**2, axis=-1)/2
        ek = jnp.sum(jnp.real(coefficients*jnp.conj(coefficients))*diagonal[None, :, None, ...]*occ[..., None, None, None])
        c = jnp.transpose(state, (0, 1, 3, 2))
        overlaps = jnp.einsum('skbg,kcmg->skbcm', c, jnp.conj(projector))
        hn = jnp.einsum('skigm,gh,skjhm->skij', jnp.conj(overlaps), coupling, overlaps)/volume
        en = jnp.real(jnp.einsum('skii,ski->', hn, occ))
        return rho, ek, en
    def compact(state):
        return terms(state, occ, g, k, projector, volume, coupling)
    old = full(q); new = compact(q)
    errors = [float(jnp.max(jnp.abs(a-b))) for a,b in zip(old,new)]
    # Include a nonlinear density energy; testing energy alone misses the FFT VJP.
    def objective(values):
        rho, ek, en = values
        return ek+en+.037*jnp.sum(rho**2)
    gf = jax.grad(lambda x:objective(full(x)))(q)
    gc = jax.grad(lambda x:objective(compact(x)))(q)
    grad_error = float(jnp.max(jnp.abs(gf-gc)))
    if max(errors)>1e-10 or grad_error>1e-10:
        raise AssertionError((errors, grad_error))
    return dict(passed=True, density_max_error=errors[0], kinetic_error=errors[1],
                nonlocal_error=errors[2], complex_gradient_max_error=grad_error,
                scope='Independent small-grid full FFT versus compact formulas, including nonlinear density-energy gradient; physical checkpoint comparison additionally required.')


if __name__ == '__main__':
    import json
    print(json.dumps(parity_check(), indent=2))
