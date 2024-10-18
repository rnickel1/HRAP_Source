from core import store_x, make_part

import jax.numpy as jnp
from jax.lax import cond

from functools import partial

def d_grain_constOF(s, x, xmap, fshape):
    mdot_ox = x[xmap['']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    
    
    # Current arc length of exposed grain on the cross section
    arc = fshape(d, s, x, xmap)
    
    # Rate of volume consumption (positive)
    Vdot = rho * mdot
    
    # Rate of cross section area loss
    Adot = Vdot / L
    
    # Cross sectional area linearization (i.e. volume of thin shell, Adot = arc * ddot)
    ddot = Adot / arc
    
    # Store result
    dx = store_x(x, xmap, grn_Adot=Adot, grn_ddot=ddot)

    return x

def d_grain_fit(s, x, xmap, fshape):
    # Get exposed area along the grain
    A_burn = arc * L

def u_grain(s, x, xmap):
    x = store_x(x, xmap,
        grn_A = jnp.maximum(x[xmap['grn_A']], 0.0),
        grn_d = jnp.minimum(x[xmap['grn_d']], s['grn_OD']/2)
    )
    
    return x

def i_circle(s, x, xmap):
    
    return s, x

def make_circle_shape(**kwargs):
    def fcircle(d, s, x, xmap):
        return np.pi * (s['grn_shape_ID'] + 2*d)
    
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'ID': 0.1,
            'L': 0.1,
            'OF': 1.0,
            'rho': 1000.0,
        },
        x = {
            'A': 0.1,
            'P': 101e3,
            'd': 0.0,   # Distance regressed, i.e. increasing
        },
        
        # Required and integrated variables
        req_s = ['OD', 'L', 'OF'],
        req_x = ['A'],
        dx    = {'A': 'Adot', 'd': 'ddot'},

        typename = 'grn_shape',

        fshape = fcircle,

        # The user-specified static and initial dynamic variables
        **kwargs,
    )

def make_constOF_grain(shape, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'OD': 0.1,
            'L': 0.1,
            'OF': 1.0,
            'rho': 1000.0,
        },
        x = {
            'A': 0.1,
            'P': 101e3,
            'd': 0.0,   # Distance regressed, i.e. increasing
        },
        
        # Required and integrated variables
        req_s = ['OD', 'L', 'OF'],
        req_x = ['A'],
        dx    = {'A': 'Adot', 'd': 'ddot'},

        # Designation and associated functions
        typename = 'cmbr',
        fderiv  = partial(d_grain_constOF, fshape=shape['fshape']),
        fupdate = u_grain,

        # The user-specified static and initial dynamic variables
        **kwargs,
    )
