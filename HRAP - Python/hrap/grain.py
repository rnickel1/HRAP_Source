# Purpose: Model regression of motor grains
# Authors: Thomas A. Scott

from hrap.core import store_x, make_part

import numpy as np

import jax.numpy as jnp
from jax.lax import cond

from functools import partial

def d_grain_constOF(s, x, xmap, fshape):
    mdot_inj = x[xmap['tnk_mdot_inj']] # TODO: using vent?
    A = x[xmap['grn_A']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    OF = s['grn_OF']
    
    # Current arc length of exposed grain on the cross section
    arc = fshape(d, s, x, xmap)
    
    # Current volume
    V = L * A
    
    # Grain consumption rate (positive)
    mdot = mdot_inj / OF
    mdot = cond(A <= 0.0, lambda val: 0.0, lambda val: val, mdot)
    
    # Rate of volume consumption (positive)
    Vdot = mdot / rho
    
    # Rate of cross section area loss
    Adot = Vdot / L
    
    # Cross sectional area linearization (i.e. volume of thin shell, Adot = arc * ddot)
    ddot = Adot / arc
    
    # Store result
    x = store_x(x, xmap, grn_Adot=-Adot, grn_ddot=ddot, grn_V=V, grn_mdot=mdot, grn_Vdot=Vdot, cmbr_OF=OF)

    return x

def d_grain_shiftOF(s, x, xmap, fshape):
    mdot_inj = x[xmap['tnk_mdot_inj']] # TODO: using vent?
    A = x[xmap['grn_A']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    Reg = s['grn_Reg']
    
    # Current arc length of exposed grain on the cross section
    arc = fshape(d, s, x, xmap)
    # Current volume
    V = L * A
    
    G    = mdot_inj/A;
    ddot = 0.001*Reg[0]*G**Reg[1]*L**Reg[2]
    
    Adot = ddot*arc
    Vdot = Adot*L
    
    mdot = Vdot*rho
    mdot = cond(A <= 0.0, lambda val: 0.0, lambda val: val, mdot)
    
    OF = mdot_inj / mdot
    
    # Store result
    x = store_x(x, xmap, grn_Adot=-Adot, grn_ddot=ddot, grn_V=V, grn_mdot=mdot, grn_Vdot=Vdot, cmbr_OF=OF)
    
    return x

# def d_grain_fit(s, x, xmap, fshape):
#     # Get exposed area along the grain
#     A_burn = arc * L

def u_grain(s, x, xmap):
    x = store_x(x, xmap,
        grn_A = jnp.maximum(x[xmap['grn_A']], 0.0),
        grn_d = jnp.minimum(x[xmap['grn_d']], s['grn_OD']/2) # TODO: shouldnt be necessary
    )
    
    return x

# def i_circle(s, x, xmap):
    
#     return s, x

def make_circle_shape(**kwargs):
    def fcircle(d, s, x, xmap):
        return np.pi * (s['grn_shape_ID'] + 2*d)
    
    def preprs(s, x, xmap):
        OD, ID = s['grn_OD'], s['grn_shape_ID']
        A = np.pi/4 * (OD**2 - ID**2)
        x = x.at[xmap['grn_A']].set(A)
        
        return x
    
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'ID': 0.1,
        },
        x = {
        },
        
        # Required and integrated variables
        req_s = ['ID'],
        req_x = [],
        dx    = { },

        typename = 'shape',

        fshape = fcircle,
        fpreprs = preprs,

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
            **shape['s'],
        },
        x = {
            'A': 0.1,
            'd': 0.0,   # Distance regressed, i.e. increasing during burn
            
            # Calculated variables
            'V': 0.0,
            'Vdot': 0.0,
            'P': 101e3,
            'mdot': 0.0,
            
            **shape['x'],
        },
        
        # Required and integrated variables
        req_s = ['OD', 'L', 'OF'],
        req_x = ['A'],
        dx    = {'A': 'Adot', 'd': 'ddot'},

        # Designation and associated functions
        typename = 'grn',
        fderiv  = partial(d_grain_constOF, fshape=shape['shape_fshape']),
        fupdate = u_grain,
        fpreprs = shape['fpreprs'],

        # The user-specified static and initial dynamic variables
        **kwargs,
    )

def make_shiftOF_grain(shape, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'OD': 0.1,
            'L': 0.1,
            'Reg': FixedList([0.0]*3),
            'rho': 1000.0,
            **shape['s'],
        },
        x = {
            'A': 0.1,
            'd': 0.0,   # Distance regressed, i.e. increasing during burn
            
            # Calculated variables
            'V': 0.0,
            'Vdot': 0.0,
            'P': 101e3,
            'mdot': 0.0,
            
            **shape['x'],
        },
        
        # Required and integrated variables
        req_s = ['OD', 'L', 'OF'],
        req_x = ['A'],
        dx    = {'A': 'Adot', 'd': 'ddot'},

        # Designation and associated functions
        typename = 'grn',
        fderiv  = partial(d_grain_shiftOF, fshape=shape['shape_fshape']),
        fupdate = u_grain,
        fpreprs = shape['fpreprs'],

        # The user-specified static and initial dynamic variables
        **kwargs,
    )
