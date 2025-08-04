# Purpose: Provide chamber dynamics from tabulated chemical equilibrum properties
# Authors: Drew Nickel, Thomas A. Scott

from hrap.core import store_x, make_part

import jax
import jax.numpy as jnp
from jax.lax import cond

def d_chamber(s, x, xmap):
    Pc       = x[xmap['cmbr_P']]        # Combustion chamber pressure
    grn_mdot = x[xmap['grn_mdot']]         # Rate of grain mass being consumed
    inj_mdot = x[xmap['tnk_mdot_inj']]      # Rate of tank propellants being consumed
    noz_mdot = x[xmap['noz_mdot']]      # Rate of mass exiting through nozzle
    m_g      = x[xmap['cmbr_m_g']]      # Mass of gas currently in chamber
    V        = x[xmap['cmbr_V0']] - x[xmap['grn_V']] # Gas volume in chamber
    dV       = x[xmap['grn_Vdot']] # Gas volume in chamber derivative
    OF       = x[xmap['cmbr_OF']] # O/F ratio, set by grain file

    # Chamber stored mass derivative
    mdot_g = grn_mdot + inj_mdot - noz_mdot
    mdot_g = cond((m_g <= 0.0) & (mdot_g < 0.0), lambda val: 0.0, lambda val: val, mdot_g)

    # Chamber pressure derivative
    Pdot = Pc*(mdot_g/m_g - dV/V)
    Pdot = cond(((Pc <= s['Pa']) & (Pdot < 0.0)) | (m_g <= 0.0), lambda val: 0.0, lambda val: val, Pdot)
    
    # Get chamber properties and update cstar
    # interp_point = jnp.array([[OF, Pc]])
    # TODO: Need an error if out of bounds?
    ig = s['chem_interp_k'].grid # interp grid
    ip = jnp.array([[OF, Pc]]) # interp point
    for i in range(2):
        ip = ip.at[:,i].set(jnp.minimum(jnp.maximum(ip[:,i], ig[i][0]), ig[i][-1]))
    k = s['chem_interp_k'](ip)[0]
    M = s['chem_interp_M'](ip)[0]
    T = s['chem_interp_T'](ip)[0]
    # jax.debug.print('cmbr, k={a}, M={b}, T={c} from OF={d}, Pc={e}', a=k, b=M, c=T, d=OF, e=Pc)
    
    R = 8314.5 / M
    rho = Pc/(R * T)
    cstar = s['cmbr_cstar_eff']*jnp.sqrt((R*T)/(k*(2/(k+1))**((k+1)/(k-1))))
    
    # Store derivatives
    x = store_x(x, xmap, cmbr_mdot_g=mdot_g, cmbr_Pdot=Pdot, cmbr_k=k, cmbr_T=T, cmbr_cstar=cstar)

    return x

# Preprocessing
def p_chamber(s, x0, xmap):
    V0 = x0[xmap['cmbr_V0']]
    
    V0 = cond(V0 == 0.0, lambda Va, Vb: Vb, lambda Va, Vb: Va,   V0, s['grn_L']*(jnp.pi/4*s['grn_OD']**2))
    m_g0 = (101e3*29/8314/294) * V0 # p = rho R T

    x0 = store_x(x0, xmap, cmbr_V0=V0, cmbr_m_g=m_g0)

    return x0

def u_chamber(s, x, xmap):
    # Limit stored and gas and pressure to reasonable values
    x = store_x(x, xmap,
        cmbr_m_g = jnp.maximum(x[xmap['cmbr_m_g']],     0.0),
        cmbr_P   = jnp.maximum(x[xmap['cmbr_P']],   s['Pa']),
    )
    
    return x

def make_chamber(**kwargs):
    return make_part(
        # Default static parameters
        s = {
            'cstar_eff': 1.0,
        },
        
        # Default initial variables
        x = {
            'V0': 0.0, # Empty volume, doesn't change after initialization

            'P':   101e3,
            'm_g': 1E-3,
            
            # Calculated variables
            'k': 0.0,
            'T': 0.0,
            'cstar': 0.0,
            'OF': 0.0, # Handled by grain file
        },
        
        # Required and integrated variables
        req_s = [],
        req_x = [],
        
        # State name to derivative name for integrated variables
        dx = { 'P': 'Pdot', 'm_g': 'mdot_g' } ,

        # Designation and associated functions
        typename = 'cmbr',
        fpreprs = p_chamber,
        fderiv  = d_chamber,
        fupdate = u_chamber,

        # The user-specified s or x entries
        **kwargs,
    )
