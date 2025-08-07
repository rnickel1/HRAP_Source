# Purpose: Model the blowdown of a saturated fluid with a second gas that will not dissolve or condense in a simple tank
# Authors: Thomas A. Scott

# For sat, fugacity should match pure
# Based on derivatives or can we 

from hrap.core import store_x, make_part, StaticVar

import jax
import jax.numpy as jnp
from jax.lax import cond

from functools import partial

# Assume that pressure derivative is historical average
def avg_liq_blowdown(x, xmap, dP_dT):
    Pdot = x[xmap['tnk_Pdot_sum']] / x[xmap['tnk_Pdot_N']]
    Tdot = Pdot / dP_dT
    
    return Tdot, Pdot

# Evaporation based liquid blowdown that falls back to average pressure drop if condensation is indicated
def liq_blowdow(mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT):
    m_liq, m_vap = x[xmap['tnk_m_ox_liq']], x[xmap['tnk_m_ox_vap']]
    rho_v, rho_l = ox['rho_v'], ox['rho_l']
    
    # Find evap cooling rate, using total oxidizer mass since we consider thermal equilibrium
    A = (m_liq*drho_l__dT/(rho_l**2) + m_vap*drho_v__dT/(rho_v**2)) / (1/rho_v-1/rho_l)
    B = -mdot_ox/(rho_l/rho_v-1) # TODO: sign?
    C = -ox['Hv'] / ((m_liq+m_vap)*ox['Cp'])
    Tdot = B*C / (1-A*C)
    # jax.debug.print("TANK Debug {x} {y} {z} {w} {a} {b}", x=mdot_ox, y=Tdot, z=m_liq, w=m_vap, a=ox['Hv'], b=ox['Cp'])
    # jax.debug.print("TANK Debug {x} {y} {c} {d} {e}", x=mdot_ox, y=Tdot, c=drho_v__dT, d=drho_l__dT, e=dP_dT)
    mdot_evap = Tdot*A+B
    
    Tdot, Pdot = cond(
        mdot_evap > 0.0,
        lambda x, xmap, Tdot, dP_dT:
            (Tdot, Tdot * dP_dT),
        lambda x, xmap, Tdot, dP_dT:
            avg_liq_blowdown(x, xmap, dP_dT),
        x, xmap, Tdot, dP_dT
    )
    
    return Tdot, Pdot

# Assume vapor remains along saturation line, which has been experimentally validated for nitrous oxide
def sat_vap_blowdown(T, m_ox, mdot_ox, ox, dP_dT, get_sat_props):
    delta = 1E-7 # FD step
    m_2__m_1 = (m_ox + mdot_ox*delta) / m_ox
    
    # TODO: precompute high order curve-fit using complex step results
    def Z_body(args):
        eps, Z_i, Z_1, T_i, T_1, m_2__m_1 = args
        
        T_new = T_1*pow(Z_i/Z_1 * m_2__m_1, 0.3)
        Z_new = get_sat_props(T_new)['Z']
        
        # Get error and force convergence
        eps = jnp.abs(Z_i - Z_new)
        Z_new = (Z_i + Z_new) / 2
        # jax.debug.print("TANKi Debug {x} {y} {Z}", x=Z_new, y=Z_i, Z=eps)
        
        return eps, Z_new, Z_1, T_new, T_1, m_2__m_1
    
    res = jax.lax.while_loop(lambda val: jnp.abs(val[0]) > 1E-9, Z_body, (1.0, ox['Z'], ox['Z'], T, T, m_2__m_1))
    # jax.debug.print('{a}', a=res[0])
    T_2 = res[3]
    
    Tdot = (T_2 - T) / delta
    Pdot = Tdot * dP_dT
    # jax.debug.print("TANK Debug {x} {y} {c} {d} {e}", x=Tdot, y=Pdot, c=m_ox, d=mdot_ox, e=dP_dT)
    
    return Tdot, Pdot

def d_sat_tank(s, x, xmap, get_sat_props):
    _sat_vap_blowdown = partial(sat_vap_blowdown, get_sat_props=get_sat_props)
    
    m_ox = x[xmap['tnk_m_ox']]
    T    = x[xmap['tnk_T']]
    inj_CdA = s['tnk_inj_CdA']
    inj_N   = s['tnk_inj_N']
    vnt_CdA = s['tnk_vnt_CdA']
    tnk_V   = s['tnk_V']
    
    # Find oxidizer thermophysical properties
    ox = get_sat_props(T)
    rho_v, rho_l = ox['rho_v'], ox['rho_l']
    
    # Use analytical derivative to get saturation pressure derivative
    dP_dT = jax.grad(lambda T: get_sat_props(T)['Pv'])(T)
    # Get saturation density derivatives w.r.t. temperature
    drho_v__dT = jax.grad(lambda T: get_sat_props(T)['rho_v'])(T)
    drho_l__dT = jax.grad(lambda T: get_sat_props(T)['rho_l'])(T)
    # dox_dT = jax.grad(get_sat_props)(T)
    
    # Get mass of oxidizer currently in the phases
    m_liq = jnp.maximum((tnk_V - (m_ox/rho_v))/ ((1/rho_l)-(1/rho_v)), 0.0)
    m_vap = m_ox - m_liq
    x = store_x(x, xmap, tnk_m_ox_liq=m_liq, tnk_m_ox_vap=m_vap) # Store here because needed in blowdown funcs
    x = store_x(x, xmap, tnk_rho_ox_liq=rho_l, tnk_rho_ox_vap=rho_v)
    
    Pc = x[xmap['cmbr_P']]
    Pt = ox['Pv']
    Pa = s['Pa']
    
    # Find oxidizer mass flow rate
    
    dP = Pt - Pc
    
    Mcc  = jnp.sqrt(ox['Z']*1.31*188.91*T*(Pc/Pt)**(0.31/1.31))
    Matm = jnp.sqrt(ox['Z']*1.31*188.91*T*(Pa/Pt)**(0.31/1.31))
    
    Mcc  = jnp.minimum(Mcc,  1)
    Matm = jnp.minimum(Matm, 1)
    dP   = jnp.maximum(dP,   0)
    
    # Get vented vapor mass flow rate
    mdot_vnt = cond(
        s['tnk_vnt_S'] == 0,
        lambda *args:
            0.0,
        lambda vnt_CdA, T, Z, Matm:
            (vnt_CdA*Pt/jnp.sqrt(T))*jnp.sqrt(1.31/(Z*188.91))*Matm*(1+(0.31)/2*Matm**2)**(-2.31/0.62),
        vnt_CdA, T, ox['Z'], Matm
    )
    
    # Select injector models at compile time
    def inj_liq_model(m_vap, inj_NCdA, T, Z, Mcc, rho_l, rho_v, dP):
        if s['tnk_inj_liq_model'] == 'Incompressible':
            return inj_NCdA*jnp.sqrt(2*rho_l*dP)
    def inj_vap_model(inj_NCdA, T, Z, Mcc, dP, rho_v):
        if s['tnk_inj_vap_model'] == 'Incompressible':
            return inj_NCdA*jnp.sqrt(2*rho_v*dP)
        elif s['tnk_inj_vap_model'] == 'Real Gas':
            return (inj_NCdA*Pt/jnp.sqrt(T))*jnp.sqrt(1.31/(Z*188.91))*Mcc*(1+(0.31)/2*Mcc**2)**(-2.31/0.62)

    # Get injected vapor or liquid oxidizer mass flow rate
    mdot_inj = cond(
        m_liq <= 1E-3,
        lambda m_vap, inj_NCdA, T, Z, Mcc, rho_l, rho_v, dP: cond(
            m_vap <= 0.0,
            lambda *vargs: 0.0,
            lambda *vargs: inj_vap_model(*vargs),
            inj_NCdA, T, ox['Z'], Mcc, dP, rho_v
        ),
        lambda *vargs: inj_liq_model(*vargs),
        m_vap, inj_N*inj_CdA, T, ox['Z'], Mcc, rho_l, rho_v, dP
    )
    # jax.debug.print("INJ Debug {a} {b} {x} {y} {c} {d} {e} {f}", a=m_liq, b=m_vap, x=Pt, y=T, c=ox['Z'], d=Mcc, e=mdot_inj, f=(inj_CdA*inj_N*Pt/jnp.sqrt(T))*jnp.sqrt(1.31/(ox['Z']*188.91))*Mcc*(1+(0.31)/2*Mcc**2)**(-2.31/0.62))
    
    # Total loss rate of oxidizer = base injected rate + vent rate
    mdot_ox = -(mdot_inj + mdot_vnt)
    
    # Add vent flow rate to injector rate if plumbed to chamber
    mdot_inj = cond(
        s['tnk_vnt_S'] != 2,
        lambda mdot_inj, mdot_vnt:
            mdot_inj,
        lambda mdot_inj, mdot_vnt:
            mdot_inj + mdot_vnt,
        mdot_inj, mdot_vnt
    )
    
    # Get temperature and pressure rates at various stages of blowdown
    Tdot, Pdot = cond(m_ox <= 0.0, lambda *args: (0.0, 0.0),
        lambda *args: cond(m_liq > 0.0,
            lambda T, m_ox, mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT:
                liq_blowdow(mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT),
            lambda T, m_ox, mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT:
                _sat_vap_blowdown(T, m_ox, mdot_ox, ox, dP_dT),
            *args
        ),
        T, m_ox, mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT
    )
    
    # Set temperature rate to 0 if outside supported range and not headed back
    
    # Store state and derivatives
    x = store_x(x, xmap, tnk_P=Pt, tnk_Tdot=Tdot, tnk_Pdot=Pdot, tnk_mdot_ox=mdot_ox, tnk_mdot_inj=mdot_inj, tnk_mdot_vnt=mdot_vnt)
    
    return x

# TODO: remove
def p_sat_tank(s, x0, xmap):

    # x0 = store_x(x0, xmap, cmbr_V=V)

    return x0

def u_sat_tank(s, x, xmap):
    x = store_x(x, xmap,
        # Limit to reasonable values
        tnk_m_ox = jnp.maximum(x[xmap['tnk_m_ox']], 0.0),
        # Record needed for avg pressure drop blowdown model
        tnk_Pdot_sum = x[xmap['tnk_Pdot_sum']] + x[xmap['tnk_Pdot']],
        tnk_Pdot_N = x[xmap['tnk_Pdot_N']] + 1,
    )

    return x

def make_super_tank(get_sat_props, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'V': 0.0,
            'vnt_S': 0,
            'vnt_CdA': 0.0,
            'inj_CdA': 0.0,
            'inj_N': 1,
            'inj_vap_model': StaticVar('Incompressible'),
            'inj_liq_model': StaticVar('Incompressible'),
        },
        x = {
            'T':   293.0,
            'm_ox': 1.0,
            
            # Calculated variables
            'Pdot': 0.0,
            'mdot_inj': 0.0,
            'mdot_vnt': 0.0,
            'P':   101e3,
            'm_ox_liq': 0.0,
            'm_ox_vap': 0.0,
            'rho_ox_liq': 0.0,
            'rho_ox_vap': 0.0,
            'Pdot_sum': 0.0,
            'Pdot_N': 0,
        },
        
        # Required and integrated variables
        req_s = [ 'V', 'inj_CdA' ],
        req_x = [ 'm_ox' ],
        
        # State name to derivative name for integrated variables
        dx = { 'm_ox': 'mdot_ox', 'T': 'Tdot' },

        # Designation and associated functions
        typename = 'tnk',
        fpreprs = p_sat_tank,
        fderiv  = partial(d_sat_tank, get_sat_props=get_sat_props),
        fupdate = u_sat_tank,

        # The user-specified s or x entries
        **kwargs,
    )
