# Purpose: Wrapper for coolprop to provide compilable property tables
# Authors: Thomas A. Scott

import numpy as np
import jax.numpy as jnp
import CoolProp.CoolProp as CP
import interpax

def bake_sat_props(fluid, T_eval):
    _Pv, _rho_l, _rho_v, _Hv, _Cp, _Z = [np.zeros_like(T_eval) for i in range(6)]
    for i, T in enumerate(T_eval):
        _Pv   [i] = CP.PropsSI('P', 'T', T, 'Q', 0, fluid)
        _rho_l[i] = CP.PropsSI('D', 'T', T, 'Q', 0, fluid)
        _rho_v[i] = CP.PropsSI('D', 'T', T, 'Q', 1, fluid)
        _Z    [i] = CP.PropsSI('Z', 'T', T, 'Q', 1, fluid)
        _Cp   [i] = CP.PropsSI('CPMASS', 'T', T, 'Q', 1, fluid)
        _Hv   [i] = CP.PropsSI('H', 'T', T, 'Q', 1, fluid) - CP.PropsSI('H', 'T', T, 'Q', 0, fluid)
    # print(_Pv)
    # print(_Hv)
    
    # TODO: enable extrap? gives NaNs when off
    # Construct monotomic cubic splines to interpolate
    Pv, rho_l, rho_v, Hv, Cp, Z = [interpax.PchipInterpolator(T_eval, props) for props in [_Pv, _rho_l, _rho_v, _Hv, _Cp, _Z]]
    
    # Staticly supply interpolators to a new sat props function
    def get_my_sat_props(T, Pv=Pv, rho_l=rho_l, rho_v=rho_v, Hv=Hv, Cp=Cp, Z=Z):
        return { 'Pv': Pv(T), 'rho_l': rho_l(T), 'rho_v': rho_v(T), 'Hv': Hv(T), 'Cp': Cp(T), 'Z': Z(T) }

    return get_my_sat_props
