# Purpose: Provide throat mass flow rate and nozzle exit conditions (using frozen equilibrium from chamber)
# Authors: Drew Nickel, Thomas A. Scott

from hrap.core import store_x, make_part

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax import cond

from functools import partial

# Solve for exit mach given specific heat ratio and exit ratio
def M_solve(k, ER):
    def get_error(Me, k, ER):
        error = ((k+1)/2)**(-(k+1)/ \
            (2*(k-1)))*(1+(k-1)/2*Me**2)** \
            ((k+1)/(2*(k-1)))/Me- \
            ER
        
        return error
    
    get_derror_dMe = jax.grad(get_error)
    
    def body(val):
        _, Me, k, ER, i = val
        
        error = get_error(Me, k, ER)
        Me -= error / get_derror_dMe(Me, k, ER)
        # error = get_error(Me, k, ER)

        return (error, Me, k, ER, i+1)

    res = jax.lax.while_loop(lambda val: (jnp.abs(val[0]) > 1E-8) & (val[4]<10), body, (1.0, 3.0, k, ER, 0))
    # jax.debug.print('final error {a}, Me={c}, iter={b}', a=get_error(res[1], k, ER), c=res[1], b=res[4])
    
    return res[1]

def d_cd_nozzle(s, x, xmap):
    Pc     = x[xmap['cmbr_P']]          # Chamber pressure
    Pa     = s['Pa']                    # Atmosphere pressure
    k      = x[xmap['cmbr_k']]          # Specific heat ratio
    A_thrt = np.pi/4 * s['noz_thrt']**2 # Nozzle throat area
    
    # Nozzle mass flow rate
    mdot = cond(Pc <= Pa, lambda v: 0.0, lambda v: v, Pc*s['noz_Cd'] * A_thrt/x[xmap['cmbr_cstar']])
    
    # Exit Mach
    Me = M_solve(k, s['noz_ER'])

    # Exit pressure
    Pe = Pc*(1+0.5*(k-1)*Me**2)**(-k/(k-1))
    
    #
    Cf = jnp.sqrt(((2*k**2)/(k-1))*(2/(k+1))**((k+1)/ \
        (k-1))*(1-(Pe/Pc)**((k-1)/k)))+ \
        ((Pe-Pa)*(A_thrt*s['noz_ER']))/ \
        (Pc*A_thrt)
    
    # Thrust
    thrust = s['noz_eff']*Cf*A_thrt*Pc*s['noz_Cd']

    # Store state
    x = store_x(x, xmap, noz_mdot=mdot, noz_Me=Me, noz_Pe=Pe, noz_thrust=thrust)

    return x

def u_cd_nozzle(s, x, xmap):
    # Limit thrust to positive
    x = store_x(x, xmap,
        noz_thrust = jnp.maximum(x[xmap['noz_thrust']], 0.0),
    )
    
    return x

def make_cd_nozzle(**kwargs):
    return make_part(
        # Default static parameters
        s = {
            'Cd':    1.0,
            'eff':   1.0,
            'thrt':  None,
            'ER':    None,
        },
        
        # Default initial variables
        x = {
            'mdot':   0.0,
            'Me':     0.0,
            'Pe':     0.0,
            'thrust': 0.0,
        },
        
        # Required parameters and variables
        req_s = ['thrt', 'ER'],
        req_x = [],
        
        # State name to derivative name for integrated variables
        dx = { },

        # Designation and associated functions
        typename = 'noz',
        fderiv   = d_cd_nozzle,
        fupdate  = u_cd_nozzle,

        # The user-specified s or x entries
        **kwargs,
    )
