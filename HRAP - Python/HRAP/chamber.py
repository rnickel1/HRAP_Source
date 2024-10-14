from core import store_x, make_part

from jax.lax import cond

def d_chamber(s, x, dx, smap, xmap):
    # if s['cmbr_V'] == 0:
    #     V = np.pi/4 * x['grn_ID']**2 * s['grn_L']
    # else
    Pt       = x [xmap['cmbr_P']]        # Combustion chamber pressure
    grn_mdot = dx[xmap['grn_m']]         # Rate of grain mass being consumed
    inj_mdot = x [xmap['inj_mdot']]      # Rate of tank propellants being consumed
    noz_mdot = x [xmap['noz_mdot']]      # Rate of mass exiting through nozzle
    m_g      = x [xmap['cmbr_m_g']]      # Mass of gas currently in chamber
    V        = s['cmbr_V'] -  x['grn_V'] # Gas volume in chamber
    dV       = s['cmbr_V'] - dx['grn_V'] # Gas volume in chamber derivative

    # Chamber stored mass derivative
    mdot_g = cond(m_g <= 0.0, 0.0, grn_mdot + inj_mdot - noz_mdot)

    # Chamber pressure derivative
    Pdot = cond(Pt <= s['Pa'], 0.0, Pt*(mdot_g/m_g - dV/V))

    # Store derivatives
    dx = store_x(x, xmap, cmbr_mdot_g=mdot_g, cmbr_P=Pdot)

    return x

def u_chamber(s, x, dx, smap, xmap):
    # Limit stored and gas and pressure to reasonable values
    x = store_x(x, xmap,
        cmbr_m_g = jnp.maximum(x[xmap['cmbr_m_g']],     0.0)
        cmbr_P   = jnp.maximum(x[xmap['cmbr_P']],   s['Pa'])
    )
    
    return x, dx

def make_chamber(**kwargs):
    return make_part(
        # Default static parameters
        s = {
            'V': 0.0,
        },
        
        # Default initial variables
        x = {
            'P':   101e3,
            'm_g': 1E-3,
        },
        
        # Required and integrated variables
        req_s = [],
        req_x = [],
        
        # State name to derivative name for integrated variables
        dx = { 'P': 'Pdot', 'm_g': 'mdot_g' } ,

        # Designation and associated functions
        typename = 'cmbr',
        fderiv  = d_chamber,
        fupdate = u_chamber,

        # The user-specified s or x entries
        **kwargs,
    )
