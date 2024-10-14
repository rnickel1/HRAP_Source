
# Assume that pressure derivative is historical average
def avg_liq_blowdown(x, dox_dT):
    Pdot = x['tnk_Pdot_sum'] / x['tnk_Pdot_N']
    Tdot = Pdot / dox_dT['Pv']
    
    return Tdot, Pdot

# Evaporation based liquid blowdown that falls back to average pressure drop if condensation is indicated
def liq_blowdow(x, ox, dox_dT);
    # Get saturation density derivatives w.r.t. temperature
    drho_v__dT = dox_dT['rho_v']
    drho_l__dT = dox_dT['rho_l']
    
    # Find evap cooling rate, using total oxidizer mass since we consider thermal equilibrium
    A = (m_liq*drho_l__dT/(rho_l*rho_l) + m_vap*drho_v__dT/(rho_v*rho_v)) / (1/rho_v-1/rho_l)
    B = -mdot_ox/(rho_l/rho_v-1) # TODO: sign?
    C = -ox['Hv'] / ((m_liq+m_vap)*ox['Cp'])
    Tdot = B*C / (1-A*C)
    mdot_evap = Tdot*A+B
    
    return cond(mdot_evap > 0.0, (Tdot, Tdot * dox_dT['Pv']), avg_liq_blowdown(x, dox_dT))

# Assume vapor remains along saturation line, which has been experimentally validated for nitrous oxide
def sat_vap_blowdown(T, m_ox, mdot_ox, ox, dox_dT)
    delta = 1E-4 # FD step
    m_2__m_1 = (m_ox + mdot_ox*delta) / m_ox
    
    def Z_body(args):
        eps, Z_i, Z_1, T_i, T_1, m_2__m_1 = args
        
        T_new = T_1*pow(Z_i/Z_1 * m_2__m_1, 0.3)
        Z_new = get_sat_props(T_new)
        
        # Get error and force convergence
        eps = jnp.abs(Z_i - Z_new)
        Z_new = (Z_i + Z_new) / 2
        
        return eps, Z_new, Z_1, T_new, T_1, m_2__m_1
    
    res = jax.lax.while_loop(jnp.abs(val[0]) < 1E-7, Z_body, (1.0, ox['Z'], ox['Z'], T, T, m_2__m_1))
    T_2 = res[3]
    
    Tdot = (T_2 - T) / delta
    Pdot = Tdot * dox_dT['Pv']

def d_sat_tank(s, x, xmap, get_sat_props):
    m_ox = x[xmap['m_ox']]
    T    = x[xmap['tnk_T']]
    inj_CdA = s['tnk_inj_CdA']
    inj_N   = s['tnk_inj_N']
    vnt_CdA = s['tnk_vnt_CdA']
    tnk_V   = s['tnk_V']
    
    # Find oxidizer thermophysical properties
    ox = get_sat_props(T)
    rho_v, rho_l = ox['rho_v'], ox['rho_l']
    
    # Use analytical derivative to get saturation pressure derivative
    # dP_dT = jax.grad(lambda T: get_sat_props(T)['Pv'])(T)
    dox_dT = jax.grad(get_sat_props)(T)
    
    # Get mass of oxidizer currently in the phases
    m_liq = (tnk_V - (m_ox/rho_v))/ ((1/rho_l)-(1/rho_v))
    m_vap = m_ox - m_liq
    
    Pc = x[xvec['cmbr_P']]
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
    mdot_vnt = cond(s['vnt_S'] == 0, 0.0, (vnt_CdA*x.P_tnk/jnp.sqrt(T))*jnp.sqrt(1.31/(ox['Z']*188.91))*Matm*(1+(0.31)/2*Matm**2)**(-2.31/0.62))
    
    # Get injected vapor or liquid oxidizer mass flow rate
    mdot_inj = cond(m_liq == 0.0,
        (inj_CdA*inj_N*x.P_tnk/jnp.sqrt(T))*jnp.sqrt(1.31/(ox['Z']*188.91))*Mcc*(1+(0.31)/2*Mcc**2)**(-2.31/0.62),
        inj_CdA*inj_N*jnp.sqrt(2*rho_l*dP)
    
    # Total loss rate of oxidizer = base injected rate + vent rate
    mdot_ox = -(mdot_ox + mdot_vnt)
    
    # Add vent flow rate to injector rate if plumbed to chamber
    mdot_inj = cond(s['vnt_S'] ~= 2, mdot_inj, mdot_inj + mdot_vnt)
    
    # Get temperature and pressure rates at various stages of blowdown
    Tdot, Pdot = cond(mdot_ox <= 0.0, (0.0, 0.0),
        cond(m_liq > 0.0,
            liq_blowdow(x, ox, dox_dT),
            sat_vap_blowdown(T, m_ox, mdot_ox, ox, dox_dT)
        )
    
    # Set temperature rate to 0 if outside supported range and not headed back
    
    # Store state and derivatives
    x = store_x(x, xmap, tnk_P=Pt, tnk_Tdot=Tdot, tnk_Pdot=Pdot, tnk_mdot_ox=mdot_ox, tnk_mdot_inj=mdot_inj, tnk_mdot_vnt=mdot_vnt)

def u_sat_tank(s, x, xmap):
    x = store_x(x, xmap,
        # Limit to reasonable values
        m_ox = jnp.maximum(x[xmap['m_ox']], 0.0),
        # Record needed for avg pressure drop blowdown model
        tnk_Pdot_sum = x[xmap['tnk_Pdot_sum']] + x[xmap['tnk_Pdot']],
        tnk_Pdot_N = x[xmap['tnk_Pdot_N']] + 1,
    )

def make_sat_tank(get_sat_props, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'V': 0.0,
            'vnt_CdA': 0.0,
            'inj_CdA': 0.0,
            'inj_N': 1,
        },
        x = {
            'T':   293.0,
            'P':   101e3,
            'm_ox': 1.0,
            'Pdot': 0.0,
            'mdot_inj': 0.0,
            'mdot_vnt': 0.0,
            'tnk_Pdot_sum': 0.0,
            'tnk_Pdot_N': 0,
        },
        
        # Required and integrated variables
        req_s = [ 'V', 'inj_CdA' ],
        req_x = [ 'm_ox' ],
        
        # State name to derivative name for integrated variables
        dx = { 'm_ox': 'mdot_ox', 'T': 'Tdot' },

        # Designation and associated functions
        type = 'cmbr',
        fderiv  = partial(d_sat_tank, get_sat_props=get_sat_props),
        fupdate = u_sat_tank,

        # The user-specified s or x entries
        **kwargs,
    )
