
# Solve for exit mach given specific heat ratio and exit ratio
def M_solve(k, ER):
    def body(val):
        _, Me, k, ER = val
        error = ((k+1)/2)**(-(k+1)/ \
            (2*(k-1)))*(1+(k-1)/2*Me**2)** \
            ((k+1)/(2*(k-1)))/Me- \
            ER

        return (error, Me, k, ER)

    res = jax.lax.while_loop(jnp.abs(val[0]) < 1E-8, body, (1.0, 3.0, k, ER))
    
    return res[1]

def d_cd_nozzle(s, x, dx, smap, xmap):
    Pt     = x[xmap['cmbr_P']]          # Chamber pressure
    Pa     = s['Pa']                    # Atmosphere pressure
    k      = x[xmap['cmbr_k']]          # Specific heat ratio
    A_thrt = np.pi/4 * s['noz_thrt']**2 # Nozzle throat area
    
    # Nozzle mass flow rate
    mdot = cond(Pt <= Pa, 0.0, Pt*s['noz_Cd'] * A_thrt/s['cstar'])
    
    # Exit Mach
    Me = M_solve(k, s['noz_ER'])

    # Exit pressure
    Pe = Pt*(1+0.5*(k-1)*Me**2)**(-k/(k-1))
    
    #
    Cf = jnp.sqrt(((2*k**2)/(k-1))*(2/(k+1))**((k+1)/ \
        (k-1))*(1-(Pe/Pt)**((k-1)/k)))+ \
        ((Pe-Pa)*(A_thrt*s['noz_ER']))/ \
        (Pt*A_thrt)
    
    # Thrust
    thrust = s['noz_eff']*Cf*A_thrt*Pt*s['noz_Cd']

    # Store everything
    x = store_x(x, xmap, noz_mdot=mdot, noz_Me=Me, noz_Pe=Pe, noz_thrust=thrust)

    return x, dx

def make_cd_nozzle(**kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'Cd':    1.0,
            'eff':   1.0,
            'cstar': 1.0,
            'thrt':  None,
            'ER':    None,
        },
        x = {
            'mdot':   0.0,
            'Me':     0.0,
            'Pe':     0.0,
            'thrust': 0.0,
        },
        
        # Required and integrated variables
        req_s = ['thrt', 'ER'],
        req_x = [],
        dx    = [],

        # Designation and associated functions
        type = 'noz',
        fderiv  = d_cd_nozzle,
        fupdate = None,

        # The user-specified static and initial dynamic variables
        **kwargs,
    )
