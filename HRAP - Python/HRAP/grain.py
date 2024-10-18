def d_grain_constOF(s, x, xmap, fshape):
    mdot_ox = x[xmap['']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    
    
    # Current arc length of exposed grain on the cross section
    arc = fshape(d)
    
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

def make_circle_shape():
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
        dx    = ['A': 'Adot', 'd': 'ddot'],

        # Designation and associated functions
        type = 'grn_shape',
        fderiv  = partial(d_grain_constOF, fshape=fshape),
        fupdate = u_chamber,

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
        dx    = ['A': 'Adot', 'd': 'ddot'],

        # Designation and associated functions
        type = 'cmbr',
        fderiv  = partial(d_grain_constOF, fshape=fshape),
        fupdate = u_chamber,

        # The user-specified static and initial dynamic variables
        **kwargs,
    )
