def d_grain(s, x, dx):
    

def u_grain(s, x, dx):

def make_grain(fshape, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
        },
        x = {
            'V':   0.0,
            'P':   101e3,
            'm_g': 1E-3,
        },
        
        # Required and integrated variables
        req_s = [],
        req_x = [],
        dx    = ['P', 'm_g'],

        # Designation and associated functions
        type = 'cmbr',
        fderiv  = partial(d_grain, fshape=fshape),
        fupdate = u_chamber,

        # The user-specified static and initial dynamic variables
        **kwargs,
    )
