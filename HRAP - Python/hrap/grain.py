# Purpose: Model regression of motor grains
# Authors: Thomas A. Scott

import numpy as np
import jax.numpy as jnp
from jax.lax import cond
from functools import partial, reduce
from skimage import measure
import interpax
from hrap.core import store_x, make_part
import hrap.sdf as sdf
import matplotlib.cm as cm
from scipy import ndimage
import cv2
import math


def d_grain_constOF(s, x, xmap, d2a):
    mdot_inj = x[xmap['tnk_mdot_inj']] # TODO: using vent?
    A = x[xmap['grn_A']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    OF = s['grn_OF']
    
    # Current arc length of exposed grain on the cross section
    arc = d2a(d, s, x, xmap)
    
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

def d_grain_shiftOF(s, x, xmap, d2a):
    mdot_inj = x[xmap['tnk_mdot_inj']] # TODO: using vent?
    A = x[xmap['grn_A']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    Reg = s['grn_Reg']
    
    # Current arc length of exposed grain on the cross section
    arc = d2a(d, s, x, xmap)
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

def u_grain(s, x, xmap):
    x = store_x(x, xmap,
        grn_A = jnp.maximum(x[xmap['grn_A']], 0.0),
        grn_d = jnp.minimum(x[xmap['grn_d']], s['grn_OD']/2) # TODO: shouldnt be necessary
    )
    
    return x

def make_star_vertices(grn_ID, grn_TD, N_tip):
    grn_r = np.array([grn_ID/2, grn_TD/2]*N_tip)
    grn_t = np.linspace(0.0, 2*np.pi, 2*N_tip, endpoint=False)
    grn_xy = np.stack([grn_r*np.cos(grn_t), grn_r*np.sin(grn_t)], axis=1)
    return grn_xy

def bake_d2a(grn_OD, ports_xy, Nx, Nd):
    """Use description of ports to precompute the regression behavor of the grain

    Args:
      grn_OD: outer diameter of the grain
      ports_xy: list of or single array of shape (N,2)
        each array can have a different leading dimension,
        each describes a closed loop representing an initial port,
        each cannot intersect itself or others upon specification.
      Nx: spatial resolution
      Nd: regression resolution
    Returns:
      initial cross section area, tabulated regression distance (d2a input), tabulated exposed arc length (d2a output), list of contous for plotting
    """
    if type(ports_xy) not in [list, tuple]: ports_xy = [ports_xy]

    sdf_x, sdf_y = np.meshgrid(*[((np.arange(Nx)+0.5)/Nx - 0.5)*grn_OD]*2,indexing='ij')
    sdf_xy = np.stack([sdf_x.ravel(), sdf_y.ravel()], axis=1)
    sdf_v = reduce(lambda sdf1,sdf2: np.minimum(sdf1,sdf2), [sdf.sd_poly(port_xy, sdf_xy) for port_xy in ports_xy])
    is_outside = sdf_xy[:,0]**2+sdf_xy[:,1]**2 > (grn_OD/2)**2
    R = np.max(sdf_v[~is_outside]) # Distance from initial grain to outer wall
    sdf_v = sdf_v.reshape((Nx,Nx)) # Unflatten
    A0 = np.pi/4*grn_OD**2 - reduce(lambda v1,v2: v1+v2, [sdf.area_poly(port_xy) for port_xy in ports_xy])

    grn_d2a = np.zeros(Nd)
    grn_d = np.arange(Nd)/(Nd-1)*R
    # grn_xy1 = np.roll(grn_xy,1,axis=0) # Previoius vertices
    # grn_d2a[0] = np.sum(np.sqrt((grn_xy[:,0]-grn_xy1[:,0])**2 + (grn_xy[:,1]-grn_xy1[:,1])**2))
    grn_d2a[0] = reduce(lambda v1,v2: v1+v2, [np.sum(np.sqrt(np.sum((port_xy-np.roll(port_xy,1,axis=0))**2,axis=-1))) for port_xy in ports_xy])
    # Use original specification for initial grain state
    all_contours = [np.append(port_xy, [port_xy[0,:]], axis=0) for port_xy in ports_xy]
    # For each remaining regression query point, use sdf detection
    for i in range(1, Nd):
        contours = measure.find_contours(sdf_v, i/(Nd-1)*R)
        for contour in contours:
            contour = ((contour+0.5)/Nx - 0.5)*grn_OD
            is_inside = contour[:,0]**2+contour[:,1]**2 <= (grn_OD/2)**2
            contour[~is_inside,:] = np.nan
            all_contours.append(contour)
            # Contours repeat the first point if it is a closed loop so no special treatment needed
            a_segs = np.sqrt((contour[1:,0]-contour[:-1,0])**2 + (contour[1:,1]-contour[:-1,1])**2)
            # Not counting NaN entries handles only counting remaining line segments
            grn_d2a[i] += np.sum(a_segs[np.isfinite(a_segs)])
    return A0, grn_d, grn_d2a, all_contours

def make_arbitrary_shape(d2a_curve, **kwargs):
    def arbitrary_d2a(d, s, x, xmap, curve):
        return curve(d) #np.pi * (s['grn_shape_ID'] + 2*d)
    
    def preprs(s, x, xmap):
        x = x.at[xmap['grn_A']].set(s['grn_shape_A0'])
        
        return x
    
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'A0': 0.1,
        },
        x = {
        },
        
        # Required and integrated variables
        req_s = ['A0'],
        req_x = [],
        dx    = { },

        typename = 'shape',

        d2a = partial(arbitrary_d2a, curve=d2a_curve),
        fpreprs = preprs,
        
        # The user-specified static and initial dynamic variables
        **kwargs,
    )

# TODO: Helper for equivalent inner and outer diameter?

def make_circle_shape(**kwargs):
    def circle_d2a(d, s, x, xmap):
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

        d2a = circle_d2a,
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
        fderiv  = partial(d_grain_constOF, d2a=shape['shape_d2a']),
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
            'Reg': jnp.zeros(3), # Regression coefficient (mm/s), regression exponent, length exponent
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
        fderiv  = partial(d_grain_shiftOF, d2a=shape['shape_d2a']),
        fupdate = u_grain,
        fpreprs = shape['fpreprs'],

        # The user-specified static and initial dynamic variables
        **kwargs,
    )

def bake_arbitrary_d2a(path, grain_diameter, n_step=100, n_visu=30):
    
    # 1. Simulation setup
    print(f"Baking file at: {path}")
    print(f"grain_diameter: {grain_diameter}")
    print(f"Settings: steps={n_step}, level curves={n_visu}")
    
    # Time vector
    distances = np.linspace(0, grain_diameter/2, n_step)


    # 2. Image processing

    # Image import
    img = cv2.imread(path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = ndimage.gaussian_filter(gray, sigma=0.6)
    _, mask_grain = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)



    # Empty image mask
    h, w = mask_grain.shape
    filled_grain = np.zeros((h, w), dtype=np.uint8)

    # Finding grain contours
    contours_all, _ = cv2.findContours(mask_grain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_ext = max(contours_all, key=cv2.contourArea)
    ((cx_pix, cy_pix), radius_pix) = cv2.minEnclosingCircle(c_ext)


    # Calculating scaling factor
    resolution = grain_diameter / (radius_pix * 2)
    print(f"Resolution detected : {resolution:.6f} m/pixel")


    # Drawing the interior of the grain diameter as white
    cv2.drawContours(filled_grain, [c_ext], -1, 255, thickness=cv2.FILLED)

    # Boolean to substract the grain from the white zone
    # Now only the hole has a value of 0
    hole = cv2.bitwise_and(filled_grain, cv2.bitwise_not(mask_grain))

    ## Cut the image to isolate and center the grain
    w,h = math.ceil(2*radius_pix),math.ceil(2*radius_pix)
    centered_hole = np.zeros((w,h), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            centered_hole[i,j] = hole[i+int(cy_pix-radius_pix),j+int(cx_pix-radius_pix)]
            
            
    # Uncomment if you want to check if your grain port was detected properly
    #plt.imshow(centered_hole),plt.show()

    # Distance map (from the center hole)
    dist_map = ndimage.distance_transform_edt(centered_hole == 0) * resolution
    dist_map = ndimage.gaussian_filter(dist_map, sigma=0.6)


    # Calculate the area of the initial hole
    hole_pixel_count = cv2.countNonZero(centered_hole)
    grn_A0 = np.pi*(grain_diameter**2/4) - hole_pixel_count * (resolution**2)



    # 4. Calculations

    perimeters = []
    contour = []
    ext_radius_square_m = (grain_diameter / 2)**2

    print(f"Starting simulation : Grain diameter = {grain_diameter} m")

    for i, d_reg in enumerate(distances):
        
        
        # Finding contours at this time
        if i == 0:
            contours = measure.find_contours(dist_map, 1e-9) 
        else:
            contours = measure.find_contours(dist_map, d_reg)


        total_p_t = 0

        for contour_pix in contours:
            # Converting pixel contour to mm contour
            c_y_m = contour_pix[:, 0] * resolution
            c_x_m = contour_pix[:, 1] * resolution

            # Hiding contours outside grain_diameter
            # Distance to grain center
            dist_sq_to_center = (c_x_m - grain_diameter/2)**2 + (c_y_m - grain_diameter/2)**2

            # Boolean mask : true if contour outside grain_diameter
            is_outside = dist_sq_to_center >= ext_radius_square_m

            # Copy for plot
            plot_x = c_x_m.copy()
            plot_y = c_y_m.copy()


            # Hiding curves outside grain_diameter by replacing them with NaN
            plot_x[is_outside] = np.nan
            plot_y[is_outside] = np.nan

            
            if (i % math.ceil(n_step/n_visu) == 0):
                # Plotting curve
                contour.append([plot_x,plot_y,i])

            # Perimeter calculations
            # np.diff calculates [x[1]-x[0], x[2]-x[1], ...]
            dx = np.diff(c_x_m)
            dy = np.diff(c_y_m)
            segment_lengths = np.sqrt(dx**2 + dy**2)


            # Only counts segments that are inside grain_diameter
            valid_segment_mask = ~is_outside[:-1]


            # Add all valid segments length
            total_p_t += np.sum(segment_lengths[valid_segment_mask])


        perimeters.append(total_p_t)
        if (total_p_t==0):
            break

    distances = [distances[k] for k in range(len(perimeters))]

    return (distances,perimeters,contour,grn_A0)