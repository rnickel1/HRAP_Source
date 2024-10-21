import sys
sys.path.insert(1, '../HRAP/')

import scipy

import core
from tank    import *
from grain   import *
from chamber import *
from nozzle  import *
from sat_nos import *
from units   import _in

"""
numpy
scipy
jax
dearpygui
"""

# Initialization
tank = make_sat_tank(
    get_sat_nos_props,
    V = 0.01,
    inj_CdA=0.01,
    m_ox=5.0, # TODO: init limit
)

shape = make_circle_shape(
)
grn = make_constOF_grain(
    shape,
)

cmbr = make_chamber(
)

noz = make_cd_nozzle(
    thrt = 0.5 * _in, # Throat diameter
    ER = 4.0,         # Exit/throat area ratio
)

chem = scipy.io.loadmat('../../propellant_configs/HTPB.mat')
chem = chem['s'][0][0]
chem_OF = chem[1].ravel()
chem_Pc = chem[0].ravel()
chem_k = chem[2]
chem_M = chem[3]
chem_T = chem[4]
print(chem_OF)
print(chem_Pc)
print(chem_k)

# TODO: Make sure second arg arrays are right transposed
from jax.scipy.interpolate import RegularGridInterpolator
chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k)
chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M)
chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T)

s, x, method = core.make_engine(
    tank, grn, cmbr, noz,
    chem_interp_k=chem_interp_T, chem_interp_M=chem_interp_M, chem_interp_T=chem_interp_T,
    Pa=101e3,
)



# Create the function for firing engines
#   This will be compiled the first time you call it during a run
fire_engine = core.make_integrator(
    core.step_rk4,
    method,
)

# Integrate the engine state
print('run 1')
t, x, xstack = fire_engine(s, x, dt=1E-2, T=10.0)
print('run 2')
t, x, xstack = fire_engine(s, x, dt=1E-2, T=10.0)
print('done')

# Unpack the dynamic engine state
tank, grn, cmbr, noz = unpack_engine(s, x, xstack)



# Visualization
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,7))
axs = np.array(axs).ravel()

# Plot thrust
axs[0].plot(t, noz['thrust'])

# Plot oxidizer flow rate
axs[0].plot(t, tank['mdot_ox'])

# Plot nozzle flow rate

# Open plot
fig.show()

# Save plot
fig.savefig(str(f'./results/nitrous_plastic_plots')+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
