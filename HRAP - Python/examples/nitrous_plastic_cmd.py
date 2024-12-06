import sys
sys.path.insert(1, '../HRAP/')

import scipy
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

import core
from tank    import *
from grain   import *
from chamber import *
from nozzle  import *
from sat_nos import *
from units   import _in, _ft

jax.config.update("jax_enable_x64", True)
# np.random.seed(42)

"""
numpy
scipy
jax
dearpygui
"""

# Initialization
tnk = make_sat_tank(
    get_sat_nos_props,
    V = (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
    inj_CdA= 0.5 * (np.pi/4 * 0.5**2 * _in**2),
    m_ox=14.0, # TODO: init limit
    # m_ox = 3.0,
)
# print('INJ TEST', 0.5 * (np.pi/4 * 0.5**2 * _in**2))

shape = make_circle_shape(
    ID = 2.5 * _in,
)
grn = make_constOF_grain(
    shape,
    OF = 3.0,
    OD = 5.0 * _in,
    L = 4.0 * _ft,
)

cmbr = make_chamber(
)

noz = make_cd_nozzle(
    thrt = 1.5 * _in, # Throat diameter
    ER = 5.0,         # Exit/throat area ratio
)

chem = scipy.io.loadmat('../../propellant_configs/HTPB.mat')
chem = chem['s'][0][0]
chem_OF = chem[1].ravel()
chem_Pc = chem[0].ravel()
chem_k = chem[2]
chem_M = chem[3]
chem_T = chem[4]
# print(chem_OF)
# print(chem_Pc)
# print(chem_k)

# TODO: Make sure second arg arrays are right transposed
from jax.scipy.interpolate import RegularGridInterpolator
chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k, fill_value=1.4)
chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M, fill_value=29.0)
chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T, fill_value=293.0)

s, x, method = core.make_engine(
    tnk, grn, cmbr, noz,
    chem_interp_k=chem_interp_k, chem_interp_M=chem_interp_M, chem_interp_T=chem_interp_T,
    Pa=101e3,
)



# Create the function for firing engines
#   This will be compiled the first time you call it during a run
fire_engine = core.make_integrator(
    # core.step_rk4,
    core.step_fe,
    method,
)

# Integrate the engine state
# T = 1.0
T = 10.0
print('run 1')
t, x, xstack = fire_engine(s, x, dt=1E-3, T=T)
# print('run 2')
# TODO: new x0!
# t, x, xstack = fire_engine(s, x, dt=1E-2, T=T)
print('done')
N_t = xstack.shape[0]
# print(xstack.shape)

# Unpack the dynamic engine state
tnk, grn, cmbr, noz = core.unpack_engine(s, xstack, method)
# print('tnk', tnk.keys())
# print('grn', grn.keys())
# print('cmbr', cmbr.keys())
# print('noz', noz.keys())
print()
print('Post-run arrays:')
for name, obj in (('tnk', tnk), ('grn', grn), ('cmbr', cmbr), ('noz', noz)):
    print(name+':')
    for key, val in obj.items():
        print(key+':', val)
    print()



# Visualization
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
axs = np.array(axs).ravel()

# Plot thrust
axs[0].plot(np.linspace(0.0, T, N_t), noz['thrust'])
axs[0].set_title('Thrust')

# Plot oxidizer flow rate
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_ox'], label='mdot_ox')
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_inj'], label='mdot_inj')
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_vnt'], label='mdot_vnt')
axs[1].legend()

axs[1].set_title('mdot_ox')

axs[2].plot(np.linspace(0.0, T, N_t), tnk['P'] - cmbr['P'])
axs[2].set_title('Ptank - Pchamber')

axs[3].plot(np.linspace(0.0, T, N_t), tnk['T'])
axs[3].set_title('T tank')

axs[4].plot(np.linspace(0.0, T, N_t), tnk['m_ox_liq'])
axs[4].plot(np.linspace(0.0, T, N_t), tnk['m_ox_vap'])
axs[4].set_title('m tank')



# Plot nozzle flow rate

# Open plot
# fig.show()

# Save plot
Path('./results').mkdir(parents=True, exist_ok=True)
fig.savefig(str(f'./results/nitrous_plastic_plots')+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# plt.show()
