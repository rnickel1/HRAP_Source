# Purpose: Demonstrate API for arbitrary saturated fluids and non-circular ports with a LOX plastisol hybrid star grain example
# Authors: Thomas A. Scott, Roman Dickerson

import scipy
import numpy as np
from pathlib import Path
from importlib.resources import files as imp_files

import matplotlib.pyplot as plt

import hrap.core as core
import hrap.chem as chem
import hrap.fluid as fluid
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.units   import _in, _ft, _lbf, _atm

jax.config.update("jax_enable_x64", True)
hrap_root = Path(imp_files('hrap'))
file_prefix = 'lox_plastisol_star'



print('Building combustion chemistry table...')
plastisol = chem.make_basic_reactant(
    formula = 'Plastisol-362',
    composition = { 'C': 7.200, 'H': 10.82, 'O': 1.14, 'Cl': 0.669 },
    M = 140.86, # kg/kmol
    T0 = 298.15, # K
    h0 = -2.6535755e7, # J/kmol
)
comb = chem.ChemSolver([hrap_root/'thermo.dat', plastisol])
chem_Pc, chem_OF = np.linspace(10*_atm, 50*_atm, 10), np.linspace(1.0, 10.0, 20)
chem_k, chem_M, chem_T = [np.zeros((chem_Pc.size, chem_OF.size)) for i in range(3)]
ox, fu_1, fu_2 = 'O2(L)', 'Plastisol-362', 'AL(cr)'
mfrac_al = 0.2
internal_state = None
for j, OF in enumerate(chem_OF):
    for i, Pc in enumerate(chem_Pc):
        # print('OF={OF}, Pc={Pc}atm'.format(OF=OF, Pc=Pc/_atm))
        o = OF / (1 + OF) # o/f = OF, o+f=1 => o=OF/(1 + OF)
        flame, internal_state = comb.solve(Pc, {ox: o, fu_1: (1-mfrac_al)*(1-o), fu_2: mfrac_al*(1-o)}, max_iters=150, internal_state=internal_state)
        chem_k[i,j], chem_M[i,j], chem_T[i,j] = flame.gamma, flame.M, flame.T

print('Baking LOX saturated property curves...')
get_sat_lox_props = fluid.bake_sat_coolprop('Oxygen', np.linspace(75, 150, 20))

print('Baking grain geometry')
# Define star using inner diameter, tip diameter, and number of tips; then bake into regress distance to exposed arc area curve fit
grn_ID, grn_TD, grn_OD, N_tip = 2.0*_in, 3.5*_in, 4.5*_in, 6
grn_xy = make_star_vertices(grn_ID, grn_TD, N_tip)
grn_A0, grn_d, grn_d2a, all_contours = bake_d2a(grn_OD, grn_xy, Nx=64, Nd=16)
d2a_curve = interpax.Interpolator1D(grn_d, grn_d2a, method='akima')

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
# axs = np.array(axs).ravel()
# Nd_plt = 128
# plt_grn_xy = np.append(grn_xy, [grn_xy[0,:]], axis=0)
# plt_t = np.linspace(0.0, 2*np.pi, 64)
# plt_d = np.arange(Nd_plt)/(Nd_plt-1)*grn_d[-1]
# eq_r = grn_d2a[0]/(2*np.pi)
# eq_d = np.arange(Nd_plt)/(Nd_plt-1)*(grn_OD/2 - eq_r)

# for contour in all_contours:
    # axs[0].plot(contour[:,0], contour[:,1], color='black')
# axs[0].plot(plt_grn_xy[:,0], plt_grn_xy[:,1], color='black', linestyle='dashed')
# axs[0].plot(np.cos(plt_t)*grn_OD/2, np.sin(plt_t)*grn_OD/2, color='black', linestyle='dashed')
# axs[0].plot(np.cos(plt_t)*eq_r, np.sin(plt_t)*eq_r, color='blue', linestyle='dashed', label='equivalent diameter')
# axs[0].legend(loc='upper right')

# axs[1].plot(plt_d, d2a_curve(plt_d), label='star')
# axs[1].scatter(grn_d, grn_d2a)
# axs[1].plot(np.append(eq_d, eq_d[-1]), np.append(2*np.pi*(eq_r + eq_d),0.0), label='circular w/ equivalent diameter')
# axs[1].legend(loc='upper right')
# axs[1].set_xlabel('Regressed Distance [m]')
# axs[1].set_ylabel('Exposed Arc Length [m]')
# plt.show()



print('Initializing engine...')
tnk = make_sat_tank(
    get_sat_lox_props,
    V = (np.pi/4 * 4.75**2 * _in**2) * (4.0 * _ft),
    inj_CdA = 0.22 * (np.pi/4 * 0.5**2 * _in**2),
    m_ox = 3.0, # TODO: init limit
    T = 125,
)

# shape = make_circle_shape(
    # ID = 2.5 * _in,
# )
shape = make_arbitrary_shape(
    d2a_curve,
    A0 = grn_A0,
)
grn = make_constOF_grain(
    shape,
    OF = 4.5,
    OD = grn_OD,
    L = 20.0 * _in,
    rho = 1117.0,
)

cmbr = make_chamber(
    V0 = 0.0,
    cstar_eff = 1.0,#0.95
)

noz = make_cd_nozzle(
    thrt = 1.75 * _in, # Throat diameter
    ER = 4.99,         # Exit/throat area ratio
    eff = 0.97,
    C_d = 0.995,
)

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
T = 7.0
print('Running...')
import time
t1 = time.time()
t, _x, xstack = fire_engine(s, x, dt=1E-3, T=T)
jax.block_until_ready(xstack)
t2  = time.time()
t, x, xstack = fire_engine(s, x, dt=1E-3, T=T)
jax.block_until_ready(xstack)
t3 = time.time()
print('done, compile+run was {a:.2f}s, just run was {b:.2f}s'.format(a=t2-t1, b=t3-t2))

# Unpack the dynamic engine state
N_t = xstack.shape[0]
tnk, grn, cmbr, noz = core.unpack_engine(s, xstack, method)

# Ensure results folder exists
results_path = Path('./results')
results_path.mkdir(parents=True, exist_ok=True)



# Visualization
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12,7))
axs = np.array(axs).ravel()

# Plot thrust
axs[0].plot(np.linspace(0.0, T, N_t), noz['thrust'], label='sim')
axs[0].set_title('Thrust')

# Plot oxidizer flow rate
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_ox'], label='mdot_ox')
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_inj'], label='mdot_inj')
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_vnt'], label='mdot_vnt')
axs[1].plot(np.linspace(0.0, T, N_t), grn['mdot'], label='mdot_grn')
axs[1].plot(np.linspace(0.0, T, N_t), noz['mdot'], label='mdot_noz')
axs[1].plot(np.linspace(0.0, T, N_t), cmbr['mdot_g'], label='mdot_cmbr')
axs[1].legend(loc='upper right')
axs[1].set_title('mdot')

axs[2].plot(np.linspace(0.0, T, N_t), cmbr['P'], label='chamber')
axs[2].plot(np.linspace(0.0, T, N_t), tnk['P'], label='tank')
axs[2].plot(np.linspace(0.0, T, N_t), noz['Pe'], label='noz exit')
axs[2].legend(loc='upper right')
axs[2].set_title('P')

axs[3].plot(np.linspace(0.0, T, N_t), tnk['T'])
axs[3].set_title('T tank')

axs[4].plot(np.linspace(0.0, T, N_t), tnk['m_ox_liq'], label='ox liq')
axs[4].plot(np.linspace(0.0, T, N_t), tnk['m_ox_vap'], label='ox vap')
axs[4].plot(np.linspace(0.0, T, N_t), cmbr['m_g'], label='cmbr stored')
axs[4].plot(np.linspace(0.0, T, N_t), grn['V']*grn['rho'], label='grain')
axs[4].legend()
axs[4].set_title('m')

D = (4.5 - 2.5)*_in # TODO: get
axs[5].plot([0.0, T], [D]*2, label='grain thickness')
axs[5].plot(np.linspace(0.0, T, N_t), grn['d'], label='net regression')
# axs[5].plot(np.linspace(0.0, T, N_t), grn['V'], label='grain volume')
axs[5].legend()

axs[6].plot(np.linspace(0.0, T, N_t), noz['Me'], label='Mach exit')
axs[6].legend()

# axs[7].plot(np.linspace(0.0, T, N_t), cmbr['V0'] - 0*grn['V'], label='cmbr V0')
# axs[7].plot(np.linspace(0.0, T, N_t), grn['V'], label='grain V')
axs[7].plot(np.linspace(0.0, T, N_t), cmbr['cstar'], label='cstar')
axs[7].plot(np.linspace(0.0, T, N_t), cmbr['T'], label='cmbr T')
# axs[7].plot(np.linspace(0.0, T, N_t), cmbr['V0'] - grn['V'], label='Empty cmbr V')
# axs[5].plot(np.linspace(0.0, T, N_t), grn['V'], label='grain volume')
axs[7].legend()

# axs[8].plot(np.linspace(0.0, T, N_t), cmbr['k'], label='cmbr k')
axs[8].plot(np.linspace(0.0, T, N_t), cmbr['V0'] - grn['V'], label='Empty cmbr V')
# axs[8].plot(np.linspace(0.0, T, N_t), grn['V'], label='grain volume')
# axs[8].plot(np.linspace(0.0, T, N_t), cmbr['Pdot'], label='Pc dot')
# axs[8].plot(np.linspace(0.0, T, N_t), grn['Vdot'], label='grain V dot')
# axs[8].plot(np.linspace(0.0, T, N_t), grn['Vdot']/(cmbr['V0'] - grn['V']), label='Pc dot, V comb')
# Pc*(mdot_g/m_g - dV/V)
axs[8].legend()



# Save plot
fig.tight_layout()
fig.savefig(results_path/(file_prefix+'_plots.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
