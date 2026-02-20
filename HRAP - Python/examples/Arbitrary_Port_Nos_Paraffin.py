# Purpose: Demonstrate arbitrary grain port analysis feasability
# Authors: Timon Jacquemin

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
file_prefix = 'Nos-Paraffin-Arbitrary'



print('Building combustion chemistry table...')
paraffin_vybar = chem.make_basic_reactant(
    formula = 'paraffin_vybar',
    composition = { 'C': 71.02, 'H': 145.83 },
    M = 422.8, # kg/kmol
    T0 = 298.15, # K
    h0 = -9.25e6, # J/kmol
)
comb = chem.ChemSolver([hrap_root/'thermo.dat', paraffin_vybar])
chem_Pc, chem_OF = np.linspace(10*_atm, 50*_atm, 10), np.linspace(1.0, 15.0, 25)
chem_k, chem_M, chem_T = [np.zeros((chem_Pc.size, chem_OF.size)) for i in range(3)]
ox, fu_1, fu_2 = 'N2O', 'paraffin_vybar', 'AL(cr)'
mfrac_al = 0.2
internal_state = None
for j, OF in enumerate(chem_OF):
    for i, Pc in enumerate(chem_Pc):
        # print('OF={OF}, Pc={Pc}atm'.format(OF=OF, Pc=Pc/_atm))
        o = OF / (1 + OF) # o/f = OF, o+f=1 => o=OF/(1 + OF)
        flame, internal_state = comb.solve(Pc, {ox: o, fu_1: (1-mfrac_al)*(1-o), fu_2: mfrac_al*(1-o)}, max_iters=150, internal_state=internal_state)
        chem_k[i,j], chem_M[i,j], chem_T[i,j] = flame.gamma, flame.M, flame.T

print('Baking N2O saturated property curves...')
get_sat_nos_props = fluid.bake_sat_coolprop('NitrousOxide', np.linspace(240, 305, 25))


print('Baking grain geometry')

grain_diameter=0.023

distances,perimeters,contours,grn_A0 = bake_arbitrary_d2a(
    'C:/Users/timon/OneDrive/Desktop/Grain_shapes/Grain2.png',
    grain_diameter=grain_diameter, 
    n_step=150, 
    n_visu=50
    )

d2a_curve = interpax.Interpolator1D(distances, perimeters, method='akima')



# Graph things

# Create output figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Color vector
colors = cm.viridis(np.linspace(0, 1, len(perimeters)))

# Contour Plot
ax1.set_title("Grain regression")
ax1.set_aspect('equal')
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")

for cont in contours:
    ax1.plot(cont[0], cont[1], linewidth=2, color=colors[cont[2]], alpha=0.8)
       
# Show grain exterior diameter
grain_ext = plt.Circle((grain_diameter/2, grain_diameter/2), grain_diameter/2,color='black', fill=False, linewidth=4, zorder=10)
ax1.add_patch(grain_ext)

# Data plot
# Colorbar (matches contour plot)
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=grain_diameter))
cbar = plt.colorbar(sm, ax=ax2)
cbar.set_label('Distances (m)')

# Perimeter curve with matching colors
for i in range(len(distances) - 1):
    ax2.plot(distances[i:i+2], perimeters[i:i+2], color=colors[i], linewidth=3)    

ax2.plot(distances, perimeters, 'k.', markersize=5, alpha=0.5 , label='Grain') # points noirs discrets

ax2.set_title("Surface evolution ($A_b$)")
ax2.set_xlabel("Regressed distance (m)")
ax2.set_ylabel("Perimeters (m)")
ax2.grid(True, linestyle='--', alpha=0.6)
# Set y minimum to 0
ax2.set_ylim(bottom=0)

Nd_plt = 100
plt_t = np.linspace(0.0, 2*np.pi, 64)
plt_d = np.arange(Nd_plt)/(Nd_plt-1)*distances[-1]
eq_r = perimeters[0]/(2*np.pi)
eq_d = np.arange(Nd_plt)/(Nd_plt-1)*(grain_diameter/2 - eq_r)

ax1.plot(list(map(lambda x: x + grain_diameter/2, np.cos(plt_t)*eq_r)), list(map(lambda x: x + grain_diameter/2,  np.sin(plt_t)*eq_r)), color='blue', linestyle='dashed', label='equivalent diameter')
ax1.legend(loc='upper right')

if (eq_r < grain_diameter/2):
    ax2.plot(np.append(eq_d, eq_d[-1]), np.append(2*np.pi*(eq_r + eq_d),0.0), color='tab:orange', label='circular w/ equivalent diameter')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()





print('Initializing engine...')

#tank
V = (np.pi/4 * 0.0256**2) * 0.124
T_init = 293.15 # Kelvin
rho_liq = get_sat_nos_props(T_init)['rho_l']
fill_level = 0.95
m_ox = V * fill_level * rho_liq

tnk = make_sat_tank(
    get_sat_nos_props,
    V = V ,
    inj_CdA = 0.45 * (np.pi/4 * 0.001**2) * 2,
    m_ox = m_ox,
    T = T_init,
    inj_vap_model = StaticVar('Real Gas'),
)


#grain
shape = make_arbitrary_shape(
    d2a_curve,
    A0 = grn_A0,
)

grn = make_shiftOF_grain(
    shape,
    OD = grain_diameter,
    L = 0.035,
    Reg = jnp.array([0.12, 0.55, 0.0]),
    rho = 900,
)


#chamber
cmbr = make_chamber(
    V0 = 0.0,
    cstar_eff = 1.0,#0.95
)


#nozzle
d_throat = 0.008
d_exit = 0.0165

noz = make_cd_nozzle(
    thrt = d_throat, # Throat diameter
    ER = (d_exit / d_throat)**2,         # Exit/throat area ratio
    eff = 1.0,
    C_d = 1.0,
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
T = 2.0
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