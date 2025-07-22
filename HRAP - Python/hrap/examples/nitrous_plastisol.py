import sys
sys.path.insert(1, '../HRAP/')

import scipy
import numpy as np
from pathlib import Path
from importlib.resources import files as imp_files

import matplotlib.pyplot as plt

# import cantera as ct

import hrap.core as core
import hrap.chem as chem
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.sat_nos import *
from hrap.units   import _in, _ft, _lbf, _atm


jax.config.update("jax_enable_x64", True)
# np.random.seed(42)
hrap_root = Path(imp_files('hrap'))



print('Initializing chemistry...')
use_prebuilt_chem = True
# use_prebuilt_chem = False
if use_prebuilt_chem:
    chem = scipy.io.loadmat(hrap_root/'resources'/'propellant_configs'/'Metalized_Plastisol.mat')
    chem = chem['s'][0][0]
    chem_OF = chem['prop_OF'].ravel()
    chem_Pc = chem['prop_Pc'].ravel()
    chem_k, chem_M, chem_T = chem['prop_k'], chem['prop_M'], chem['prop_T']
    if chem_k.size == 1: chem_k = np.full_like(chem_T, chem_k.item())
else:
    plastisol = chem.make_basic_reactant(
        formula = 'Plastisol-362',
        composition = { 'C': 7.200, 'H': 10.82, 'O': 1.14, 'Cl': 0.669 },
        M = 140.86, # kg/kmol
        T0 = 298.15, # K
        h0 = -265357.55, # J/mol
    )

    comb = chem.ChemSolver(['./ssts_thermochem.txt', plastisol])
    print('Building combustion table')
    # TODO: separate 3-table for vapor like ssts
    chem_OF, chem_Pc = np.linspace(1.0, 10.0, 10), np.linspace(10*_atm, 50*_atm, 10)
    ox, fu = 'N2O(L),298.15K', 'Plastisol-362'
    for OF in chem_OF:
        for Pc in chem_Pc:
            o = OF / (1 + OF) # o/f = OF, o+f=1 => o=OF/(1 + OF)
            res = comb.solve(Pc, {ox: (o, 298, 1*_atm), fu: (1-o, 298, 1*_atm)})
            print('OF={OF}, Pc={Pc}atm'.format(OF=OF, Pc=Pc/_atm))
print('Loading chemistry table...')
chem = scipy.io.loadmat(hrap_root/'resources'/'propellant_configs'/'Metalized_Plastisol.mat')
chem = chem['s'][0][0]
chem_OF = chem['prop_OF'].ravel()
chem_Pc = chem['prop_Pc'].ravel()
chem_k, chem_M, chem_T = chem['prop_k'], chem['prop_M'], chem['prop_T']
if chem_k.size == 1: chem_k = np.full_like(chem_T, chem_k.item())



# Initialization
tnk = make_sat_tank(
    get_sat_nos_props,
    V = (np.pi/4 * 4.75**2 * _in**2) * (7.0 * _ft),
    inj_CdA= 0.22 * (np.pi/4 * 0.5**2 * _in**2),
    m_ox=12.6, # TODO: init limit
    T = 294,
)

shape = make_circle_shape(
    ID = 2.5 * _in,
)
grn = make_constOF_grain(
    shape,
    OF = 3.5,
    OD = 4.5 * _in,
    L = 30.0 * _in,
    rho = 1117.0,
)

prepost_ID = 4.25*_in                              # Inner diameter of pre and post combustion chambers (m)
prepost_V  = (3.5+1.7)*_in * np.pi/4*prepost_ID**2 # Empty volume of pre and post combustion chambers (m^3)
rings_V    = 3 * (1/8*_in) * np.pi*(2.5/2 * _in)**2  # Empty volume of phenolic rings (m^3)
fuel_V     = (30.0 * _in) * np.pi*(4.5/2 * _in)**2   # Empty volume of grain footprint (m^3)
cmbr = make_chamber(
    # V0 =  prepost_V + rings_V + fuel_V,            # Volume of chamber w/o grain (m^3)
    V0 = 0.0, # Sim can be a bit unstable with this and incompressible injetor
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
T = 12.0
# T = 10E-2
print('Running...')
import time
t1 = time.time()
t, _x, xstack = fire_engine(s, x, dt=1E-3, T=T)
jax.block_until_ready(xstack)
t2  = time.time()
t, x, xstack = fire_engine(s, x, dt=1E-3, T=T)
jax.block_until_ready(xstack)
t3 = time.time()
print('done, first run was {a:.2f}s, second run was {b:.2f}s'.format(a=t2-t1, b=t3-t2))

# Unpack the dynamic engine state
N_t = xstack.shape[0]
tnk, grn, cmbr, noz = core.unpack_engine(s, xstack, method)
# print('tnk', tnk.keys())
# print('grn', grn.keys())
# print('cmbr', cmbr.keys())
# print('noz', noz.keys())
# print()
# print('Post-run arrays:')
# for name, obj in (('tnk', tnk), ('grn', grn), ('cmbr', cmbr), ('noz', noz)):
#     print(name+':')
#     for key, val in obj.items():
#         print(key+':', val)
#     print()

# Ensure results folder exists
results_path = Path('./results')
results_path.mkdir(parents=True, exist_ok=True)

OD, L = 5*_in, 10*_ft
file_prefix = 'nitrous_plastisol'
core.export_rse(
    results_path/(file_prefix+'.rse'),
    t, noz['thrust'].ravel(), noz['mdot'].ravel(), t*0, t*0,
    OD=OD, L=L, D_throat=s['noz_thrt'], D_exit=np.sqrt(s['noz_ER'])*s['noz_thrt'],
    motor_type='hybrid', mfg='HRAP',
)
core.export_eng(
    results_path/(file_prefix+'.eng'),
    t, noz['thrust'], t*0,
    OD=OD, L=L,
    mfg='HRAP',
)



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


# Write thrust validation, big hybrid 7-26-23
daq = np.genfromtxt(hrap_root/'resources'/'validation'/'hybrid_fire_7_26_23.csv', delimiter=',', names=True, dtype=float, encoding='utf-8', deletechars='')
axs[0].plot(daq['time'], daq['thrust']*_lbf, label='daq')
axs[0].legend()
# axs[0].plot()

# Plot nozzle flow rate

# Save plot
fig.tight_layout()
fig.savefig(results_path/(file_prefix+'_plots.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
