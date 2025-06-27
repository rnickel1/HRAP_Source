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
from hrap.units   import _in, _ft, _lbf, _atm, _psi



jax.config.update("jax_enable_x64", True)

# Non thermal equilibrium (lengthwise) test on a tank pressure vs time

k = 0.0865 # W / (m K), from liquid nitrous at around 60def F
# k*=100

from hrap.sat_nos import get_sat_nos_props
get_sat_props = get_sat_nos_props

dt = 1E-3
Nt = 2000
Nl = 64
Nt_plt = 5

T0 = 294.0 # K
Pc = 400.0 * _psi
L = 2 # m
A = np.pi/4 * 4.75**2 * _in**2
V = L*A
dl = L / (Nl-1)
lvec = np.arange(Nl)*dl

m_liq, m_vap = np.zeros(Nt), np.zeros(Nt)
m_liq[0] = V * get_sat_props(T0)['rho_l']
m_vap[0] = get_sat_props(T0)['rho_v'] * (V - m_liq[0] / get_sat_props(T0)['rho_l'])

Teq = np.full(Nt, T0)
T = np.full((Nt, Nl), T0)
tvec = np.arange(Nt)*dt

CdA = 0.2 * (np.pi/4 * 0.5**2 * _in**2)

for it in range(1, Nt):
    bprops = get_sat_props(T[it-1,0]) # bottom node
    lprops = get_sat_props(T[it-1,1:-1]) # Properties of last state on interior nodes
    tprops = get_sat_props(T[it-1,-1]) # top node

    # mdot_out = CdA*np.sqrt(np.max([2 * bprops['rho_l'] * (tprops['Pv'] - Pc), 0.0]))
    mdot_out = CdA*np.sqrt(np.max([2 * bprops['rho_l'] * tprops['Pv']/2, 0.0]))

    # TODO: will need to consider density along length to get ullage conditions (will expand as heats) - combine with finite evap rate?
    # V_liq = np.sum(lprops['rho_l'] * )
    V_liq = m_liq[it-1] / np.mean(lprops['rho_l'])
    V_vap = m_vap[it-1] / tprops['rho_v']

    # Evap to fill remaining space
    mdot_evap = np.max([tprops['rho_v'] * (V - V_liq - V_vap), 0.0])/dt
    # tprops['Hv']*mdot_evap describes rate of total heat removal due to evap (W)
    # Hv*mdot*dt
    q_evap = -tprops['Hv']*mdot_evap / A # Flux, W/m^2
    if it > 1:
        q_evap -= (T[it-1,-1] - T[it-2,-1])/dt * m_vap[it-1] * tprops['Cp'] / A # Cancel out heat supplied to equilibrium with vapor
        # print(q_evap, (T[it-1,-1] - T[it-2,-1])/dt * m_vap[it-1] * tprops['Cp'] / A)
    # print(V, (V - V_liq - V_vap))#V_liq, V_vap)

    eq_props = get_sat_props(Teq[it-1])
    Teq[it] = Teq[it-1] - (eq_props['Hv']*mdot_evap*dt)/((m_vap[it-1]+m_liq[it-1])*eq_props['Cp'])
    
    alpha = k / (lprops['rho_l'] * lprops['Cp']) # TODO; all
    # print(mdot_evap, q_evap, alpha[-1], q_evap*alpha[-1]/dl, T[it-1,-1])
    # print('rates: cond', alpha[-1]*(T[it-1,-2]-T[it-1,-1])/dl**2, 'from', T[it-1,-1], T[it-1,-2], 'in', alpha[-1]*2*q_evap/dl, 'from', mdot_evap, q_evap)
    T[it,-1] = T[it-1,-1] + alpha[-1]*dt*(2*(T[it-1,-2]-T[it-1,-1])/dl**2 + 2*q_evap/dl) # Heat lost from evap
    T[it,1:-1] = T[it-1,1:-1] + alpha*dt * (T[it-1,2:] - 2*T[it-1,1:-1] + T[it-1,:-2])/dl**2
    T[it,0] = T[it,1] # 0 Neumann BCs on bottom

    m_liq[it] = m_liq[it-1] - (mdot_out + mdot_evap) * dt
    m_vap[it] = m_vap[it-1] +  mdot_evap * dt

# fig
ax = plt.subplot(1, 3, 1)
for j in range(Nt_plt):
    it = j * ((Nt-1)//(Nt_plt-1))
    ax.plot(T[it,:], lvec, linestyle='solid', color='C{i}'.format(i=j), label='t={t}'.format(t=tvec[it]))
    ax.plot([Teq[it]]*2, [lvec[0], lvec[-1]], linestyle='dashed', color='C{i}'.format(i=j))
ax.legend()

ax = plt.subplot(1, 3, 2)
ax.plot(tvec, m_liq, label='liquid')
ax.plot(tvec, m_vap, label='vapor')
ax.plot(tvec, m_liq+m_vap, label='total')
ax.legend()

ax = plt.subplot(1, 3, 3)
ax.plot(tvec, get_sat_props(T[:,-1])['Pv'])
# print(T[:,-1])

plt.show()
