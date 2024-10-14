import core
from tank    import *
from grain   import *
from chamber import *
from nozzle  import *
from units   import *
from sat_nos import *

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

grn = make_grain(
)

cmbr = make_chamber(
)

noz = make_cd_nozzle(
    thrt = 0.5 * _in, # Throat diameter
    ER = 4.0,         # Exit/throat area ratio
)

s, x, method = core.make_engine(
    tank, grn, cmbr, noz,
    'cstar': 1,
)



# Create the function for firing engines
#   This will be compiled the first time you call it during a run
fire_engine = core.make_integrator(
    core.step_rk4,
    method,
)

# Integrate the engine state
t, x = fire_engine(s, x, dt=1E-2, T=10.0)

# Unpack the dynamic engine state
tank, grn, cmbr, noz = unpack_engine(s, x)



# Plot thrust
axs[0].plot(t, noz['thrust'])

# Plot oxidizer flow rate
axs[0].plot(t, tank['mdot_ox'])

# Plot nozzle flow rate

# Save plot
