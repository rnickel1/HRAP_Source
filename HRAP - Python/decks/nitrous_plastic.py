# Initialize the components
tank = make_sat_nos_tank(

)
cmbr = make_chamber(
)
noz = make_cd_nozzle(
    thrt = 0.5 * _in,
    ER = 4.0,
)

s, x = make_engine(tank=tank, grn=grn, inj=inj, cmbr=cmbr, noz=noz)
s['cstar'] = 1



# Create the function for firing engines
#   This will be compiled the first time you call it during a run
fire_engine = core.make_integrator(
    core.step_rk4,
    eng,
)

# Integrate the engine state
t, x, dx = fire_engine(s, x, dt=1E-2, T=10.0)

# Unpack the dynamic engine state
tank, grn, inj, cmbr, noz = unpack_engine(s, x, dx)



# Plot thrust
axs[0].plot(t, noz['thrust'])

# Plot oxidizer flow rate
axs[0].plot(t, tank['thrust'])

# Plot nozzle flow rate

# Save plot
