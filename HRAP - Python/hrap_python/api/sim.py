"""
HRAP Simulation Components (`hrap_python.api.sim`)

State-transition components for HRAP simulations
"""
from math import pi, sqrt
from util.nox import get_NOX_properties, vapor_pressure
from util.units import d_to_a
from .models import SimulationConfig, SimulationState, VentConfig, TankState
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve   

def tank(config: SimulationConfig, state: SimulationState):
    """
        Iterates the current tank conditions.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        state : SimulationState
            The current simulation state

        Returns
        -------
        None
    """
    dt = config.timestep

    # Find oxidizer thermophysical properties
    state.tank.ox_props = get_NOX_properties(state.tank.temp)

    state.tank.pressure = state.tank.ox_props.Pv

    # Find oxidizer mass flow rate
    dP = state.tank.pressure - state.chamber.pressure

    if dP < 0:
        dP = 0.0

    state.pressure_drop = dP

    if config.max_burn_time == 0 or state.time <= config.max_burn_time:
        if config.hardware.vent_state == VentConfig.NONE:
            # No vent - no gas escapes the tank
            state.tank.mdot_vapor = 0.0
            if state.tank.mLiq_new == 0:
                state.tank.mdot_ox = config.hardware.injector_cda_N*sqrt(2*state.tank.ox_props.rho_v*dP)
            else:
                state.tank.mdot_ox = config.hardware.injector_cda_N*sqrt(2*state.tank.ox_props.rho_l*dP)
            
            mass_discharged = (state.tank.mdot_ox+state.tank.mdot_vapor)*dt
        elif config.hardware.vent_state == VentConfig.EXTERNAL:
            # External vent - gases exit through tank forward closure
            state.tank.mdot_vapor = config.hardware.vent_cda*sqrt(2*state.tank.ox_props.rho_v*dP)
            if state.tank.mLiq_new == 0:
                state.tank.mdot_ox = config.hardware.injector_cda_N*sqrt(2*state.tank.ox_props.rho_v*dP)
            else:
                state.tank.mdot_ox = config.hardware.injector_cda_N*sqrt(2*state.tank.ox_props.rho_l*dP)
            
            mass_discharged = (state.tank.mdot_ox+state.tank.mdot_vapor)*dt
        elif config.hardware.vent_state == VentConfig.INTERNAL:
            # Internal vent - gases enter combustion chamber
            state.tank.mdot_vapor = config.vnt_CdA*sqrt(2*state.tank.ox_props.rho_v*dP)
            if state.tank.mLiq_new == 0:
                state.tank.mdot_ox = config.hardware.injector_cda_N*sqrt(2*state.tank.ox_props.rho_v*dP) + state.tank.mdot_vapor
            else:
                state.tank.mdot_ox = config.hardware.injector_cda_N*sqrt(2*state.tank.ox_props.rho_l*dP) + state.tank.mdot_vapor
            
            mass_discharged = state.tank.mdot_ox*dt
        else:
            print('Error: Vent State Undefined')
            exit(-1)
        
    elif config.max_burn_time > 0 and state.time > config.max_burn_time:
        state.tank.mdot_ox  = 0
        mass_discharged     = 0

    # Find mass discharged during time step
    state.tank.mass_ox_old  = state.tank.mass_ox
    state.tank.mass_ox          = state.tank.mass_ox - state.tank.mdot_ox*dt

    if state.tank.mLiq_new < state.tank.mLiq_old and state.tank.mLiq_new > 0 and state.tank.mdot_ox > 0:

        # Find mass of liquid nitrous evaporated during time step
        state.tank.mLiq_old = state.tank.mLiq_new - mass_discharged
        state.tank.ox_props = get_NOX_properties(state.tank.temp)
        state.tank.mLiq_new = TankState.liquid_mass(config.hardware.tank_volume, state.tank.mass_ox, state.tank.ox_props)
        mv = state.tank.mLiq_old - state.tank.mLiq_new

        # Find heat removed from liquid
        dT = -mv*state.tank.ox_props.Hv/(state.tank.mLiq_new*state.tank.ox_props.Cp)
        state.tank.temp = state.tank.temp + dT
        op = get_NOX_properties(state.tank.temp)

        vapor_dP = op.Pv - state.tank.pressure
        if vapor_dP < 0:
            state.total_vapor_dP += vapor_dP
            state.interpolation_timesteps += 1.0

    elif state.tank.mLiq_new >= state.tank.mLiq_old and state.tank.mLiq_new > 0 and state.tank.mdot_ox > 0:
        dP_avg = state.total_vapor_dP / state.interpolation_timesteps

        P_new = state.tank.pressure + dP_avg

        state.tank.temp = fsolve(lambda T: vapor_pressure(T) - P_new, state.tank.temp)[0] # TODO: Error handling

        vapor_dP = state.tank.ox_props.Pv - state.tank.pressure
        if vapor_dP < 0:
            state.total_vapor_dP += vapor_dP
            state.interpolation_timesteps += 1.0

        state.tank.ox_props = get_NOX_properties(state.tank.temp)

        state.tank.mLiq_new = TankState.liquid_mass(config.hardware.tank_volume, state.tank.mass_ox, state.tank.ox_props)
        state.tank.mLiq_old = 0

    elif state.tank.mLiq_new <= 0 and state.tank.mdot_ox > 0:
        if state.tank.mLiq_new != 0:
            state.tank.mLiq_new = 0

        # Find Z factor
        Z_old = state.tank.ox_props.Z

        Zguess = Z_old
        epsilon = 1.0
        
        T_i = state.tank.temp
        P_i = state.tank.pressure

        while epsilon >= 0.000001:
            T_ratio = ((Zguess * state.tank.mass_ox) / (Z_old * state.tank.mass_ox_old)) ** 0.3
            state.tank.temp = T_ratio * T_i
            P_ratio = T_ratio ** (1.3 / 0.3)
            state.tank.pressure = P_ratio * P_i

            state.tank.ox_props = get_NOX_properties(state.tank.temp)

            Z = state.tank.ox_props.Z
            
            epsilon = abs(Zguess - Z)

            Zguess = (Zguess + Z) / 2

def shift_OF(config: SimulationConfig, state: SimulationState):
    """
        Iterates the propellant regression state assuming a shifting OF-ratio model.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        state : SimulationState
            The current simulation state

        Returns
        -------
        None
    """
    dt = config.timestep

    reg_coeff               = config.prop.material.regression_coeff
    ox_flux                 = state.tank.mdot_ox / d_to_a(state.chamber.grain_ID)
    # Convert regression rate from m/s to mm/s
    # Not a fan of the magic number here, but unfortunately the ballistic equation is
    # given in units of mm/s. At least it's not leagues per fortnight
    state.chamber.rdot      = 0.001 * reg_coeff[0] * (ox_flux ** reg_coeff[1]) * (config.prop.length ** reg_coeff[2])
    state.chamber.mdot_fuel = config.prop.material.rho * state.chamber.rdot * pi * state.chamber.grain_ID * config.prop.length
    state.chamber.OF_ratio  = state.tank.mdot_ox / state.chamber.mdot_fuel
    
    if state.chamber.mdot_fuel == 0:
        state.chamber.OF_ratio = 0

    new_grain_ID                = state.chamber.grain_ID + (2 * state.chamber.rdot * dt)
    # TODO: Prefer vdot * rho for mdot calculation?
    state.chamber.vdot_fuel     = (d_to_a(new_grain_ID) - d_to_a(state.chamber.grain_ID)) * config.prop.length / dt
    state.chamber.grain_ID      = new_grain_ID
    state.chamber.mass_fuel     = state.chamber.mass_fuel - (state.chamber.mdot_fuel * dt)

def const_OF(config: SimulationConfig, state: SimulationState):
    """
        Iterates the propellant regression state assuming a constant OF-ratio model.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        state : SimulationState
            The current simulation state

        Returns
        -------
        None
    """
    dt = config.timestep

    state.chamber.mdot_fuel     = state.tank.mdot_ox / config.prop.const_OF_ratio
    state.rdot                  = state.chamber.mdot_fuel / (config.prop.material.rho * pi * state.chamber.grain_ID * config.prop.length)

    new_grain_ID                = state.chamber.grain_ID + (2 * state.rdot * dt)
    state.chamber.vdot_fuel     = (d_to_a(new_grain_ID) - d_to_a(state.chamber.grain_ID)) * config.prop.length / dt
    state.chamber.grain_ID      = new_grain_ID
    state.chamber.mass_fuel     = state.chamber.mass_fuel - (state.chamber.mdot_fuel * dt)

def combustion(config: SimulationConfig, state: SimulationState):
    """
        Iterates the combustion state.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        state : SimulationState
            The current simulation state

        Returns
        -------
        None
    """
    if state.time <= config.max_burn_time or config.max_burn_time == 0:
        data = config.prop.material
        k_interp            = RegularGridInterpolator((data.Pc, data.OF), data.k, bounds_error=False, fill_value=None) # TODO: Interpolation method, optimization
        k                   = k_interp((state.chamber.pressure, state.chamber.OF_ratio))
        state.combustion.k  = k # Easier to just reference k
        M_interp            = RegularGridInterpolator((data.Pc, data.OF), data.M, bounds_error=False, fill_value=None)
        state.combustion.M  = M_interp((state.chamber.pressure, state.chamber.OF_ratio))
        T_interp            = RegularGridInterpolator((data.Pc, data.OF), data.T, bounds_error=False, fill_value=None)
        state.combustion.T  = T_interp((state.chamber.pressure, state.chamber.OF_ratio))
        
        state.combustion.R      = 8314.5 / state.combustion.M
        state.rho               = state.chamber.pressure / (state.combustion.R * state.combustion.T)
        state.combustion.cstar  = config.prop.cstar_eff * sqrt((state.combustion.R * state.combustion.T) / (k * ((2/(k+1)) ** ((k+1)/(k-1)))))

def chamber(config: SimulationConfig, state: SimulationState):
    """
        Iterates the chamber state (mass flux, pressure change, etc.).

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        state : SimulationState
            The current simulation state

        Returns
        -------
        None
    """
    dt = config.timestep

    if config.hardware.chamber_volume == 0:
        V = d_to_a(state.chamber.grain_ID) * config.prop.length
    else:
        V = config.hardware.chamber_volume - (d_to_a(config.prop.OD) - d_to_a(state.chamber.grain_ID)) * config.prop.length
    
    state.chamber.mdot_exit = state.chamber.pressure * config.nozzle.Cd * d_to_a(config.nozzle.throat) / state.combustion.cstar

    dm_g = state.chamber.mdot_fuel + state.tank.mdot_ox - state.chamber.mdot_exit
    
    if state.tank.mdot_ox == 0:
        state.dm_g = -1 * state.chamber.mdot_exit

    state.chamber.mass_gas = state.chamber.mass_gas + (dm_g * dt)

    dP = state.chamber.pressure * ((dm_g / state.chamber.mass_gas) - state.chamber.vdot_fuel/V)

    state.chamber.pressure = state.chamber.pressure + (dP * dt)

    if state.chamber.pressure <= config.ambient_pressure:
        state.chamber.pressure = config.ambient_pressure
        state.chamber.mdot_exit = 0

def nozzle(config: SimulationConfig, state: SimulationState):
    """
        Iterates the nozzle state - the final state transition in a typical simulation iteration.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        state : SimulationState
            The current simulation state

        Returns
        -------
        None
    """
    if state.chamber.pressure > config.ambient_pressure:
        k = state.combustion.k # Simpler to just reference k
        A_Ratio         = lambda M: (
                                      ((((k+1)/2) ** (-(k+1)/(2*(k-1)))) *
                                      ((1+(k-1)/2*(M**2)) ** ((k+1)/(2*(k-1)))) / M)
                                      - config.nozzle.exp_ratio
                                    )
        M               = fsolve(A_Ratio, 3.0)[0] # TODO: Error handling
        Pe              = state.chamber.pressure * ((1 + 0.5*(k-1)*(M**2)) ** (-1*k/(k-1)))
        Cf              = (
                                sqrt(((2*(k**2))/(k-1)) *
                                    ((2/(k+1)) ** ((k+1)/(k-1))) * 
                                    (1-((Pe/state.chamber.pressure) ** ((k-1)/k)))) +
                                ((Pe-config.ambient_pressure)*(d_to_a(config.nozzle.throat)*config.nozzle.exp_ratio)) /
                                (state.chamber.pressure*d_to_a(config.nozzle.throat))
                          )
        state.thrust    = config.nozzle.efficiency * Cf * d_to_a(config.nozzle.throat) * state.chamber.pressure * config.nozzle.Cd
        
        if state.thrust < 0:
            state.thrust = 0

    else:
        state.thrust = 0

def mass(config: SimulationConfig, state: SimulationState):
    """
        Iterates the mass properties of a state - optional, only when specified.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        state : SimulationState
            The current simulation state

        Returns
        -------
        None
    """
    state.mass.mass_total = config.mass_properties.motor_mass + state.tank.mass_ox + state.chamber.mass_fuel

    if state.tank.mLiq_new < 0:
        state.tank.mLiq_new = 0

    tA = d_to_a(config.tank_diameter)

    if state.tank.mLiq_new > 0:
        m_v     = state.tank.mass_ox - state.tank.mLiq_new

        vl      = state.tank.mLiq_new/state.tank.ox_props.rho_l
        vv      = config.tnk_V - vl

        hl      = vl/tA
        hv      = vv/tA

        CoMl    = config.tnk_X - (hl / 2)
        CoMv    = config.tnk_X - hl - (hv / 2)
        CoMf    = config.cmbr_X - (config.prop.length / 2)

        state.mass.cg = (state.tank.mLiq_new*CoMl + m_v*CoMv + state.chamber.mass_fuel*CoMf + config.mass_properties.motor_mass*config.mass_properties.motor_cg) / state.mass.mass_total

    elif state.tank.mLiq_new == 0:
        m_v     = state.tank.mass_ox

        vv      = config.tnk_V

        hv      = vv / tA

        CoMv    = config.tnk_X - (hv / 2)
        CoMf    = config.cmbr_X - (config.prop.length / 2)

        state.mass.cg = (m_v*CoMv + state.chamber.mass_fuel*CoMf + config.mass_properties.motor_mass*config.mass_properties.motor_cg) / state.mass.mass_total
