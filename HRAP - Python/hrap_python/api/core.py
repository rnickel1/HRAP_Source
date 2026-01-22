"""
HRAP Simulation Core (`hrap_python.api.core`)

Core HRAP simulation functions
"""
from .models import *
from .sim import *

def sim_loop(config: SimulationConfig, init_conditions: InitialConditions) -> SimulationOutput:
    """
        Runs a simulation.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        init_conditions : InitialConditions
            The initial conditions for the simulation state

        Returns
        -------
        SimulationOutput
    """
    t = 0.0
    i = 1
    dt = config.timestep

    state = SimulationState(config, init_conditions)
    output = SimulationOutput(config)

    while True:
        t = (i-1)*dt
        i = i+1

        sim_iteration(config, state, output)

        if state.chamber.grain_ID >= config.prop.OD:
            output.end_condition = 'Fuel Depleted'
            break
        elif state.tank.mass_ox <= 0:
            output.end_condition = 'Oxidizer Depleted'
            break
        elif t >= config.max_run_time:
            output.end_condition = 'Max Simulation Time Reached'
            break
        elif state.chamber.pressure <= config.ambient_pressure:
            output.end_condition = 'Burn Complete'
            break
    
    return output

def sim_iteration(config: SimulationConfig, state: SimulationState, output: SimulationOutput) -> None:
    """
        A single iteration of a simulation. Contains all simulation state transitions for a given timestep.

        Parameters
        ----------
        config : SimulationConfig
            The simulation configuration parameters
        
        state : SimulationState
            The current simulation state

        output : SimulationOutput
            The tentative simulation output

        Returns
        -------
        None
    """
    tank(config, state)
    config.prop.regression_model(config, state)

    combustion(config, state)
    chamber(config, state)          
    nozzle(config, state)
    
    if config.mass_properties is not None:
        mass(config, state)

    output.append_state(state)
    state.time += config.timestep
