"""
Simulation configuration (contains all other models)
"""


class SimulationConfiguration:
    def __init__(self, tank_temp, chamber_pressure, oxidizer_mass, ambient_press,
                 max_sim_runtime, max_burn_time, sim_timestep, regression_model):
        self.tank_temp = tank_temp  # Default K. TODO: account for ppl inputting pressure instead
        self.chamber_pressure = chamber_pressure  # Default atm
        self.starting_ox_mass = oxidizer_mass  # Default kg. TODO: account for tank fill percentage input
        self.ambient_pressure = ambient_press  # Default atm

        self.max_sim_runtime = max_sim_runtime  # seconds
        self.max_burn_time = max_burn_time  # seconds
        self.sim_timestep = sim_timestep  # seconds
        self.regression_model = regression_model

