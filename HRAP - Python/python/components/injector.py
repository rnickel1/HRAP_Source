"""
Injector!
"""


class Injector:
    def __init__(self, diameter, c_discharge, num_injectors, vent_state, vent_diameter=None, vent_c_discharge=None):
        self.diameter = diameter  # Default cm
        self.c_discharge = c_discharge
        self.num_injectors = num_injectors
        self.vent_state = vent_state
        self.vent_diameter = vent_diameter  # Default cm
        self.vent_c_discharge = vent_c_discharge

