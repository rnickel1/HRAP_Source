"""
Representation of the nozzle.
"""


class Nozzle:
    def __init__(self, throat_diam, efficiency, c_discharge):
        self.throat_diameter = throat_diam  # How to handle different units? Universal units array?
        self.efficiency = efficiency  # Comes in as a percentage
        self.c_discharge = c_discharge
        self.expansion_ratio = None  # Not sure what this is, also exit diameter
