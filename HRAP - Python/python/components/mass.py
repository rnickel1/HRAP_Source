"""
Calculating the mass properties (optional field)
"""


class MassProperties:

    def __init__(self, ox_tank_location, fuel_grain_location, empty_motor_com, empty_motor_mass):
        self.ox_tank_loc = ox_tank_location  # Default cm, measured from fore
        self.fuel_grain_loc = fuel_grain_location  # Default cm, measured from fore
        self.empty_motor_com = empty_motor_com  # Default cm, measured from fore
        self.empty_motor_mass = empty_motor_mass  # Default kg
