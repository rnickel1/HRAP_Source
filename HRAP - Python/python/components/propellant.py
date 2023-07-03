"""
Propellant. This can be either loaded in or taken from UI
"""


class Propellant:

    def __init__(self, name, density, const_of, const_of_ratio, c_star_eff, grain_id, grain_od, grain_length):
        self.name = name
        self.density = density  # Default kg/m^3
        if const_of:  # Regression model is Constant OF
            self.const_of_ratio = const_of_ratio
        else:  # Regression model is Shifting OF
            self.const_of_ratio = None
        self.c_star_eff = c_star_eff  # C* efficiency, percent
        self.grain_id = grain_id  # Default cm
        self.grain_od = grain_od  # Default cm
        self.grain_length = grain_length  # Default cm


def load_from_csv(filepath):
    return Propellant()
