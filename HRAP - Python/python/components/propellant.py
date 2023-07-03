"""
Propellant. This can be either loaded in or taken from UI
"""


class Propellant:

    def __init__(self, name, density, c_star_eff, grain_id, grain_od,
                 grain_length, const_of_ratio=None, c_regression=None, exp_regression=None, exp_length=None):
        self.name = name
        self.density = density  # Default kg/m^3
        self.const_of_ratio = const_of_ratio  # Constant OF only
        self.c_regression = c_regression  # mm/s, Regression only
        self.exp_regression = exp_regression  # kg/m^2-s, Regression only
        self.exp_length = exp_length  # m, Regression only
        self.c_star_eff = c_star_eff  # C* efficiency, percent
        self.grain_id = grain_id  # Default cm
        self.grain_od = grain_od  # Default cm
        self.grain_length = grain_length  # Default cm


def load_from_csv(filepath):
    # You can also load from CSV! headers should be:
    # name,density,const_of,const_of_ratio,c_star_eff,grain_id,grain_od,grain_length
    return Propellant()
