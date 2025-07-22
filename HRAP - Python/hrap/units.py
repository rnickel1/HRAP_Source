# Purpose: Provide unit conversions to and from SI units
# Authors: Thomas A. Scott

# Internal units:
#   Length      - meters
#   Time        - seconds
#   Temperature - Kelvin
#   Pressure    - Pascals

_mm = 1E-3
_cm = 1E-2
_in = 0.0254
_ft = _in*12
_m = 1.0

_cc = _cm**3
_L = 1E-3
_m3 = 1.0
_gal = 0.00378541

_Pa = 1.0
_kPa = 1E3
_atm = 101325
_psi = 6895.0

_g = 1E-3
_kg = 1.0
_lbm = 0.4536

_N = 1.0
_kN = 1E3
_lbf  = 4.448

unit_conversions = {
    'length':  {'mm': _mm, 'cm': _cm, 'm': _m, 'in': _in, 'ft': _ft},
    'volume':  {'cc': _cc, 'L': _L, 'm^3': _m3, 'gal': _gal},
    'pressure': {'kPa': _kPa, 'atm': _atm, 'psi': _psi},
    'mass': {'g': _g, 'kg': _kg, 'lbm': _lbm},
    'force': {'N': _N, 'kN': _kN, 'lbf': _lbf},
    # 'temperature': {'deg C': , 'K': 1.0, 'deg F': }
}
inv_unit_conversions = {
    'length':  {},
    'volume':  {},
    'pressure': {},
    'mass': {},
    'force': {},
}
for unit_type, units in unit_conversions.items():
    for unit, val in units.items():
        # print(unit, isinstance(val, float), type(val), unit in inv_unit_conversions[unit_type])
        if isinstance(val, float) and not unit in inv_unit_conversions[unit_type]:
            # print('AAA')
            inv_unit_conversions[unit_type][unit] = lambda x, f=val: x/f
for unit_type, units in unit_conversions.items():
    for unit, val in units.items():
        if isinstance(val, float):
            unit_conversions[unit_type][unit] = lambda x, f=val: x*f

def get_unit_type(unit):
    for unit_type, units in unit_conversions.items():
        if unit in units: return unit_type
    return None
