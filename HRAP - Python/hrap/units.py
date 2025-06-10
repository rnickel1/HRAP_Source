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

_lbm = 0.4535924
_lbf  = 4.44822162

_Pa = 1.0
_kPa = 1E3
_atm = 101325
_psi = 6894.757

unit_conversions = {
    'length':  {'mm': _mm, 'cm': _cm, 'm': _m, 'in': _in, 'ft': _ft},
    'volume':  {'cc': _cc, 'L': _L, 'm^3': _m3, 'gal': _gal},
    'pressure': {'kPa': _kPa, 'atm': _atm, 'psi': _psi},
    # 'temperature': {'deg C': , 'K': 1.0, 'deg F': }
}

def get_unit_type(unit):
    for unit_type, units in unit_conversions.items():
        if unit in units: return unit_type
    return None