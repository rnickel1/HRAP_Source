"""
HRAP Data Models (`hrap_python.api.models`)

Configuration and state models for HRAP simulations
"""
from enum import Enum
from util.units import *
from util.nox import *
from scipy.optimize import fsolve
from typing import List
import numpy as np

###
### CONFIGURATION MODELS
###
class VentConfig(Enum):
    NONE = 1
    EXTERNAL = 2
    INTERNAL = 3

class MaterialData:
    # TODO: Better way of loading file?
    def __init__(self, filename: str):
        container = np.load(filename) 
        for label in ['metadata', 'regression_coeff', 'OF', 'Pc', 'k', 'M', 'T']:
            assert label in container, f'Data for "{label}" not found in saved combustion data file {filename}!'
        # Metadata can be extended to include other mixed data - although it's unclear whether NoneTypes are supported
        self.name   = container['metadata'][0]
        self.rho    = float(container['metadata'][1])
        # All coefficients will be negative (typically -1) if a shifting-OF model is not supported
        # 0 is not reliable due to floating-point precision errors (may become positive or negative)
        self.regression_coeff = container['regression_coeff']
        self.OF     = container['OF']
        self.Pc     = container['Pc']
        self.k      = container['k']
        self.M      = container['M']
        self.T      = container['T']

class PropellantConfig:
    def __init__(self, ID: LengthValue, OD: LengthValue, length: LengthValue, cstar_eff: float, 
                 material: MaterialData, regression_model, const_OF_ratio: float | None = None):
        self.ID     = ID.base_value         # Grain ID (m)
        self.OD     = OD.base_value         # Grain OD (m)
        self.length = length.base_value     # Grain Length (m)

        self.cstar_eff  = cstar_eff # C-star efficiency
        self.material   = material  # Material Data - imported from file

        assert ((material.regression_coeff[0] > 0 and const_OF_ratio is None)
                or (const_OF_ratio is not None and const_OF_ratio > 0)), "Material must support shifting OF ratio, or constant OF ratio must be provided!"
    
        self.regression_model = regression_model
        self.const_OF_ratio = const_OF_ratio # Constant OF ratio (optional)

class HardwareConfig:
    def __init__(self, tank_volume: VolumeValue, chamber_volume: VolumeValue | None, injector_Cd: float, injector_D: LengthValue,
                 injector_N: int, vent_state: VentConfig, vent_Cd: float | None = None, vent_D: LengthValue | None = None):
        self.tank_volume = tank_volume.base_value       # Tank Volume (m^3)
        if chamber_volume is None:
            self.chamber_volume = None
        else:
            self.chamber_volume = chamber_volume.base_value # Chamber Volume (m^3) - if None, computed from grain dimensions

        self.injector_cda_N = injector_Cd * d_to_a(injector_D.base_value) * injector_N

        assert vent_state == VentConfig.NONE or (vent_Cd is not None and vent_D is not None), "Must input valid vent configuration!"
        self.vent_state = vent_state
        self.vent_cda = vent_Cd * d_to_a(vent_D.base_value)

class NozzleConfig:
    def __init__(self, Cd: float, throat: LengthValue, exp_ratio: float, efficiency: float):
        self.Cd = Cd                    # Coefficient of dispersion
        self.throat = throat.base_value # Throat diameter (m)
        self.exp_ratio = exp_ratio      # Expansion Ratio (unitless)
        self.efficiency = efficiency    # Efficiency (>0, <1.0)

class MassProperties:
    def __init__(self, motor_mass: MassValue, motor_cg: LengthValue, tank_location: LengthValue, grain_location: LengthValue):
        self.motor_mass = motor_mass.base_value         # Motor Mass (kg) 
        self.motor_cg = motor_cg.base_value             # Motor center of gravity (m)
        self.tank_location = tank_location.base_value   # Tank center of gravity (m)
        self.grain_location = grain_location.base_value # Grain center of gravity (m)

class SimulationConfig:
    def __init__(self, timestep: float, max_run_time: float, max_burn_time: float, ambient_pressure: PressureValue,
                 hardware: HardwareConfig, prop: PropellantConfig, nozzle: NozzleConfig, mass_properties: MassProperties | None = None):
        self.timestep = timestep
        self.max_run_time = max_run_time
        self.max_burn_time = max_burn_time
        self.ambient_pressure = ambient_pressure.base_value

        grain_volume = d_to_a(prop.OD) * prop.length
        if hardware.chamber_volume is None:
            hardware.chamber_volume = grain_volume
        
        assert hardware.chamber_volume >= grain_volume, "Chammber cannot be smaller than fuel grain!"

        self.hardware = hardware
        self.prop = prop
        self.nozzle = nozzle
        self.mass_properties = mass_properties

###
### STATE MODELS
###
class InitialConditions:
    def __init__(self, chamber_pressure: PressureValue, mass_ox: MassValue, tank_pressure: PressureValue):
        self.chamber_pressure = chamber_pressure.base_value
        self.mass_ox = mass_ox.base_value
        self.tank_temp = None
        self.tank_pressure = tank_pressure.base_value

    def __init__(self, chamber_pressure: PressureValue, mass_ox: MassValue, tank_temp: TemperatureValue):
        self.chamber_pressure = chamber_pressure.base_value
        self.mass_ox = mass_ox.base_value
        self.tank_temp = tank_temp.base_value
        self.tank_pressure = None

class TankState:
    def __init__(self, config: SimulationConfig, init_conditions: InitialConditions):
        if init_conditions.tank_temp is None:
            self.temp = fsolve(lambda T: vapor_pressure(T) - init_conditions.tank_pressure, 273.15)[0] # TODO: Error handling (see RootResults), root method selection
        else:
            self.temp = init_conditions.tank_temp
        self.ox_props = get_NOX_properties(self.temp)
        self.mass_ox = init_conditions.mass_ox
        self.mass_ox_old = self.mass_ox
        self.mdot_ox = 0.0
        self.mdot_vapor = 0.0
        self.pressure = self.ox_props.Pv

        self.mLiq_new = TankState.liquid_mass(config.hardware.tank_volume, self.mass_ox, self.ox_props)
        self.mLiq_old = self.mLiq_new + 1.0

    def liquid_mass(tank_vol: float, mass_ox: float, ox_props: NOXProperties) -> float:
        # This can take in standard floats because unit normalization should have already been done... should have.
        return ((tank_vol - (mass_ox / ox_props.rho_v)) / 
                ((1 / ox_props.rho_l) - (1 / ox_props.rho_v)))

class ChamberState:
    AIR_RHO = 1.225 # Density of air (kg/m^3)
    
    def __init__(self, config: SimulationConfig, init_conditions: InitialConditions):
        self.pressure = init_conditions.chamber_pressure
        
        grain_volume    = (d_to_a(config.prop.OD) - d_to_a(config.prop.ID)) * config.prop.length
        self.mass_fuel  = grain_volume * config.prop.material.rho
        self.mass_gas   = ChamberState.AIR_RHO * (config.hardware.chamber_volume - grain_volume) 
        
        self.vdot_fuel = 0.0
        self.mdot_fuel = 0.0
        self.mdot_exit = 0.0
        self.rdot = 0.0

        self.grain_ID = config.prop.ID
        self.OF_ratio = config.prop.const_OF_ratio if config.prop.const_OF_ratio is not None else 0.0

class MassState:
    def __init__(self, config: SimulationConfig, init_conditions: InitialConditions):
        self.cg = 0.0
        self.mass_total = 0.0

class CombustionState:
    def __init__(self):
        self.k = 0.0
        self.T = 0.0
        self.M = 0.0
        self.R = 0.0
        self.rho = 0.0
        self.cstar = 0.0

class SimulationState:
    def __init__(self, config: SimulationConfig, init_conditions: InitialConditions):
        self.time = 0.0
        
        self.tank = TankState(config, init_conditions)
        self.chamber = ChamberState(config, init_conditions)
        self.combustion = CombustionState()
        self.mass = MassState(config, init_conditions)
        self.thrust = 0.0
        self.pressure_drop = 0.0

        # For interpolating oxidizer pressure curve
        self.total_vapor_dP = 0.0
        self.interpolation_timesteps = 1e-10 # Avoids potential divide-by-zero

###
### SIMULATION OUTPUT
###
class SimulationOutput:
    def __init__(self, config: SimulationConfig):
        self.time : List[float] = []

        self.mass_ox : List[MassValue] = []
        self.mdot_ox : List[MassValue] = []
        self.tank_pressure : List[PressureValue] = []

        self.mass_fuel : List[MassValue] = []
        self.mdot_fuel : List[MassValue] = []
        self.mdot_exit : List[MassValue] = []
        self.rdot : List[LengthValue] = []
        self.chamber_pressure : List[PressureValue] = []
        self.OF_ratio : List[float] = []
        self.grain_ID : List[LengthValue] = []
        
        self.pressure_drop : List[PressureValue] = []
        self.thrust : List[MassValue] = []

        if config.mass_properties is not None:
            self.total_mass : List[MassValue] = []
            self.cg : List[LengthValue] = []
            self.include_mass_props = True
        else:
            self.include_mass_props = False

        self.end_condition : str = "Simulation Unfinished"
    
    def append_state(self, state: SimulationState):
        self.time.append(state.time)

        # TODO: Mass flux / length 'flux' unit? Probably not necessary - denominator is always 'seconds'
        self.mass_ox.append(MassValue(state.tank.mass_ox, MassUnit.KILOGRAMS))
        self.mdot_ox.append(MassValue(state.tank.mdot_ox, MassUnit.KILOGRAMS))
        self.tank_pressure.append(PressureValue(state.tank.pressure, PressureUnit.PA))

        self.mass_fuel.append(MassValue(state.chamber.mass_fuel, MassUnit.KILOGRAMS))
        self.mdot_fuel.append(MassValue(state.chamber.mdot_fuel, MassUnit.KILOGRAMS))
        self.mdot_exit.append(MassValue(state.chamber.mdot_exit, MassUnit.KILOGRAMS))
        self.rdot.append(LengthValue(state.chamber.rdot, LengthUnit.MILLIMETERS)) # Regression is in mm/s by default
        self.chamber_pressure.append(PressureValue(state.chamber.pressure, PressureUnit.PA))
        self.OF_ratio.append(state.chamber.OF_ratio)
        self.grain_ID.append(LengthValue(state.chamber.grain_ID, LengthUnit.METERS))

        self.pressure_drop.append(PressureValue(state.pressure_drop, PressureUnit.PA))
        self.thrust.append(MassValue(state.thrust, MassUnit.NEWTONS))

        if self.include_mass_props:
            self.total_mass.append(MassValue(state.mass.mass_total, MassUnit.KILOGRAMS))
            self.cg.append(LengthUnit(state.mass.cg, LengthUnit.METERS))
    
    def max_thrust(self):
        return MassValue(max([t.base_value for t in self.thrust], MassUnit.NEWTONS))