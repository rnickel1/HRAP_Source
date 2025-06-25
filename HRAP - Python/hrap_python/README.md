# HRAP Python

## Purpose
This package provides a Python port of the HRAP utility, allowing users to simulate hybrid rocket engines without the need for a MATLAB license. The package exposes a core API which can be used to generate custom scripts, as well as a GUI for ease of use.

## Getting Started
To run a simulation, you'll first need to generate a `SimulationConfig` and `InitialConditions`, which are both pretty self-explanatory. A `SimulationConfig` is composed of several child components corresponding to the different parts of a hybrid engine: `PropellantConfig`, `HardwareConfig`, and `NozzleConfig`. `PropellantConfig` requires the creation of a `PropellantData` object, which loads tabulated propellant data from one of the `.npz` files in the `propellants` folder. All told, that looks like: 

```python
from api.core import sim_loop
from api.models import *
from api.sim import shift_OF

material = MaterialData('propellants/Paraffin.npz')
prop_config = PropellantConfig(ID=LengthValue(2.0, LengthUnit.INCHES),
                               OD=LengthValue(4.0, LengthUnit.INCHES),
                               length=LengthValue(12.0, LengthUnit.INCHES),
                               cstar_eff=1.0,
                               material=material,
                               regression_model=shift_OF)

hardware_config = HardwareConfig(tank_volume=VolumeValue(7000.0, VolumeUnit.CU_CENTIMETERS),
                                 chamber_volume=None,
                                 injector_Cd=0.6,
                                 injector_D=LengthValue(0.375, LengthUnit.INCHES),
                                 injector_N=1,
                                 vent_state=VentConfig.EXTERNAL,
                                 vent_Cd=0.6,
                                 vent_D=LengthValue(0.028, LengthUnit.INCHES))

nozzle_config = NozzleConfig(Cd=1.0,
                            throat=LengthValue(1.2, LengthUnit.INCHES),
                            exp_ratio=4.0,
                            efficiency=1.0)

sim_config = SimulationConfig(timestep=0.001, 
                              max_run_time=10.0,
                              max_burn_time=10.0,
                              ambient_pressure=PressureValue(1.0, PressureUnit.ATM),
                              hardware=hardware_config,
                              prop=prop_config,
                              nozzle=nozzle_config)
```

Initial conditions are generated in much the same way:
```python
init_conditions = InitialConditions(chamber_pressure=PressureValue(1.0, PressureUnit.ATM),
                                    mass_ox=MassValue(7.0, MassUnit.KILOGRAM),
                                    tank_temp=TemperatureValue(63.0, TemperatureUnit.FAHRENHEIT))
```

Finally, the simulation can be run by providing the constructed `SimulationConfig` and `InitialConditions` objects to `sim_loop`, which returns a `SimulationOutput` object containing all relevant simulation data.
```python
output = sim_loop(sim_config, init_conditions)
```

Outputs, much like inputs, will be given as a `*Value`. To convert between a list of `*Value`s to a list of conventional floats, simply use `get_as` in a list comprehension like so:
```python
thrust_data = [t.get_as(MassUnit.NEWTONS) for t in output.thrust]
```

The full code, along with a plotting example, is available in `demo.py`

## Testing
Unit tests are provided to ensure that behavior is well-defined and doesn't change version-to-version. Tests are located in the `test` module, and can be run from this directory with `python3 -m unittest discover -s test -v`. Some 'magic number' values in the unit tests are copied from the MATLAB output of the program for reference.