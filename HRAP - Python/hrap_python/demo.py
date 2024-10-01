from api.core import sim_loop
from api.models import *
from api.sim import shift_OF
from util.units import *
import matplotlib.pyplot as plt

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

init_conditions = InitialConditions(chamber_pressure=PressureValue(1.0, PressureUnit.ATM),
                                    mass_ox=MassValue(7.0, MassUnit.KILOGRAMS),
                                    tank_temp=TemperatureValue(63.0, TemperatureUnit.FAHRENHEIT))

output = sim_loop(sim_config, init_conditions)

time_data = output.time
thrust_data = [t.get_as(MassUnit.NEWTONS) for t in output.thrust]
plt.plot(time_data, thrust_data)
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.title('Predicted Thrust')
plt.show()
