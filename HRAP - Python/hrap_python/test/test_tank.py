from api.models import *
from api.sim import tank, shift_OF
import unittest

class TestTankSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material = MaterialData('propellants/Paraffin.npz')
        cls.prop_config = PropellantConfig(ID=LengthValue(2.0, LengthUnit.INCHES),
                                    OD=LengthValue(4.0, LengthUnit.INCHES),
                                    length=LengthValue(12.0, LengthUnit.INCHES),
                                    cstar_eff=1.0,
                                    material=cls.material,
                                    regression_model=shift_OF)

        cls.nozzle_config = NozzleConfig(Cd=1.0,
                                    throat=LengthValue(1.2, LengthUnit.INCHES),
                                    exp_ratio=4.0,
                                    efficiency=1.0)

    def test_large_tank(self):
        hardware_config = HardwareConfig(tank_volume=VolumeValue(7000.0, VolumeUnit.CU_CENTIMETERS),
                                        chamber_volume=None,
                                        injector_Cd=0.6,
                                        injector_D=LengthValue(0.25, LengthUnit.INCHES),
                                        injector_N=1,
                                        vent_state=VentConfig.EXTERNAL,
                                        vent_Cd=0.6,
                                        vent_D=LengthValue(0.028, LengthUnit.INCHES))
        
        sim_config = SimulationConfig(timestep=0.001, 
                                    max_run_time=10.0,
                                    max_burn_time=10.0,
                                    ambient_pressure=PressureValue(1.0, PressureUnit.ATM),
                                    hardware=hardware_config,
                                    prop=self.prop_config,
                                    nozzle=self.nozzle_config)

        init_conditions = InitialConditions(chamber_pressure=PressureValue(1.0, PressureUnit.ATM),
                                            mass_ox=MassValue(5.0, MassUnit.KILOGRAMS),
                                            tank_temp=TemperatureValue(293.15, TemperatureUnit.KELVIN))
        state = SimulationState(sim_config, init_conditions)
        tank(sim_config, state)
        self.assertAlmostEqual(state.tank.mass_ox, 4.9983216, delta=1e-6)
        self.assertAlmostEqual(state.tank.mLiq_new, 4.8706516, delta=1e-6)
        self.assertAlmostEqual(state.tank.mLiq_old, 4.8710643, delta=1e-6)

    def test_small_tank(self):
        hardware_config = HardwareConfig(tank_volume=VolumeValue(2000.0, VolumeUnit.CU_CENTIMETERS),
                                        chamber_volume=None,
                                        injector_Cd=0.6,
                                        injector_D=LengthValue(0.06125, LengthUnit.INCHES),
                                        injector_N=1,
                                        vent_state=VentConfig.EXTERNAL,
                                        vent_Cd=0.6,
                                        vent_D=LengthValue(0.028, LengthUnit.INCHES))
        
        sim_config = SimulationConfig(timestep=0.001, 
                                    max_run_time=10.0,
                                    max_burn_time=10.0,
                                    ambient_pressure=PressureValue(1.0, PressureUnit.ATM),
                                    hardware=hardware_config,
                                    prop=self.prop_config,
                                    nozzle=self.nozzle_config)

        init_conditions = InitialConditions(chamber_pressure=PressureValue(1.0, PressureUnit.ATM),
                                            mass_ox=MassValue(1.5, MassUnit.KILOGRAMS),
                                            tank_temp=TemperatureValue(293.15, TemperatureUnit.KELVIN))
        state = SimulationState(sim_config, init_conditions)
        tank(sim_config, state)
        self.assertAlmostEqual(state.tank.mass_ox, 1.4998993, delta=1e-6)
        self.assertAlmostEqual(state.tank.mLiq_new, 1.4814849, delta=1e-6)
        self.assertAlmostEqual(state.tank.mLiq_old, 1.4815008, delta=1e-6)
        

if __name__ == '__main__':
    unittest.main()