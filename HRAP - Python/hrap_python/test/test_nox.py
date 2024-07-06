from util.nox import *
import unittest

class TestNOXOutputs(unittest.TestCase):
    TEMPLATE = "Value %s is not within acceptable bounds at temperature %.2fK!"
    def test_nominal_value(self):
        T = 273.15
        props = get_NOX_properties(T)
        self.assertAlmostEqual(props.Pv, 3.1266e6, delta=10)
        self.assertAlmostEqual(props.rho_l, 907.407, delta=1e-3)
        self.assertAlmostEqual(props.rho_v, 84.8624, delta=1e-4)
        self.assertAlmostEqual(props.Hv, 232.19, delta=1e-3)
        self.assertAlmostEqual(props.Cp, 2.2741, delta=1e-5)
        self.assertAlmostEqual(props.Z, 0.714004, delta=1e-6)

    def test_max_value(self):
        T = 309.0
        props = get_NOX_properties(T)
        self.assertAlmostEqual(props.Pv, 7.162454e6, delta=10)
        self.assertAlmostEqual(props.rho_l, 551.81, delta=1e-3)
        self.assertAlmostEqual(props.rho_v, 367.76, delta=1e-3)
        self.assertAlmostEqual(props.Hv, 45.9456, delta=1e-4)
        self.assertAlmostEqual(props.Cp, 34.3239, delta=1e-4)
        self.assertAlmostEqual(props.Z, 0.333644, delta=1e-6)
    
    def test_min_value(self):
        T = 183.15
        props = get_NOX_properties(T)
        self.assertAlmostEqual(props.Pv, 9.2287e4, delta=10)
        self.assertAlmostEqual(props.rho_l, 1220.6, delta=1)
        self.assertAlmostEqual(props.rho_v, 2.73844, delta=1e-5)
        self.assertAlmostEqual(props.Hv, 376.51, delta=1e-3)
        self.assertAlmostEqual(props.Cp, 1.75001, delta=1e-5)
        self.assertAlmostEqual(props.Z, 0.974032, delta=1e-6)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            get_NOX_properties(310)
        with self.assertRaises(ValueError):
            get_NOX_properties(183)

if __name__ == '__main__':
    unittest.main()