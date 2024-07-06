"""
units.py

Utilities for converting between common units
"""
from enum import Enum
from math import pi

def d_to_a(d: float) -> float:
    return 0.25 * pi * (d ** 2)

# Converts total impulse to motor class and some percentage
def impulse_to_class_percent(impulse: float) -> tuple[str, float]:
    if impulse <= 1.25:
        return "A", 100.0 * impulse / 1.25
    elif impulse <= 5:
        return "B", 100.0 * (impulse - 1.25) / 2.5
    elif impulse <= 10:
        return "C", 100.0 * (impulse - 5) / 5
    elif impulse <= 20:
        return "D", 100.0 * (impulse - 10) / 10
    elif impulse <= 40:
        return "E", 100.0 * (impulse - 20) / 20
    elif impulse <= 80:
        return "F", 100.0 * (impulse - 40) / 40
    elif impulse <= 160:
        return "G", 100.0 * (impulse - 80) / 80
    elif impulse <= 320:
        return "H", 100.0 * (impulse - 160) / 160
    elif impulse <= 640:
        return "I", 100.0 * (impulse - 320) / 320
    elif impulse <= 1280:
        return "J", 100.0 * (impulse - 640) / 640
    elif impulse <= 2560:
        return "K", 100.0 * (impulse - 1280) / 1280
    elif impulse <= 5120:
        return "L", 100.0 * (impulse - 2560) / 2560
    elif impulse <= 10240:
        return "M", 100.0 * (impulse - 5120) / 5120
    elif impulse <= 20480:
        return "N", 100.0 * (impulse - 10240) / 10240
    elif impulse <= 40960:
        return "O", 100.0 * (impulse - 20480) / 20480
    elif impulse <= 81920:
        return "P", 100.0 * (impulse - 40960) / 40960
    elif impulse <= 163840:
        return "Q", 100.0 * (impulse - 81920) / 81920
    elif impulse <= 327680:
        return "R", 100.0 * (impulse - 163840) / 163840
    elif impulse <= 655360:
        return "S", 100.0 * (impulse - 327680) / 327680
    elif impulse <= 1310720:
        return "T", 100.0 * (impulse - 655360) / 655360
    else:
        return None, None

class LengthUnit(Enum):
    METERS = 1
    CENTIMETERS = 2
    MILLIMETERS = 3
    FEET = 4
    INCHES = 5

class LengthValue:
    CONVERSIONS = {
        LengthUnit.METERS: 1.0,
        LengthUnit.CENTIMETERS: 0.01,
        LengthUnit.MILLIMETERS: 1e-3,
        LengthUnit.FEET: 0.3048,
        LengthUnit.INCHES: 0.0254
    }

    def __init__(self, value: float, unit: LengthUnit) -> None:
        self.base_value = value * LengthValue.CONVERSIONS[unit]

    def get_as(self, unit: LengthUnit) -> float:
        return self.base_value / LengthValue.CONVERSIONS[unit]

class VolumeUnit(Enum):
    CU_METERS = 1
    CU_CENTIMETERS = 2
    LITERS = 3
    CU_FEET = 4
    CU_INCHES = 5
    GALLONS = 6

class VolumeValue:
    CONVERSIONS = {
        VolumeUnit.CU_METERS: 1.0,
        VolumeUnit.CU_CENTIMETERS: 1e-6,
        VolumeUnit.LITERS: 1e-3,
        VolumeUnit.CU_FEET: 0.3048 ** 3,
        VolumeUnit.CU_INCHES: 0.0254 ** 3,
        VolumeUnit.GALLONS: 3.78541e-3
    }

    def __init__(self, value: float, unit: VolumeUnit) -> None:
        self.base_value = value * VolumeValue.CONVERSIONS[unit]

    def get_as(self, unit: VolumeUnit) -> float:
        return self.base_value / VolumeValue.CONVERSIONS[unit]


class PressureUnit(Enum):
    PA = 1
    KPA = 2
    MPA = 3
    ATM = 4
    BAR = 5
    PSI = 6
    PSF = 7

class PressureValue:
    CONVERSIONS = {
        PressureUnit.PA: 1.0,
        PressureUnit.KPA: 1e3,
        PressureUnit.MPA: 1e6,
        PressureUnit.ATM: 101325,
        PressureUnit.BAR: 1e5,
        PressureUnit.PSI: 101325/14.696,
        PressureUnit.PSF: 101325/14.686*144 
    }

    def __init__(self, value: float, unit: PressureUnit) -> None:
        self.base_value = value * PressureValue.CONVERSIONS[unit]

    def get_as(self, unit: PressureUnit) -> float:
        return self.base_value / PressureValue.CONVERSIONS[unit]

class DensityUnit(Enum):
    KG_CU_METER = 1
    GRAM_CU_CENTIMETER = 2
    POUND_CU_INCH = 3
    POUND_CU_FOOT = 4

class DensityValue:
    CONVERSIONS = {
        DensityUnit.KG_CU_METER: 1.0,
        DensityUnit.GRAM_CU_CENTIMETER: 1e3,
        DensityUnit.POUND_CU_INCH: 1.0 / (2.205 * (0.0254**3)),
        DensityUnit.POUND_CU_FOOT: 1.0 / (2.205 * (0.3048**3))
    }

    def __init__(self, value: float, unit: DensityUnit) -> None:
        self.base_value = value * DensityValue.CONVERSIONS[unit]

    def get_as(self, unit: DensityUnit) -> float:
        return self.base_value / DensityValue.CONVERSIONS[unit]

class MassUnit(Enum):
    KILOGRAMS = 1
    GRAMS = 2
    POUNDS = 3
    OUNCES = 4
    NEWTONS = 5 # Included for simplicity - no need to separate mass and force

class MassValue:
    CONVERSIONS = {
        MassUnit.KILOGRAMS: 1.0,
        MassUnit.GRAMS: 1e-3,
        MassUnit.POUNDS: 0.453592,
        MassUnit.OUNCES: 0.0283495,
        MassUnit.NEWTONS: 0.10197
    }

    def __init__(self, value: float, unit: MassUnit) -> None:
        self.base_value = value * MassValue.CONVERSIONS[unit]

    def get_as(self, unit: MassUnit) -> float:
        return self.base_value / MassValue.CONVERSIONS[unit]

class TemperatureUnit(Enum):
    KELVIN = 1
    CELSIUS = 2
    RANKINE = 3
    FAHRENHEIT = 4

class TemperatureValue:
    def __init__(self, value: float, unit: TemperatureUnit) -> None:
        # These conversions are done manually due to the addition/subtraction in Fahrenheit conversions
        if unit == TemperatureUnit.CELSIUS:
            self.base_value = value + 273.15
        elif unit == TemperatureUnit.RANKINE:
            self.base_value = value / 1.8
        elif unit == TemperatureUnit.FAHRENHEIT:
            self.base_value = ((value - 32) / 1.8) + 273.15
        else:
            self.base_value = value

    def get_as(self, unit: TemperatureUnit) -> float:
        if unit == TemperatureUnit.CELSIUS:
            return self.base_value - 273.15
        elif unit == TemperatureUnit.RANKINE:
            return self.base_value * 1.8
        elif unit == TemperatureUnit.FAHRENHEIT:
            return ((self.base_value - 273.15) * 1.8) + 32
        else:
            return self.base_value
    