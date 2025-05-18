# HRAP_Source
The Hybrid Rocket Analysis Program (HRAP) was developed by Robert Nickel for use by the University of Tennessee Rocket Engineering Team. HRAP is a versatile tool utilizing a thermodynamic equilibrium model for  simulation of self-pressurizing hybrid rocket motors, specifically those powered with Nitrous Oxide stored as a saturated liquid-vapor mixture. 

## Python - Hybrid Rocket Analysis Program
Navigate to the "HRAP - Python" directory for installation and usage instructions.

## MATLAB - Hybrid Rocket Analysis Program
Navigate to the "Installers/MATLAB version" directory. Prerequisites for the installation include having the MATLAB runtime installed (free.) and in order to save/export data the application needs to be run as administrator.

## Call for volunteers!

In its current state this program can model an adiabatic oxidizer tank and combustion chamber and an isentropic nozzle, along with semi-empirical correction factors such as combustion/nozzle efficiency. HRAP also allows the exporting of a .RSE engine file for use in OpenRocket or RockSim, which allows not only the mass of the motor to be captured but also the approximate center of mass of the motor.

HRAP could be significantly improved by some helpful volunteers in the following categories: 
- Non-equilibrium tank model
- Two-phase injector model
- Non-cylindrical ports
- Option to export a .eng engine file for use in OpenRocket
- More config files describing burn characteristics of different propellants
- Potentially a shift from look-up tables of propellant data to integrating NASA's CEA tool into the combustion modeling
- integrating CoolProp for thermodynamic modeling
