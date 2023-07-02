# HRAP_Source
## Hybrid Rocket Analysis Program

For the MATLAB version, navigate to the "Installers/MATLAB version" directory. Prerequisites for the installation include having the MATLAB runtime installed (free.) and in order to save/export data the application needs to be run as administrator.

A Python version is also in progress but not complete yet.

The Hybrid Rocket Analysis Program (HRAP) was developed by Robert Nickel for use by the University of Tennessee Rocket Engineering Team. HRAP is a versatile tool utilizing a thermodynamic equilibrium model for  simulation of self-pressurizing hybrid rocket motors, especially those powered with Nitrous Oxide stored as a saturated liquid-vapor mixture. 

## Call for volunteers!

In its current state this program can model an adiabatic oxidizer tank and combustion chamber and an isentropic nozzle. However, this could be significantly improved by some helpful volunteers in the following categories: 
- Non-equilibrium tank model
- Two-phase injector model
- Compressible real gas model for vapor discharge
- Improved nozzle model
- Non-cylindrical ports
- Option to export a .eng engine file for use in OpenRocket
- Use of other self-pressurizing oxidizers such as Nytrox
- More config files describing burn characteristics of different propellants
Future iterations will account for subsonic flow and flow separation to better model thrust, add an option to export a .eng or .rse engine file for use in OpenRocket or RockSim, and will allow the use of other self-pressurizing oxidizers such as Nytrox. Development is also needed for more configuration files describing burn characteristics of different propellants.