# HRAP_Source
## Hybrid Rocket Analysis Program

For the MATLAB version, navigate to the "Installers/MATLAB version" directory. Prerequisites for the installation include having the MATLAB runtime installed (free.) and in order to save/export data the application needs to be run as administrator.

A Python version is also in progress but not complete yet.

The Hybrid Rocket Analysis Program (HRAP) was developed by Robert Nickel for use by the University of Tennessee Rocket Engineering Team (part of the Student Space Technology Association at UTK). HRAP is a versatile tool utilizing a thermodynamic equilibrium model for  simulation of self-pressurizing hybrid rocket motors, specifically those powered with Nitrous Oxide stored as a saturated liquid-vapor mixture. 

## Validation Cases:
Baltic Space HyPEx hot fire (r/rocketry user FlyingBanana)

![image](https://github.com/user-attachments/assets/4e048de8-12b3-4299-89f5-ec4241b3ccb2)

Equatorial Space Systems 750 N subscale demonstrator (James Anderson)

![image](https://github.com/user-attachments/assets/63330cf6-2fe3-4712-a0ec-310baa33e389)

Student Space Technology Association 38mm Subscale Motor

![image](https://github.com/user-attachments/assets/508ee5ae-4a46-4c41-8743-3ec823128cfe)

Student Space Technology Association 127mm Flight Motor

![image](https://github.com/user-attachments/assets/89afa8e9-32a9-4057-92d3-aaae7cde900f)

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
