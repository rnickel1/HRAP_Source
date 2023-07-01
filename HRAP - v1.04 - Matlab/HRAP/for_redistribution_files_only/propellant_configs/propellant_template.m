%--------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  propellant_template
% 
% Purpose:  Use this script to generate new propellant configurations for
%           HRAP. Change the propellant name in the last line.
%
%--------------------------------------------------------------------------

%Input 1xm array of chamber pressure values in pascals
s.prop_Pc = [100000 1000000 2000000 3000000 4000000];

%Input nx1 array of OF ratio values
s.prop_OF = [1 2 3 4]';

%input nxm array for specific heat ratio where n is the length of OF and m is the length of Pc
s.prop_k = [1.4 1.4 1.4 1.4 1.4
            1.4 1.4 1.4 1.4 1.4
            1.4 1.4 1.4 1.4 1.4
            1.4 1.4 1.4 1.4 1.4];

%input nxm array for gas molecular mass in g/mol where n is the length of OF and m is the length of Pc
s.prop_M = [28.97   28.97   28.97   28.97   28.97
            28.97   28.97   28.97   28.97   28.97
            28.97   28.97   28.97   28.97   28.97
            28.97   28.97   28.97   28.97   28.97];

%input nxm array for adiabatic flame temperature where n is the length of OF and m is the length of Pc in Kelvin
s.prop_T = [3000 3000 3000 3000 3000
            3000 3000 3000 3000 3000
            3000 3000 3000 3000 3000
            3000 3000 3000 3000 3000];

%name the propellant
s.prop_nm = 'hot air';

%Input regression coefficients as [a,n,m], if neglecting m, set m to zero, proper units for G, L and rdot are kg/m^2/s, m, and mm/s
s.prop_Reg = [0.2, 0.6, 0];

%input propellant grain density in kg/m^3
s.prop_Rho = 1000;

%type in optimum OF ratio here
s.opt_OF = 2.5;

save('propellant_name.mat')