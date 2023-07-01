%-----------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  const_OF
% 
% Purpose:  model regression of fuel grain at a given value of OF
%
%-----------------------------------------------------------------------------

function [x] = const_OF(s,x)

dt = s.dt;

    x.mdot_f        = x.mdot_o/s.const_OF;
    x.rdot          = x.mdot_f/(s.prop_Rho*pi* ...
                      x.grn_ID*s.grn_L);

    x.grn_ID_old    = x.grn_ID;
    x.grn_ID        = x.grn_ID+2*x.rdot*dt;
    x.m_f           = x.m_f - x.mdot_f*dt;
