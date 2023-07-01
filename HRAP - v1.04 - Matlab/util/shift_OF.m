%-----------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  shift_OF
% 
% Purpose:  model regression of fuel grain using exponential regression law
%
%-----------------------------------------------------------------------------

function [x] = shift_OF(s,x)

dt = s.dt;

    A               = 0.25*pi*x.grn_ID^2;
    G               = x.mdot_o/A;
    x.rdot          = 0.001*s.prop_Reg(1)*G^s.prop_Reg(2)*...
                      s.grn_L^s.prop_Reg(3);
    x.mdot_f     = s.prop_Rho*x.rdot*pi*x.grn_ID* ...
                      s.grn_L;
    x.OF            = x.mdot_o/x.mdot_f;
    
    if x.mdot_f == 0
    x.OF            = 0;
    end

    x.grn_ID_old   = x.grn_ID;
    x.grn_ID       = x.grn_ID+2*x.rdot*dt;
    x.m_f          = x.m_f - x.mdot_f*dt;
