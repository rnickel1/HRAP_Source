%-----------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  chamber
% 
% Purpose:  calculate chamber pressure for current timestep
%
%-----------------------------------------------------------------------------

function [x] = chamber(s,x)

dt = s.dt;

    if s.cmbr_V == 0
        V = 0.25*pi*x.grn_ID^2*s.grn_L;
    else
        V = s.cmbr_V - 0.25*pi*(s.grn_OD^2 - x.grn_ID^2)*s.grn_L;
    end
    
    dV = 0.25*pi*(x.grn_ID^2-x.grn_ID_old^2)*s.grn_L/dt;

    x.mdot_n = x.P_cmbr*s.noz_Cd*0.25*pi*s.noz_thrt^2/x.cstar;

    dm_g = x.mdot_f + x.mdot_o - x.mdot_n;
    
    if x.mdot_o == 0
        x.dm_g = -x.mdot_n;
    end

    x.m_g = x.m_g + dm_g*dt;

    dP = x.P_cmbr*(dm_g/x.m_g - dV/V);

    x.P_cmbr = x.P_cmbr + dP.*dt;

    if x.P_cmbr <= s.Pa
        x.P_cmbr = s.Pa;
        x.mdot_n = 0;
    end
end