%--------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  comb
% 
% Purpose:  interpolate gas properties at current chamber pressure and
%           OF ratio
%
%--------------------------------------------------------------------------

function [x] = comb(s,x,t)

    if t <= s.tburn || s.tburn == 0
        x.k     = interp2x(s.prop_OF,s.prop_Pc,s.prop_k,x.OF,x.P_cmbr);
        x.M     = interp2x(s.prop_OF,s.prop_Pc,s.prop_M,x.OF,x.P_cmbr);
        x.T     = interp2x(s.prop_OF,s.prop_Pc,s.prop_T,x.OF,x.P_cmbr);
        x.R = 8314.5/x.M;
        x.rho = x.P_cmbr/(x.R*x.T);
        x.cstar = s.cstar_eff*sqrt((x.R*x.T)/(x.k*(2/(x.k+1))^((x.k+1)/(x.k-1))));
    end
