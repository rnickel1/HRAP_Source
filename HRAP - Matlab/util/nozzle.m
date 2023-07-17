%--------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  nozzle
% 
% Purpose:  calculate thrust for any given timestep
%
%--------------------------------------------------------------------------
function [x] = nozzle(s,x)

    if x.P_cmbr > s.Pa

        A_Ratio         = @(M) ((x.k+1)/2)^(-(x.k+1)/...
                          (2*(x.k-1)))*(1+(x.k-1)/2*M^2)^...
                          ((x.k+1)/(2*(x.k-1)))/M-...
                          s.noz_ER;
        M               = fzero(A_Ratio,3);
        Pe              = x.P_cmbr*(1+0.5*(x.k-1)*M^2)^(-x.k/(x.k-1));
        Cf              = sqrt(((2*x.k^2)/(x.k-1))*(2/(x.k+1))^((x.k+1)/...
                          (x.k-1))*(1-(Pe/x.P_cmbr)^((x.k-1)/x.k)))+...
                          ((Pe-s.Pa)*(0.25*pi*s.noz_thrt^2*s.noz_ER))/...
                          (x.P_cmbr*0.25*pi*s.noz_thrt^2);
        x.F_thr         = s.noz_eff*Cf*0.25*pi*s.noz_thrt^2*x.P_cmbr*...
                          s.noz_Cd;
        
        if x.F_thr < 0
            x.F_thr     = 0;
        end

    else
        x.F_thr         = 0;
    end