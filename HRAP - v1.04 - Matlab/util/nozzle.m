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
        Te              = x.T*(1 + (x.k-1)/2*M^2)^(-1);
        a               = sqrt(x.R*Te*x.k);
        Ve              = M*a;
        Pe              = x.P_cmbr*(1+(x.k-1)/2*M^2)^(-x.k/(x.k-1));
        x.F_thr         = s.noz_eff*x.mdot_n*Ve+0.25*pi*s.noz_thrt^2*s.noz_ER*(Pe - s.Pa);
        
        if x.F_thr < 0
            x.F_thr     = 0;
        end

    else
        x.F_thr         = 0;
    end
            
