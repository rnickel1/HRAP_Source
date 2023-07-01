%-----------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  sim_loop
% 
% Purpose:  run HRAP Simulation Environment to predict the performance of a
%           hybrid rocket motor
%
%-----------------------------------------------------------------------------

function [s,x,o,t] = sim_loop(s,x,o,t)

i = 1;
dt = s.dt;

    while true

        t = (i-1)*dt;
        i = i+1;

        [s,x,o,t] = sim_iteration(s,x,o,t,i);

        if x.grn_ID >= s.grn_OD
            o.sim_end_cond = 'Fuel Depleted';
            break
        elseif x.m_o <= 0
            o.sim_end_cond = 'Oxidizer Depleted';
            break
        elseif t >= s.tmax
            o.sim_end_cond = 'Max Simulation Time Reached';
            break
        elseif x.P_cmbr <= s.Pa
            o.sim_end_cond = 'Burn Complete';
            break
        end

    end
    
    o.t                 = o.t(1:sum(o.t>0)+1);
    o.m_o               = o.m_o(1:sum(o.t>0)+1);
    o.P_tnk             = o.P_tnk(1:sum(o.t>0)+1);
    o.P_cmbr            = o.P_cmbr(1:sum(o.t>0)+1);
    o.mdot_o            = o.mdot_o(1:sum(o.t>0)+1);
    o.mdot_f            = o.mdot_f(1:sum(o.t>0)+1);
    o.OF                = o.OF(1:sum(o.t>0)+1);
    o.grn_ID            = o.grn_ID(1:sum(o.t>0)+1);
    o.mdot_n            = o.mdot_n(1:sum(o.t>0)+1);
    o.rdot              = o.rdot(1:sum(o.t>0)+1);
    o.m_f               = o.m_f(1:sum(o.t>0)+1);
    o.F_thr             = o.F_thr(1:sum(o.t>0)+1);
    o.dP                = o.dP(1:sum(o.t>0)+1);

    if s.mp_calc == 1
        o.m_t           = o.m_t(1:sum(o.t>0)+1);
        o.cg            = o.cg(1:sum(o.t>0)+1);
    end

end
