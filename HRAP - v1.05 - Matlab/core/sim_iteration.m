%-----------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  sim_iteration
% 
% Purpose:  one iteration of HRAP simulation
%
%-----------------------------------------------------------------------------

function [s,x,o,t] = sim_iteration(s,x,o,t,i)

dt = s.dt;

t = t + dt;

    [x] = tank(s,o,x,t);
    
    [x] = s.regression_model(s,x);
    
	[x] = comb(s,x,t);
    
	[x] = chamber(s,x);              
    
    [x] = nozzle(s,x);
    
    if s.mp_calc == 1
        mp              = mass(s,x);
        o.m_t(i)        = mp(1);
        o.cg(i)         = mp(2);
    end

    o.t(i)              = t;
    o.m_o(i)            = x.m_o;
    o.P_tnk(i)          = x.P_tnk;
    o.P_cmbr(i)         = x.P_cmbr;
    o.mdot_o(i)         = x.mdot_o;
    o.mdot_f(i)         = x.mdot_f;
    o.OF(i)             = x.OF;
    o.grn_ID(i)         = x.grn_ID;
    o.mdot_n(i)         = x.mdot_n;
    o.rdot(i)           = x.rdot;
    o.m_f(i)            = x.m_f;
    o.dP(i)             = x.dP;
    o.F_thr(i)          = x.F_thr;

end