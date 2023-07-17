%--------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  tank
% 
% Purpose:  model oxidizer tank emptying in equilibrium
%
%--------------------------------------------------------------------------

function [x] = tank(s,o,x,t)

dt = s.dt;

%Find oxidizer thermophysical properties
    [x.ox_props] = NOX(x.T_tnk);

    x.P_tnk = x.ox_props.Pv;

%Find oxidizer mass flow rate

    dP = x.P_tnk - x.P_cmbr;

    if dP < 0
        dP = 0;
    end

    if s.tburn == 0 || t <= s.tburn
        if s.vnt_S == 0
            x.mdot_v = 0;
            if x.mLiq_new == 0
                x.mdot_o = s.inj_CdA*s.inj_N*sqrt(2*x.ox_props.rho_v*dP);
            else
                x.mdot_o = s.inj_CdA*s.inj_N*sqrt(2*x.ox_props.rho_l*dP);
            end
            mD = (x.mdot_o+x.mdot_v)*dt;
        elseif s.vnt_S == 1
            x.mdot_v     = s.vnt_CdA*sqrt(2*x.ox_props.rho_v*dP);
            if x.mLiq_new == 0
                x.mdot_o = s.inj_CdA*s.inj_N*sqrt(2*x.ox_props.rho_v*dP);
            else
                x.mdot_o = s.inj_CdA*s.inj_N*sqrt(2*x.ox_props.rho_l*dP);
            end
            mD = (x.mdot_o+x.mdot_v)*dt;
        elseif s.vnt_S == 2
            x.mdot_v     = s.vnt_CdA*sqrt(2*x.ox_props.rho_v*dP);
            if x.mLiq_new == 0
                x.mdot_o = s.inj_CdA*s.inj_N*sqrt(2*x.ox_props.rho_v*dP) + x.mdot_v;
            else
                x.mdot_o = s.inj_CdA*s.inj_N*sqrt(2*x.ox_props.rho_l*dP) + x.mdot_v;
            end
            mD = x.mdot_o*dt;
        else
            error('Error: Vent State Undefined');
        end
    elseif s.tburn > 0 && t > s.tburn
        x.mdot_o        = 0;
        mD              = 0;
    end

%Find mass discharged during time step
    m_o_old              = x.m_o;
    x.m_o                = x.m_o - x.mdot_o*dt;

if x.mLiq_new < x.mLiq_old && x.mLiq_new > 0 && x.mdot_o > 0

    %Find mass of liquid nitrous evaporated during time step
        x.mLiq_old = x.mLiq_new - mD;
        [x.ox_props] = NOX(x.T_tnk);
        x.mLiq_new = (s.tnk_V - (x.m_o/x.ox_props.rho_v))/ ...
                    ((1/x.ox_props.rho_l)-(1/x.ox_props.rho_v));
        mv = x.mLiq_old - x.mLiq_new;

    %Find heat removed from liquid
        dT = -mv*x.ox_props.Hv/(x.mLiq_new*x.ox_props.Cp);
        x.T_tnk = x.T_tnk + dT;
        [op] = NOX(x.T_tnk);
        x.dP = op.Pv - x.P_tnk;

elseif x.mLiq_new >= x.mLiq_old && x.mLiq_new > 0 && x.mdot_o > 0
    
    dP_avg = mean(o.dP(1:sum(o.dP<0)));

    P_new = x.P_tnk + dP_avg;

    vp = @(T) 7251000*exp((1/(T/309.57))*...
        (-6.71893*(1-T/309.57) + 1.35966*(1-(T/309.57))^(3/2) + -1.3779*...
        (1-(T/309.57))^(5/2) + -4.051*(1-(T/309.57))^5)) - P_new;

    x.T_tnk = fzero(vp,x.T_tnk);

    x.dP = x.ox_props.Pv - x.P_tnk;

    [x.ox_props] = NOX(x.T_tnk);

    x.mLiq_new = (s.tnk_V - (x.m_o/x.ox_props.rho_v))/ ...
                    ((1/x.ox_props.rho_l)-(1/x.ox_props.rho_v));
    x.mLiq_old = 0;

elseif x.mLiq_new <= 0 && x.mdot_o > 0
    
    if x.mLiq_new ~= 0
        x.mLiq_new = 0;
    end

    %Find Z factor

    Z_old = x.ox_props.Z;

    Zguess = Z_old;
    epsilon = 1;
    
    Ti = x.T_tnk;
    Pi = x.P_tnk;

    while epsilon >= 0.000001

        T_ratio = ((Zguess*x.m_o)/(Z_old*m_o_old))^(0.3);
        x.T_tnk = T_ratio*Ti;
        P_ratio = T_ratio^(1.3/0.3);
        x.P_tnk = P_ratio*Pi;

        [x.ox_props] = NOX(x.T_tnk);

        Z = x.ox_props.Z;
        
        epsilon = abs(Zguess - Z);

        Zguess = (Zguess + Z)/2;

    end
    
end
