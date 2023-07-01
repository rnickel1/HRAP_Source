%--------------------------------------------------------------------------
% HRAP Simulation Environment
%
% R. Nickel / The University of Tennessee - Knoxville - 2022
%
% Program:  mass
% 
% Purpose:  find center of mass of motor at a given time
%
%--------------------------------------------------------------------------

function [mp] = mass(s,x)

x.m_t       = s.mtr_m + x.m_o + x.m_f;

if x.mLiq_new < 0
    x.mLiq_new = 0;
end

tA          = 0.25*pi*s.tnk_D^2;

if x.mLiq_new > 0

    m_v     = x.m_o - x.mLiq_new;

    vl      = x.mLiq_new/x.ox_props.rho_l;
    vv      = s.tnk_V - vl;

    hl      = vl/tA;
    hv      = vv/tA;

    CoMl    = s.tnk_X - hl./2;
    CoMv    = s.tnk_X - hl - hv./2;
    CoMf    = s.cmbr_X - s.grn_L./2;

    x.cg    = (x.mLiq_new*CoMl + m_v*CoMv + x.m_f*CoMf + s.mtr_m*s.mtr_cg)./(x.m_t);

elseif x.mLiq_new == 0

    m_v     = x.m_o;

    vv      = s.tnk_V;

    hv      = vv./tA;

    CoMv    = s.tnk_X - hv./2;
    CoMf    = s.cmbr_X - s.grn_L./2;

    x.cg    = (m_v*CoMv + x.m_f*CoMf + s.mtr_m*s.mtr_cg)./(x.m_t);

end

mp = [x.m_t, x.cg];