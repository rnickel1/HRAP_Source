import time

import numpy as np

import jax
import jax.numpy as jnp



def make_part(s, x, req_s, req_x, dx, typename=None, fpreprs=None, fderiv=None, fupdate=None, **kwargs):
    part = {
        's': { },
        'x': { },
        # 's': { **s },
        # 'x': { **x },
        'dx': dx,
        'type': typename,
        'fpreprs': fpreprs,
        'fderiv': fderiv,
        'fupdate': fupdate,
    }
    
    # Load defaults
    for key, val in s.items():
        part['s'][typename + '_' + key] = val
    for key, val in x.items():
        part['x'][typename + '_' + key] = val
    
    for key, val in kwargs.items():
        # TODO: error if entry already exists
        if key in s:
            part['s'][typename + '_' + key] = val
        elif key in x:
            part['x'][typename + '_' + key] = val
        else:
            part[typename + '_' + key] = val
    # print('dx before', part['dx'])
    part['dx'] = { typename+'_'+key: typename+'_'+val for key, val in part['dx'].items() }
    
    # Add derivatives to x
    for key, val in part['dx'].items():
        print(val, 'to', 0.0)
        part['x'][val] = 0.0
    # part['dx'] = { 'FUCK_'+key: val for key, val in part['dx'].items() }
    # print('dx after', part['dx'])
    # print(part)
    # print()
    
    return part

def make_engine(tank, grn, cmbr, noz, **kwargs):
    xdict = { }
    s     = { }
    method = { 'fpreprs': [], 'fderivs': [], 'fupdates': [], 'diff_xmap': [], 'diff_dmap': [] }
    
    # Add static variables not related to an individual item
    for key, val in kwargs.items():
        s[key] = val
    
    for part in [tank, grn, cmbr, noz]:
        # print('part x', part['x'])
        # print('part s', part['s'])
        
        if part['fpreprs'] != None:
            method['fpreprs'].append(part['fpreprs'])
        if part['fderiv'] != None:
            method['fderivs'].append(part['fderiv'])
        if part['fupdate'] != None:
            method['fupdates'].append(part['fupdate'])
        
        for key, val in part['x'].items():
            xdict[key] = val
        for key, val in part['s'].items():
            s[key] = val

    Nx = len(xdict.keys())
    xmap = { key: i for key, i in zip(xdict.keys(), range(Nx)) }
    # print(xmap)
    x = np.zeros(Nx)
    for part in [tank, grn, cmbr, noz]:
        # print('PART X', part['x'])
        for key, val in part['x'].items():
            x[xmap[key]] = val
        for key, val in part['dx'].items():
            # print(key, val)
            method['diff_xmap'].append(xmap[key])
            method['diff_dmap'].append(xmap[val])
    
    method['xmap'] = xmap
    method['diff_xmap'] = jnp.array(method['diff_xmap'])
    method['diff_dmap'] = jnp.array(method['diff_dmap'])
    method['comp_names'] = [ 'tnk', 'grn', 'cmbr', 'noz' ]

    return s, jnp.array(x), method

def unpack_engine(s, xstack, method):
    # print('COMPILING unpack_engine')
    xmap = method['xmap']
    comp_names = method['comp_names']
    comps = [ { } for comp_name in comp_names ]
    
    for i in range(len(comp_names)):
        prefix = comp_names[i]
        # Unpack relevant x entries
        for xname in xmap:
            if xname.startswith(prefix):
                comps[i][xname[len(prefix)+1:]] = xstack[:,xmap[xname]]
        # Unpack relevant s entries
        for sname in s:
            if sname.startswith(prefix):
                comps[i][sname[len(prefix)+1:]] = s[sname]  
    
    return comps

def store_x(x, xmap, **kwargs):
    for key, val in kwargs.items():
        x = x.at[xmap[key]].set(val)
    
    return x

def step_fe(
    s, x, xmap, diff_xmap, diff_dmap,
    dt, fderiv,
):
    x = fderiv(s, x, xmap)
    
    return x.at[diff_xmap].add(dt*x[diff_dmap])

def step_rk(
    s, x, xmap, diff_xmap, diff_dmap,
    dt, fderiv,
    NRK, rka, rkb, rkc,
):
    resx = jnp.zeros_like(x[diff_xmap])
    for INTRK in range(NRK):        
        x = fderiv(s, x, xmap)
        
        resx = rka[INTRK]*resx + dt*x[diff_dmap]
        
        x = x.at[diff_xmap].add(rkb[INTRK]*resx)
    
    return x

# "Low memory" RK4, using for loop simplicity and generalizability
def step_rk4(
    s, x, xmap, diff_xmap, diff_dmap,
    dt, fderiv,
):
    rk4a = [
                                     0.0,
        -567301805773.0 /1357537059087.0,
        -2404267990393.0/2016746695238.0,
        -3550918686646.0/2091501179385.0,
        -1275806237668.0/ 842570457699.0,
    ]
    rk4b = [
        1432997174477.0/ 9575080441755.0,
        5161836677717.0/13612068292357.0,
        1720146321549.0/ 2090206949498.0,
        3134564353537.0/ 4481467310338.0,
        2277821191437.0/14882151754819.0,
    ]
    rk4c = [ # Unused, just for value of subtime if needed
                                    0.0,
        1432997174477.0/9575080441755.0,
        2526269341429.0/6820363962896.0,
        2006345519317.0/3224310063776.0,
        2802321613138.0/2924317926251.0,
    ]
    
    return step_rk(
        s, x, xmap, diff_xmap, diff_dmap,
        dt, fderiv,
        5, rk4a, rk4b, rk4c,
    )

# Output must be called before getting a new integrator or the behavior of the old one will be undefined
def make_integrator(fstep, method):
    def fderiv(s, x, xmap):
        for fderiv in method['fderivs']:
            x = fderiv(s, x, xmap)
        
        return x

    def step_t(i, args):
        t, dt, s, x, xmap, xstack = args

        x = fstep(s, x, xmap, method['diff_xmap'], method['diff_dmap'], dt, fderiv)

        for fupdate in method['fupdates']:
            x = fupdate(s, x, xmap)

        xstack = xstack.at[i, :].set(x)

        return t, dt, s, x, xmap, xstack

    # Note that under compilation, xmap etc. become fixed
    def run_solver(s, x, dt=1E-2, T=10.0, method=method):
        # Initialize solution field
        # x = e
        
        # Initialize integration variables
        t = 0.0
        # x = 
        # dx, resQ = jnp.zeros_like(Q), jnp.zeros_like(Q)

        # for i_dx in method.diff_dmap:
        #     method

        Nt = int(np.ceil(T / dt))
        xstack = jnp.zeros((Nt, x.size))
        # print('AGFFSAGAGS', x[method['xmap']['tnk_m_ox']])

        for fpreprs in method['fpreprs']:
            x = fpreprs(s, x, method['xmap'])
        
        xstack = xstack.at[0, :].set(x)

        # Run solver while loop and record elapsed wall time
        # wall_t1 = time.time()
        t, _, _, x, _, xstack = jax.lax.fori_loop(1, Nt+1, step_t, (t, dt, s, x, method['xmap'], xstack)) # TODO: jit outside!
        # for i in range(1, Nt+1):
        #     t, _, _, x, _, xstack = step_t(i, (t, dt, s, x, method['xmap'], xstack))
        # wall_t2 = time.time()

        # print('Solved in', wall_t2 - wall_t1, 's')

        return t, x, xstack
    
    return run_solver
    # TODO: use pytree vmap

def get_impulse_class(value_Ns: float) -> str:
    index = int(np.floor(np.log(value_Ns / 2.5)/np.log(2))) + 1
    Iclass = ''
    if index > 25: Iclass += chr(ord('A') + (index // 25 - 1))
    Iclass += chr(ord('A') + (index % 25))
    
    return Iclass

def export_rse(out_file, series, OD, L, D_throat, D_exit, motor_type='liquid', mfg='SSTA'):
    t, F, prop_mdot, m, Cg = [series[k] for k in ['t', 'F', 'prop_mdot', 'm', 'Cg']]
    
    # TODO: downsample
    # TODO: autocut?
    
    Itot = np.trapezoid(F, t)
    prop_burnt = np.trapezoid(prop_mdot, t) # not m[0] - m[-1] due to potential venting etc.
    F_max = np.max(F)
    T_burn = t[-1] - t[0]
    
    Isp = Itot / (prop_burnt*9.81)
    F_avg = Itot / T_burn
    
    with open(out_file, 'w') as f:
        f.write('<engine-database>\n    <engine-list>\n')
        f.write(\
'    <engine FDiv=\"10\" FFix=\"1\" FStep=\"-1.\" Isp=\"{Isp}\" Itot=\"{Itot}\"\
 Type=\"Hybrid\" auto-calc-cg=\"0\" auto-calc-mass=\"0\" avgThrust=\"{F_avg}\"\
 burn-time=\"{T_burn}\" cgDiv=\"10\" cgFix=\"1\" cgStep=\"-1.\" code=\"{code}{F_avg_i}\" delays=\"0\"\
 dia=\"{OD}\" D_exit=\"{D_exit}\" initWt=\"{m0}\" len=\"{L}\"\
 mDiv=\"10\" mFix=\"1\" mStep=\"-1.\" massFrac=\"{mfrac}\" mfg=\"{mfg}\" peakThrust=\"{F_max}\"\
 propWt=\"{propWt}\" tDiv=\"10\" tFix=\"1\" tStep=\"-1.\" throatDia=\"{D_throat}\">\n    <data>\n'.format(
            Isp=Isp, Itot=Itot,
            F_avg=F_avg,
            T_burn=T_burn, code=get_impulse_class(Itot), F_avg_i=int(np.round(F_avg)),
            OD=OD*1000.0, D_exit=D_exit*1000.0, m0=m[0]*1000.0, L=L*1000.0,
            mfrac=(m[0] - m[-1])/m[0], mfg=mfg, F_max=F_max,
            propWt=(m[0]-m[-1])*1000.0, D_throat=D_throat*1000.0))
        for i in range(t.size):
            f.write('        <eng-data cg=\"{Cg}\" f=\"{F}\" m=\"{m}\" t=\"{t}\"/>\n'.format(
                        Cg=Cg[i]*1000.0, F=np.max([0.0, F[i]]), m=m[i]*1000.0, t=t[i]))
        f.write('    </data>\n    </engine>\n    </engine-list>\n</engine-database>')

def export_eng(out_file, series, OD, L, mfg='SSTA'):
    t, F, m = [series[k] for k in ['t', 'F', 'm']]
    # TODO: properly downsample!
    
    Itot = np.trapezoid(F, t)
    T_burn = t[-1] - t[0]
    
    F_avg = Itot / T_burn
    
    with open(out_file, 'w') as f:
        f.write('{mfg} {OD} {L} P {propWt} {m0} {code}{F_avg_i}\n'.format(mfg=mfg, OD=1000.0 * OD, L=1000.0 * L, propWt=m[0] - m[-1], m0=m[0], code=get_impulse_class(Itot), F_avg_i=int(np.round(F_avg))))
    for i in range(31):
        f.write(' {t} {F} \n'.format(t=t[i * t.size//31], F=F[i * t.size//31]))
    f.write(' {t} 0.0 \n'.format(t=t[-1]))
