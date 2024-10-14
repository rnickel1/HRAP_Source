def make_part(s, x, req_s, req_x, dx, typename, fderiv, fupdate, **kwargs):
    item = {
        's': { **s },
        'x': { **x },
        'dx': dx,
        'type': typename,
        'fderiv': fderiv,
        'fupdate': fupdate,
    }
    for key, val in kwargs.items():
        item[key] = val

    return item

def store_x(x, xmap, **kwargs):
    for key, val in kwargs.items():
        x = x.at[xmap[key]].set(val)
    
    return x

def step_rk(
    args,
    dt, frhsQ,
    NRK, rka, rkb, rkc,
):
    # Unpack variables for this step
    t, Q, rhsQ, resQ = args
    
    for INTRK in range(NRK):
        RKtime = t + dt*rkc[INTRK]
        
        # Compute right hand side of equation
        rhsQ = frhsQ(Q, rhsQ)
        
        # Initiate and increment Runge-Kutta residuals
        resQ = rka[INTRK]*resQ + dt*rhsQ
        
        # Update fields
        Q += rkb[INTRK]*resQ
    
    t += dt
    
    # Return variables for next state
    return t, Q, rhsQ, resQ

def step_rk4(
    args,
    dt, frhsQ,
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
    rk4c = [
                                    0.0,
        1432997174477.0/9575080441755.0,
        2526269341429.0/6820363962896.0,
        2006345519317.0/3224310063776.0,
        2802321613138.0/2924317926251.0,
    ]
    
    return step_rk(
        args,
        dt, frhsQ,
        5, rk4a, rk4b, rk4c,
    )

def make_integrator(, fderivs, fupdates)
    # TODO: use pytree vmap
