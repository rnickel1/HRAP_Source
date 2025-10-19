import os
import sys
import time
from pathlib import Path
import pickle as pkl
from importlib.resources import files as imp_files
from importlib.metadata import version

import scipy
import numpy as np

import dearpygui.dearpygui as dpg

from jax.scipy.interpolate import RegularGridInterpolator

import hrap.core as core
import hrap.chem as chem
import hrap.fluid as fluid
import hrap.units as units
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.units   import _in, _ft

from hrap.gui.themes import create_babber_theme
from dearpygui_ext.themes import create_theme_imgui_light, create_theme_imgui_dark

hrap_version = version('hrap')

# Virtualized Python environments may redirect these to other locations
# For example, Windows Store Python redirects %APPDATA%\Roaming to %APPDATA%\Local\Packages\PythonSoftwareFoundation.Python.[some garbage]\LocalCache\Roaming
def get_datadir() -> Path:
    home = Path.home()
    if sys.platform == 'win32':
        return home / 'AppData/Roaming'
    elif sys.platform == 'linux':
        return home / '.local/share'
    elif sys.platform == 'darwin':
        return home / 'Library/Application Support'

# Global vars, issues unless declared outside of main
hrap_root    = None
active_file  = None
config       = { }
upd_due      = True
s, x, method = [None]*3
t, xstack = [None]*2 # TODO: Make m, Cg part of x?
fire_engine  = None
comb, oxidizers, fuels = [None]*3
get_sat_props, get_Pv_loss = None, None

N_fuel = 3

def clamped_param(val, props):
    if 'min' in props and val < props['min']:
        return [props['min']]*2
    elif 'max' in props and val > props['max']:
        return [props['max']]*2
    return val, None

def get_param(tag):
    props = config[tag]
    v = dpg.get_value(tag)
    if 'units' in props: v = props['gui2sim_units'](v)
    return v

def upd_direct_param(k): # TODO: no clam version
    global upd_due, x

    v = get_param(k)
    if k in s:
        if s[k] != v:
            s[k] = v
            upd_due = True
            print('update due to s', k)
    elif k in method['xmap']:
        # print(k, x[method['xmap'][k]], v)
        if x[method['xmap'][k]] != v:
            print('update due to x', k)
            x = x.at[method['xmap'][k]].set(v)
            upd_due = True
    else:
        print('ERROR:', k, 'is nowhere!')

def upd_param(tag):
    props = config[tag]
    v, clam = clamped_param(get_param(tag), props)
    if clam != None:
        if 'units' in props: clam = props['sim2gui_units'](clam)
        dpg.set_value(tag, clam)
    if props['direct']: upd_direct_param(tag)
    return clam

def set_param(tag, val):
    props = config[tag]
    if props['type'] == float: val = float(val) # Cast to ensure correct type for reals i.e. to convert JAX containers
    if 'units' in props: val = props['sim2gui_units'](val)
    dpg.set_value(tag, val)
    return upd_param(tag)

# Callbacks for manual adjustments (gets sets the UI components, not the motor)
def man_call_tnk_D():
    D, L, T, fill = [get_param(tag) for tag in ['tnk_D', 'tnk_L', 'tnk_T', 'tnk_fill']]
    props = get_sat_props(T)
    V = np.pi/4 * D**2 * L
    # print('TANK', V, D, L)
    set_param('tnk_V', V)
    set_param('tnk_m_ox', fill/100.0 * props['rho_l'] * V)

man_call_tnk_L = man_call_tnk_D

def man_call_tnk_V():
    V, D, T, fill = [get_param(tag) for tag in ['tnk_V', 'tnk_D', 'tnk_T', 'tnk_fill']]
    props = get_sat_props(T)
    set_param('tnk_L', V / (np.pi/4 * D**2))
    set_param('tnk_m_ox', fill/100.0 * props['rho_l'] * V)

def man_call_tnk_inj_D():
    inj_D, inj_Cd = [get_param(tag) for tag in ['tnk_inj_D', 'tnk_inj_Cd']]
    set_param('tnk_inj_CdA', inj_Cd*(np.pi/4 * inj_D**2))

man_call_tnk_inj_Cd = man_call_tnk_inj_D

def man_call_tnk_inj_CdA():
    inj_CdA, inj_Cd = [get_param(tag) for tag in ['tnk_inj_CdA', 'tnk_inj_Cd']]
    inj_D_clam = set_param('tnk_inj_D', np.sqrt(4/np.pi*inj_CdA/inj_Cd))
    if inj_D_clam != None: set_param('tnk_inj_CdA', inj_Cd*(np.pi/4 * inj_D_clam**2))

def man_call_tnk_T():
    T, V, fill = [get_param(tag) for tag in ['tnk_T', 'tnk_V', 'tnk_fill']]
    props = get_sat_props(T)
    set_param('tnk_P', props['Pv'])
    set_param('tnk_m_ox', fill/100.0 * props['rho_l'] * V)

def man_call_tnk_P():
    T_props = config['tnk_T']
    T_min, T_max = T_props['min'], T_props['max']
    P_min, P_max = [get_sat_props(T)['Pv'] for T in [T_min, T_max]]
    T, P = [get_param(tag) for tag in ['tnk_T', 'tnk_P']]
    # Manually deal with min,max since opt will fail otherwise
    if P <= P_min:
        set_param('tnk_T', T_min)
        set_param('tnk_P', P_min)
        return
    if P >= P_max:
        set_param('tnk_T', T_max)
        set_param('tnk_P', P_max)
        return
    T = scipy.optimize.brentq(get_Pv_loss, T_min, T_max, args=(P,))
    set_param('tnk_T', T)

def man_call_tnk_m_ox():
    T, V, m = [get_param(tag) for tag in ['tnk_T', 'tnk_V', 'tnk_m_ox']]
    props = get_sat_props(T)
    fill_clam = set_param('tnk_fill', 100.0 * m / (props['rho_l'] * V))
    if fill_clam != None: set_param('tnk_m_ox', fill_clam/100.0 * props['rho_l'] * V)

def man_call_tnk_fill():
    T, V, fill = [get_param(tag) for tag in ['tnk_T', 'tnk_V', 'tnk_fill']]
    props = get_sat_props(T)
    set_param('tnk_m_ox', fill/100.0 * props['rho_l'] * V)

def man_call_noz_thrt():
    D_throat, ER = [get_param(tag) for tag in ['noz_thrt', 'noz_ER']]
    set_param('noz_exit', np.sqrt(ER * D_throat**2))

def man_call_noz_D_exit():
    D_throat, D_exit = [get_param(tag) for tag in ['noz_thrt', 'noz_exit']]
    ER_clam = set_param('noz_ER', D_exit**2/D_throat**2)
    if ER_clam != None: set_param('noz_exit', np.sqrt(ER_clam * D_throat**2))

def man_call_noz_ER():
    ER, D_throat = [get_param(tag) for tag in ['noz_ER', 'noz_thrt']]
    set_param('noz_exit', np.sqrt(ER * D_throat**2))

def init_deps(): # Called after init/load to verify consistency
    man_call_tnk_D()
    man_call_tnk_inj_D()
    man_call_tnk_T()
    # man_call_ox_m()
    man_call_noz_ER()

# def load_preset_chem(name):
    # chem = scipy.io.loadmat(hrap_root/'resources'/'propellant_configs'/name)
    
    # chem = chem['s'][0][0]
    # chem_OF = chem['prop_OF'].ravel()
    # chem_Pc = chem['prop_Pc'].ravel()
    # chem_k, chem_M, chem_T = chem['prop_k'], chem['prop_M'], chem['prop_T']
    # if chem_k.size == 1: chem_k = np.full_like(chem_T, chem_k.item())

    # chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k, fill_value=1.4)
    # chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M, fill_value=29.0)
    # chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T, fill_value=293.0)

    # return chem_interp_k, chem_interp_M, chem_interp_T

def prep_chem(allow_rho_est=True):
    chem_ox = { oxidizers['Nitrous Oxide']['chem']: 1.0 }
    chem_fu = {  }
    sum_fu_mf, can_rho_est = 0.0, True
    for i in range(N_fuel):
        component_i, mfrac_i = [dpg.get_value(k.format(i)) for k in ['grn_component_{}', 'grn_mfrac_{}']]
        if component_i != 'None':
            chem_fu[fuels[component_i]['chem']] = mfrac_i
            sum_fu_mf += mfrac_i # For normalization
            if 'rho' not in fuels[component_i]:
                can_rho_est = False
    
    if allow_rho_est: # During first init call this isn't allowed as set_param isn't ready
        if can_rho_est:
            # dpg.configure_item('grn_est_rho', readonly=~can_rho_est) # Isn't supported for some reason...
            grn_rho_est = 0.0
            for i in range(N_fuel):
                component_i, mfrac_i = [dpg.get_value(k.format(i)) for k in ['grn_component_{}', 'grn_mfrac_{}']]
                if component_i != 'None':
                    # rho = (sum m_i)/(sum V_i) = (sum m mf_i)/(sum m mf_i 1/rho_i) = 1 / (sum mf_i/rho_i)
                    grn_rho_est += mfrac_i / fuels[component_i]['rho']
            grn_rho_est = 1 / grn_rho_est * sum_fu_mf
            if dpg.get_value('grn_est_rho'):
                set_param('grn_rho', grn_rho_est)
        else:
            dpg.set_value('grn_est_rho', False)
            dpg.configure_item('grn_rho', readonly=False)
    
    return chem_ox, chem_fu, sum_fu_mf

def update_chem(chem_ox, chem_fu, sum_fu_mf=1.0):
    chem_Pc, chem_OF = np.linspace(10*units._atm, 50*units._atm, 10), np.linspace(1.0, 10.0, 20)
    chem_k, chem_M, chem_T = [np.zeros((chem_Pc.size, chem_OF.size)) for i in range(3)]
    internal_state = None
    for j, OF in enumerate(chem_OF):
        for i, Pc in enumerate(chem_Pc):
            o = OF / (1 + OF) # o/f = OF, o+f=1 => o=OF/(1 + OF)
            flame, internal_state = comb.solve(Pc, {**{c: mf*o for c,mf in chem_ox.items()}, **{c: mf/sum_fu_mf*(1-o) for c,mf in chem_fu.items()}}, max_iters=150, internal_state=internal_state)
            chem_k[i,j], chem_M[i,j], chem_T[i,j] = flame.gamma, flame.M, flame.T

    chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k, fill_value=1.4)
    chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M, fill_value=29.0)
    chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T, fill_value=293.0)

    return chem_interp_k, chem_interp_M, chem_interp_T

def setup_motor(tnk_inj_vap_model, tnk_inj_liq_model, chem_interp_k, chem_interp_M, chem_interp_T):
    # Initialization
    tnk = make_sat_tank(
        get_sat_props,
        V = (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
        inj_CdA= 0.5 * (np.pi/4 * 0.5**2 * _in**2),
        m_ox=1,#14.0, # TODO: init limit
        # m_ox = 3.0,
        inj_vap_model = core.StaticVar(tnk_inj_vap_model),
        inj_liq_model = core.StaticVar(tnk_inj_liq_model),
    )
    # print('INJ TEST', 0.5 * (np.pi/4 * 0.5**2 * _in**2))

    shape = make_circle_shape(
        ID = 2.5 * _in,
    )
    grn = make_constOF_grain(
        shape,
        OF = 3.0,
        OD = 5.0 * _in,
        L = 4.0 * _ft,
    )

    cmbr = make_chamber(
    )

    noz = make_cd_nozzle(
        thrt = 1.75 * _in, # Throat diameter
        ER = 5.0,         # Exit/throat area ratio
    )

    s, x, method = core.make_engine(
        tnk, grn, cmbr, noz,
        chem_interp_k=chem_interp_k, chem_interp_M=chem_interp_M, chem_interp_T=chem_interp_T,
        Pa=101e3,
    )
    # direct_s_tags = [tag in direct_tags if tag in s]
    # direct_x_tags = [tag in direct_tags if tag in method['xmap']]

    fire_engine = core.make_integrator(
        # core.step_rk4,
        core.step_fe,
        method,
    )
    
    return s, x, method, fire_engine

def recompile_motor():
    global config, upd_due, s, x, method, fire_engine, get_sat_props, get_Pv_loss
    
    # Handle any changes in the oxidizer
    ox = oxidizers[dpg.get_value('ox_component_0')]
    # Bake saturation curves
    get_sat_props = fluid.bake_sat_coolprop(ox['coolprop'], np.linspace(ox['T_min'], ox['T_max'], 20))
    # Make loss function for Pv editor in GUI
    get_Pv_loss = jax.jit(lambda T, Pv_targ, get_sat_props=get_sat_props: get_sat_props(T)['Pv'] - Pv_targ)

    # TOOD: skip recompile if combo already in some dict?
    # gcm = dpg.get_value('select_grain_chem_mode')
    # if gcm == 'HRAP Presets':
        # print(dpg.get_value('select_grain_chem_hrap_presets')+'.mat')
        # chem_info = load_preset_chem(dpg.get_value('select_grain_chem_hrap_presets')+'.mat')
    s, x, method, fire_engine = setup_motor(dpg.get_value('tnk_inj_vap_model'), dpg.get_value('tnk_inj_liq_model'), *update_chem(*prep_chem(allow_rho_est=False)))
    upd_due = True
    # Need to respecify all internal variables based on config
    for tag, props in config.items():
        if props['direct']: upd_direct_param(tag)
    init_deps()
    prep_chem() # Call again to estimate density (needed motor to init first)
    
    # Finish oxidizer changes
    # Update GUI limits
    config['tnk_T']['min'], config['tnk_T']['max'] = ox['T_min'], ox['T_max']
    # Re-apply current T value to apply limits
    print(get_param('tnk_T'))
    set_param('tnk_T', get_param('tnk_T'))
    man_call_tnk_T()
    print('tnk T lims', config['tnk_T']['min'], config['tnk_T']['max'])

def calculate_m_Cg():
    dry_m, dry_cg, tnk_ID, ox_pos, grn_pos = [get_param(k) for k in ['dry_m', 'dry_cg', 'tnk_D', 'ox_pos', 'grn_pos']]
    T_ox, m_ox_vap, m_ox_liq, rho_ox_vap, rho_ox_liq, cmbr_m_g, grn_A = [xstack[:,method['xmap'][k]] for k in ['tnk_T', 'tnk_m_ox_vap', 'tnk_m_ox_liq', 'tnk_rho_ox_vap', 'tnk_rho_ox_liq', 'cmbr_m_g', 'grn_A']]
    grn_L, grn_rho = [s[k] for k in ['grn_L', 'grn_rho']]
    grn_m = grn_A*grn_L*grn_rho
    
    m = dry_m + m_ox_vap + m_ox_liq + grn_m + cmbr_m_g
    
    cg = jnp.full_like(m, dry_cg*dry_m)
    A_tnk = np.pi/4*tnk_ID**2
    L_ox_vap = (m_ox_vap/rho_ox_vap) / A_tnk
    L_ox_liq = (m_ox_liq/rho_ox_liq) / A_tnk
    cg += m_ox_vap * (ox_pos + L_ox_vap/2)
    cg += m_ox_liq * (ox_pos + L_ox_vap + L_ox_liq/2)
    # Approximate chamber mass Cg as center of grain
    cg += (grn_m + cmbr_m_g) * (grn_pos + grn_L/2)
    cg /= m
    
    return m, cg

def main():
    global hrap_root, config, upd_due, t, xstack, comb, oxidizers, fuels #, s, x, method

    print('beginning w/ hrap version', hrap_version)
    jax.config.update('jax_enable_x64', True)
    
    # Ensure the app data directory for hrap exists
    data_root = get_datadir()/'hrap'
    Path(data_root).mkdir(parents=True, exist_ok=True)
    print('app data will go in', os.path.realpath(data_root))

    # Ensure the autosave directory exists
    auto_root = data_root/'autosaves'
    Path(auto_root).mkdir(parents=True, exist_ok=True)
    
    # Get the HRAP install root
    hrap_root = Path(imp_files('hrap'))
    
    def apply_theme(theme, save):
        settings['theme'] = theme
        if save: save_settings(settings)
        dpg.bind_theme(themes[theme])
        dpg.bind_font(0) # dpg won't apply a new font unless go back to default first
        dpg.bind_font(babber_font if theme == 'Yellow Babber' else primary_font)
    
    def save_settings(settings):
        pkl.dump(settings, open(data_root/'settings.pkl', 'wb'))
    
    def save_config(file):
        save, save_config = { }, { }
        # TODO: save options like units etc.? should save injector model etc...
        for tag in config:
            save_config[tag] = get_param(tag)
        save['hrap_version'] = hrap_version
        save['config'] = save_config
        print(save)
        pkl.dump(save, open(file, 'wb'))

    def load_config(file):
        save = pkl.load(open(file, 'rb'))
        save_config = save['config']
        for tag, val in save['config'].items():
            set_param(tag, val)
        init_deps()
    
    default_settings = {
        'view_w': 1000, 'view_h': 1000*6//8,
        'view_x': 0, 'view_y': 0,
        'theme': 'Dark',
    }
    try:
        settings = pkl.load(open(data_root/'settings.pkl', 'rb'))
        # Add in any missing settings (such as added from version changes)
        for k, v in default_settings.items():
            if not k in settings:
                print('Settings was missing', k, 'defaulting to', v)
                settings[k] = v
    except FileNotFoundError:
        print('Settings not found, initializing...')
        settings = default_settings
    save_settings(settings)
    
    # See https://github.com/hoffstadt/DearPyGui

    dpg.create_context()
    dpg.create_viewport(title='HRAP', width=settings['view_w'], height=settings['view_h'], x_pos=settings['view_x'], y_pos=settings['view_y'], small_icon=str(hrap_root/'resources'/'icon.ico'))#, large_icon='\a.ico'
    dpg.setup_dearpygui()
    dpg.set_viewport_vsync(False)

    dark_theme   = create_theme_imgui_dark()
    light_theme  = create_theme_imgui_light()
    babber_theme = create_babber_theme()
    themes = { 'Dark': 0, 'Light': light_theme, 'Extra Dark': dark_theme, 'Yellow Babber': babber_theme }
    
    with dpg.font_registry():
        primary_font = dpg.add_font(hrap_root/'resources'/'fonts'/'Roboto-Regular.ttf', 14)
        babber_font  = dpg.add_font(hrap_root/'resources'/'fonts'/'BubblegumSans-Regular.ttf', 14)

    apply_theme(settings['theme'], False)
    
    # Create thermo database, including a few common ones not found in propep database
    plastisol = chem.make_basic_reactant(
        formula = 'Plastisol-362', composition = { 'C': 7.200, 'H': 10.82, 'O': 1.14, 'Cl': 0.669 },
        M = 140.86, T0 = 298.15, h0 = -2.6535755e7, # kg/kmol, K, J/kmol
    )
    ABS = chem.make_basic_reactant(
        formula = 'ABS', composition = { 'C': 3.85, 'H': 4.85, 'N': 0.43 },
        M = 57.15, T0=298.15, h0=-6.263e7 # kg/kmol, K, J/kmol
    )
    comb = chem.ChemSolver([hrap_root/'thermo.dat', plastisol, ABS])
    oxidizers = {
        'Nitrous Oxide': {
            'chem': 'N2O(L),298.15K',
            'coolprop': 'NitrousOxide',
            'T_min': 183.0, 'T_max': 309.0, # Low is generously high, yet leaves room for applicabiltiy and 309 is max applicability of sat nos
        },
        'Oxygen': {
            'chem': 'O2(L)',
            'coolprop': 'Oxygen',
            'T_min': 75.0, 'T_max': 150.0,
        },
    }
    # 1171 = 1 / (0.2/2700 + 0.8/x)
    # x = 0.8/(1 / 1171 - 0.2/2700) = 1026
    # 1117 is 95.4% of 20% al, 80% 362 used on redshift
    # 'ABS', 'Asphalt', 'HDPE', 'HTPB_Paraffin', 'HTPB', 'Metalized_Plastisol', 'Paraffin', 'Sorbitol'
    fuels = {
        'ABS': {
            'chem': 'ABS',
            'rho': 1070.0, # kg/m3, varies on blend so not a precise value
        },
        'Plastisol-362': {
            'chem': 'Plastisol-362',
            'rho': 1026.0, # kg/m3
        },
        'Aluminum': {
            'chem': 'AL(cr)',
            'rho': 2712.0, # kg/m3
        },
        'Paraffin': {
            'chem': 'C32H66(a)',
        }
    }
    for f in ['HTPB', 'Polyethylene']:
        fuels[f] = { 'chem': f }


    # dpg.set_viewport_vsync(True)

    # TODO: use that this also gets called on move to set intial pos
    def resize_callback():
        # Get the size of the main window
        vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        vx, vy = dpg.get_viewport_pos()
        mh = 20 # Menu height
        
        settings['view_w'] = vw; settings['view_h'] = vh
        settings['view_x'] = vx; settings['view_y'] = vy
        save_settings(settings)

        # Update the size and position of each window based on the main window's size
        dpg.set_item_width ('menu', vw)

        def set_wh(tag, w, h):
            dpg.set_item_width (tag, w)
            dpg.set_item_height(tag, h)
        def set_whxy(tag, w, h, x, y):
            set_wh(tag, w, h)
            dpg.set_item_pos(tag, [x, y])
        
        set_whxy('tank',    vw // 2, vh // 3 - mh, 0,       mh         )
        set_whxy('grain',   vw // 2, vh // 3 - mh, vw // 2, mh         )
        set_whxy('chamber', vw // 2, vh // 6,      0,       vh // 3    )
        set_whxy('misc',    vw // 2, vh // 6,      0,       vh // 2)
        set_whxy('nozzle',  vw // 2, vh // 3,      vw // 2, vh // 3    )
        set_whxy('previewL', vw // 2, vh // 3,     0,       2 * vh // 3)
        set_whxy('previewR', vw // 2, vh // 3,     vw // 2, 2 * vh // 3)

        for i in range(2): set_wh('preview_{i}'.format(i=i), vw // 2 - 18, vh // 3 - 36)

    # First row
    settings = { 'no_move': True, 'no_collapse': True, 'no_resize': True, 'no_close': True }

    def make_param(title, props):
        config[props['tag']] = props
        if not 'direct' in props: props['direct'] = False
        if props['type'] == float:
            decimal = props['decimal'] if 'decimal' in props else 3
            callbacks = [lambda *_, key=props['tag']: upd_param(key)] # All callbacks, beginning with update (clamp etc.)
            if 'man_call' in props: callbacks.append(props['man_call'])
            # callback = (None if len(callbacks) == 0 else (callbacks[0] if len(callbacks) == 1 else lambda *_, arr=callbacks: [f() for f in arr]))
            callback = lambda *_, farr=callbacks: [f() for f in farr]
            # with dpg.group(horizontal=True):
            with dpg.table_row():
                dpg.add_text(title)
                dpg.add_input_float(format=f'%.{decimal}f', tag=props['tag'], callback=callback, width=-1)
                # dpg.add_input_float(label=title, format=f'%.{decimal}f', tag=props['tag'], callback=callback)
                # dpg.add_input_float(label=title, step=props['step'], format=f'%.{decimal}f', tag=props['tag'], callback=callback)
                
                # Add unit selector, units are applied before running the motor
                if 'units' in props:
                    unit_type = units.get_unit_type(props['units'])
                    props['unit_type'] = unit_type
                    if unit_type != None:
                        props['gui2sim_units'] = units.unit_conversions[unit_type][props['units']]
                        props['sim2gui_units'] = units.inv_unit_conversions[unit_type][props['units']]
                        def _units_callback(sender, app_data, *_, tag=props['tag']):
                            # print(app_data, tag)
                            props = config[tag]
                            _v = get_param(tag)
                            props['units'] = app_data
                            props['gui2sim_units'] = units.unit_conversions[props['unit_type']][props['units']]
                            props['sim2gui_units'] = units.inv_unit_conversions[props['unit_type']][props['units']]
                            set_param(tag, _v)
                        dpg.add_combo(items=[k for k in units.unit_conversions[unit_type].keys()], default_value=props['units'], callback=_units_callback, width=48)
                    else:
                        print('Error: type of unit "{unit}" could not be found!'.format(unit=props['units']))
                        del props['units']
            
            if 'default' in props:
                _v = props['default']
                if 'units' in props: _v = props['sim2gui_units'](_v)
                dpg.set_value(props['tag'], float(_v)) # Can't use set_param as s isn't ready yet
        if props['type'] == int:
            callbacks = [lambda *_, key=props['tag']: upd_param(key)] # Same callback setup as above
            if 'man_call' in props: callbacks.append(props['man_call'])
            callback = lambda *_, farr=callbacks: [f() for f in farr]
            with dpg.table_row():
                dpg.add_text(title)
                dpg.add_input_int(tag=props['tag'], callback=callback, width=-1)
            if 'default' in props: dpg.set_value(props['tag'], int(props['default']))
        if props['type'] == list:
            callbacks = [lambda *_, key=props['tag']: upd_param(key)] # Same callback setup as above
            if 'man_call' in props: callbacks.append(props['man_call'])
            callback = lambda *_, farr=callbacks: [f() for f in farr]
            with dpg.table_row():
                dpg.add_text(title)
                dpg.add_combo(tag=props['tag'], items=props['items'], callback=callback, width=-1)
            if 'default' in props: dpg.set_value(props['tag'], str(props['default']))
    
    def save_callback():
        print('saving', active_file)
        if active_file == None: # Save as
            dpg.show_item('save_as')
        else:
            save_config(active_file)
    
    def load_callback(sender, app_data):
        global active_file
        active_file = Path(app_data['file_path_name'])
        if active_file.exists():
            print('Loading', active_file)
            load_config(active_file)
        else:
            active_file = None
            print('Loaded file doesnt exist!')

    def save_as_callback(sender, app_data):
        global active_file
        active_file = Path(app_data['file_path_name'])
        print('saving as', active_file)
        save_config(active_file)
        # print('save as', app_data)
    
    def export_rse_callback(sender, app_data):
        thrust, prop_mdot = [xstack[:,method['xmap'][k]] for k in ['noz_thrust', 'noz_mdot']]
        OD, L = [get_param(k) for k in ['dry_OD', 'dry_L']]
        m, cg = calculate_m_Cg()
        core.export_rse(
            app_data['file_path_name'],
            t,  thrust, prop_mdot, m, cg,
            OD=OD, L=L, D_throat=s['noz_thrt'], D_exit=np.sqrt(s['noz_ER'])*s['noz_thrt'],
            motor_type='hybrid', mfg=dpg.get_value('mfg'),
        )
    
    def export_eng_callback(sender, app_data):
        thrust = xstack[:,method['xmap']['noz_thrust']]
        OD, L = [get_param(k) for k in ['dry_OD', 'dry_L']]
        m, _ = calculate_m_Cg()
        core.export_eng(
            app_data['file_path_name'],
            t, thrust, m,
            OD=OD, L=L,
            mfg=dpg.get_value('mfg'),
        )
    
    def export_csv_callback(sender, app_data):
        m, cg = calculate_m_Cg()
        csv_data = np.zeros((t.size, 3+len(method['xmap'])))
        header = 't, m, cg'
        csv_data[:, 0] = t
        csv_data[:, 1] = m
        csv_data[:, 2] = cg
        for i, k in enumerate(method['xmap']):
            csv_data[:, 3+i] = xstack[:,method['xmap'][k]]
            header += ',' + k
        np.savetxt(app_data['file_path_name'], csv_data, delimiter=',', header=header, comments='')

    def key_press_handler(sender, app_data):
        global active_file
        if dpg.is_key_down(dpg.mvKey_LControl) and app_data == dpg.mvKey_S:
            if dpg.is_key_down(dpg.mvKey_LShift): active_file = None # Save as
            save_callback()
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=key_press_handler)
    
    with dpg.window(label='menu', tag='menu', no_title_bar=True, menubar=True, no_bring_to_front_on_focus=True, **settings):
        with dpg.file_dialog(tag='load', default_filename='', directory_selector=False, show=False, width=700, height=400, callback=load_callback):
            dpg.add_file_extension('.hrap')
        with dpg.file_dialog(tag='save_as', default_filename='', directory_selector=False, show=False, width=700 ,height=400, callback=save_as_callback):
            dpg.add_file_extension('.hrap')
            # with dpg.child_window(height=100):
            #     dpg.add_selectable(label='bookmark 1')
            #     dpg.add_selectable(label='bookmark 2')
            #     dpg.add_selectable(label='bookmark 3')
        with dpg.file_dialog(tag='export_rse', default_filename='', directory_selector=False, show=False, width=700 ,height=400, callback=export_rse_callback):
            dpg.add_file_extension('.rse')
        with dpg.file_dialog(tag='export_eng', default_filename='', directory_selector=False, show=False, width=700 ,height=400, callback=export_eng_callback):
            dpg.add_file_extension('.eng')
        with dpg.file_dialog(tag='export_csv', default_filename='', directory_selector=False, show=False, width=700 ,height=400, callback=export_csv_callback):
            dpg.add_file_extension('.csv')
        
        with dpg.menu_bar():
            with dpg.menu(label='File'):
                # https://dearpygui.readthedocs.io/en/latest/documentation/file-directory-selector.html
                dpg.add_menu_item(label='Load',       callback=lambda: dpg.show_item('load'))
                dpg.add_menu_item(label='Save',       callback=lambda: save_callback())
                dpg.add_menu_item(label='Save As',    callback=lambda: dpg.show_item('save_as'))
                dpg.add_menu_item(label='Export RSE', callback=lambda: dpg.show_item('export_rse'))
                dpg.add_menu_item(label='Export ENG', callback=lambda: dpg.show_item('export_eng'))
                dpg.add_menu_item(label='Export CSV', callback=lambda: dpg.show_item('export_csv'))
            with dpg.menu(label='Config'):
                dpg.add_input_text(label='Manufacturer', tag='mfg', default_value='HRAP')
            with dpg.menu(label='Theme'):
                def apply_theme_callback(sender, app_data, user_data): apply_theme(user_data, True)
                for theme in themes: dpg.add_menu_item(label=theme, callback=apply_theme_callback, user_data=theme)
        
        
        col_w = [1.0, 1.0, 0.2]
        input_table_kwargs = core.make_dict(header_row=False, resizable=False, borders_innerV=False, borders_innerH=True, policy=dpg.mvTable_SizingStretchProp)
        # Make tank window
        with dpg.window(tag='tank', label='Tank', **settings):
            # diam_units = {}
            diam_steps = {'mm': 1.0, 'cm': 0.1, 'in': 1/16}
            diam_decim = {'mm': 4, 'cm': 3, 'm': 1, 'in': 3, 'ft': 5}
            with dpg.table(**input_table_kwargs):
                for i in range(3): dpg.add_table_column(init_width_or_weight=col_w[i])
                
                make_param('Inner Diameter', {
                    'type': float, 'units': 'mm',
                    'tag': 'tnk_D',
                    'min': 0.0,
                    'default': 4.75 * _in,
                    'step': diam_steps,
                    'decimal': 4,
                    'man_call': man_call_tnk_D,
                })
                make_param('Length', {
                    'type': float, 'units': 'm',
                    'tag': 'tnk_L',
                    'min': 0.0,
                    'default': 7 * _ft,
                    # 'step': 1E-2,
                    'decimal': 4,
                    'man_call': man_call_tnk_L,
                })
                make_param('Volume', {
                    'type': float, 'units': 'cc',
                    'tag': 'tnk_V', 'direct': True,
                    'min': 0.0,
                    # 'default': (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
                    'step': 1E-4,
                    'decimal': 6,
                    'man_call': man_call_tnk_V,
                })
                make_param('Injector Vapor Model', {
                    'type': list,
                    'tag': 'tnk_inj_vap_model',
                    'items': ['Real Gas', 'Incompressible'],
                    'default': 'Real Gas',
                    'man_call': recompile_motor,
                })
                make_param('Injector Liquid Model', {
                    'type': list,
                    'tag': 'tnk_inj_liq_model',
                    'items': ['Incompressible'],
                    'default': 'Incompressible',
                    'man_call': recompile_motor,
                })
                make_param('Injector Diameter', {
                    'type': float, 'units': 'mm',
                    'tag': 'tnk_inj_D',
                    'min': 1E-3,
                    'default': 0.5 * _in,
                    'step': 1E-3,
                    'decimal': 6,
                    'man_call': man_call_tnk_inj_D,
                })
                make_param('Injector Discharge Coefficient', {
                    'type': float,
                    'tag': 'tnk_inj_Cd',
                    'min': 1E-2,
                    'default': 0.22,
                    'step': 1E-2,
                    'decimal': 3,
                    'man_call': man_call_tnk_inj_Cd,
                })
                make_param('Injector CdA', {
                    'type': float, 'units': 'mm2',
                    'tag': 'tnk_inj_CdA', 'direct': True,
                    'min': 0.0, # Keep positive, diam limits
                    # 'default': 0.5 * (np.pi/4 * 0.5**2 * _in**2),
                    'step': 1E-6,
                    'decimal': 6,
                    'man_call': man_call_tnk_inj_CdA,
                })
                make_param('Injector Count', {
                    'type': int,
                    'tag': 'tnk_inj_N', 'direct': True,
                    'min': 1,
                    'default': 1,
                    'step': 1,
                })
                make_param('Oxidizer {}'.format(0+1), {
                    'type': list,
                    'tag': 'ox_component_{}'.format(0),
                    'items': list(oxidizers.keys()),
                    'default': 'Nitrous Oxide',
                    'man_call': recompile_motor,
                })
                make_param('Oxidizer Temperature', {
                    'type': float, 'units': 'K',
                    'tag': 'tnk_T', 'direct': True,
                    'min': 240.0,
                    'max': 305.0,
                    'default': 294.0,
                    'step': 1.0,
                    'decimal': 1,
                    'man_call': man_call_tnk_T,
                })
                make_param('Oxidizer Pressure', {
                    'type': float, 'units': 'kPa',
                    'tag': 'tnk_P',
                    # 'key': 'P',
                    # 'min': 1.0,
                    # 'max': 1E+3,
                    # 'default': 293.0,
                    'step': 10.0,
                    'decimal': 0,
                    'man_call': man_call_tnk_P,
                })
                make_param('Oxidizer Mass', {
                    'type': float, 'units': 'kg',
                    'tag': 'tnk_m_ox', 'direct': True, # TODO ..., actually change
                    # 'min': 1E-3, 'max': 1E+3,
                    # 'default': 14.0,
                    'step': 1E-1,
                    'decimal': 3,
                    'man_call': man_call_tnk_m_ox,
                })
                make_param('Oxidizer Fill [%]', {
                    'type': float,
                    'tag': 'tnk_fill',
                    # 'key': 'm_ox',
                    'min': 0.0, 'max': 100.0,
                    'default': 66.2,
                    'step': 5E-1,
                    'decimal': 1,
                    'man_call': man_call_tnk_fill,
                })
        
        # Make grain window
        with dpg.window(tag='grain', label='Grain', **settings):
            # show_item, hide_item
            with dpg.table(**input_table_kwargs):
                for i in range(3): dpg.add_table_column(init_width_or_weight=col_w[i])
                
                # with dpg.table_row():
                    # dpg.add_text('Grain Shape')
                    # # dpg.add_combo(tag='select_shape', items=['Cylindrical', 'Star', 'Custom'], default_value='Cylindrical', width=-1)
                    # dpg.add_combo(tag='select_shape', items=['Cylindrical'], default_value='Cylindrical', width=-1)
                make_param('Grain Shape', {
                    'type': list,
                    'tag': 'grn_shape',
                    'items': ['Cylindrical'], # ['Cylindrical', 'Star', 'Custom']
                    'default': 'Cylindrical',
                    # 'man_call': man_call_tnk_D,
                })
                make_param('Inner diamater', {
                    'type': float, 'units': 'mm',
                    'tag': 'grn_shape_ID', 'direct': True,
                    'min': 0.001,
                    'default': 2.5 * _in,
                    'step': 1E-3,
                    'decimal': 4,
                })
                make_param('Outer diamater', {
                    'type': float, 'units': 'mm',
                    'tag': 'grn_OD', 'direct': True,
                    'min': 0.001,
                    'default': 4.5 * _in,
                    'step': 1E-3,
                    'decimal': 4,
                })
                make_param('Length', {
                    'type': float, 'units': 'mm',
                    'tag': 'grn_L', 'direct': True,
                    'min': 0.001,
                    'default': 30.0 * _in,
                    'step': 1E-2,
                    'decimal': 4,
                })
                make_param('Rate Law', {
                    'type': list,
                    'tag': 'select_regression',
                    'items': ['Constant O/F'], # ['Constant O/F', 'Regression Rate']
                    'default': 'Constant O/F',
                    # 'man_call': man_call_tnk_D,
                })
                make_param('Fixed O/F ratio', {
                    'type': float,
                    'tag': 'grn_OF', 'direct': True,
                    'min': 0.01, 'max': 100.0,
                    'default': 3.5,
                    'step': 1E-1,
                    'decimal': 2,
                })
                with dpg.table_row():
                    dpg.add_checkbox(label='Use Estimated Density', tag='grn_est_rho', default_value=True, callback=lambda: [recompile_motor(), dpg.configure_item('grn_rho', readonly=dpg.get_value('grn_est_rho'))])
                make_param('Density', {
                    'type': float,
                    'tag': 'grn_rho', 'direct': True,
                    'min': 100.0,
                    'default': 1117.0,
                    'step': 10.0,
                    'decimal': 0,
                })
                dpg.configure_item('grn_rho', readonly=True)
                
                default_fu_components = ['Plastisol-362', 'Aluminum', 'None']
                default_fu_mfracs = [0.8, 0.2, 0.01]
                for i in range(N_fuel):
                    make_param('Component {}'.format(i+1), {
                        'type': list,
                        'tag': 'grn_component_{}'.format(i),
                        'items': (['None']if i>0 else [])+list(fuels.keys()),
                        'default': default_fu_components[i],
                        'man_call': recompile_motor,
                    })
                    make_param('Mass Fraction {}'.format(i+1), {
                        'type': float,
                        'tag': 'grn_mfrac_{}'.format(i),
                        'min': 0.01,
                        'default': default_fu_mfracs[i],
                        'step': 0.01,
                        'decimal': 3,
                        'man_call': recompile_motor,
                    })
                with dpg.table_row(): # TODO: only show when doesn't sum to 1...
                    dpg.add_text('Warning: mfrac normalization has occured')
        
        # Make chamber window
        with dpg.window(tag='chamber', label='Chamber', **settings):
            with dpg.table(**input_table_kwargs):
                for i in range(3): dpg.add_table_column(init_width_or_weight=col_w[i])
                
                make_param('Volume', {
                    'type': float, units: 'cc',
                    'tag': 'cmbr_V0', 'direct': True,
                    'min': 0.0,
                    'step': 1E-4,
                    'decimal': 6,
                })
                make_param('C* Efficiency', {
                    'type': float,
                    'tag': 'cmbr_cstar_eff', 'direct': True,
                    'min': 0.01, 'max': 1.0,
                    'default': 1.0,
                    'step': 1E-2,
                    'decimal': 2,
                })
        
        # Make misc window
        with dpg.window(tag='misc', label='Export Config', **settings):
            with dpg.table(**input_table_kwargs):
                for i in range(3): dpg.add_table_column(init_width_or_weight=col_w[i])
                
                make_param('Motor Outer Diameter', {
                    'type': float, 'units': 'mm',
                    'tag': 'dry_OD',
                    'min': 0.0,
                    'default': 5.0*_in,
                    'step': 1E-1,
                    'decimal': 6,
                })
                make_param('Motor Length', {
                    'type': float, 'units': 'mm',
                    'tag': 'dry_L',
                    'min': 0.0,
                    'default': 127.78*_in,
                    'step': 1E-1,
                    'decimal': 6,
                })
                make_param('Dry Mass', {
                    'type': float, 'units': 'kg',
                    'tag': 'dry_m',
                    'min': 0.0,
                    'default': 15.69,
                    'step': 1E-1,
                    'decimal': 6,
                })
                make_param('Dry Center of Gravity (from top)', {
                    'type': float, 'units': 'mm',
                    'tag': 'dry_cg',
                    'min': 0.0,
                    'default': 1.79,
                    'step': 1E-2,
                    'decimal': 6,
                })
                make_param('Oxidizer Position (from top)', {
                    'type': float, 'units': 'mm',
                    'tag': 'ox_pos',
                    'min': 0.0,
                    'default': 0.0,
                    'step': 1E-2,
                    'decimal': 6,
                })
                make_param('Grain Position (from top)', {
                    'type': float, 'units': 'mm',
                    'tag': 'grn_pos',
                    'min': 0.0,
                    'default': 7.0*_ft + 4.17*_in,
                    'step': 1E-2,
                    'decimal': 6,
                })
        
        # Make nozzle window
        with dpg.window(tag='nozzle', label='Nozzle', **settings):
            with dpg.table(**input_table_kwargs):
                for i in range(3): dpg.add_table_column(init_width_or_weight=col_w[i])
                
                make_param('Discharge Coefficient', {
                    'type': float,
                    'tag': 'noz_Cd', 'direct': True,
                    'min': 0.01, 'max': 1.0,
                    'default': 0.995,
                    'step': 1E-2,
                    'decimal': 3,
                })
                make_param('Efficiency', {
                    'type': float,
                    'tag': 'noz_eff', 'direct': True,
                    'min': 0.01, 'max': 1.0,
                    'default': 0.97,
                    'step': 1E-2,
                    'decimal': 3,
                })
                make_param('Throat Diameter [m]', {
                    'type': float, 'units': 'mm',
                    'tag': 'noz_thrt',
                    'key': 'thrt',
                    'min': 0.001,
                    'default': 1.75 * _in,
                    'step': 1E-3,
                    'decimal': 3,
                    'man_call': man_call_noz_thrt,
                })
                make_param('Exit Diameter', {
                    'type': float, 'units': 'mm',
                    'tag': 'noz_exit',
                    # 'key': None,
                    'min': 0.0, # Just keeps positive, Exit/Throat limits
                    # 'default': 5.0,
                    'step': 1E-3,
                    'decimal': 5,
                    'man_call': man_call_noz_D_exit,
                })
                make_param('Exit/Throat Area Ratio', {
                    'type': float,
                    'tag': 'noz_ER', 'direct': True,
                    'min': 1.001,
                    'default': 4.99,
                    'step': 1E-1,
                    'decimal': 3,
                    'man_call': man_call_noz_ER,
                })
            # TODO: atm pressure, button to optimize (based on ss, mid liq?)!
        
        i, preview_win_tag = 0, 'previewL'
        # in enumerate(['previewL', 'previewR']):
        with dpg.window(tag=preview_win_tag, label='Preview', **settings):
            # dpg.add_text('Bottom Right Section')
            # dpg.add_simple_plot(label='Simple Plot', min_scale=-1.0, max_scale=1.0, height=300, tag='plot')
            # create plot
            plt_tag = f'preview_{i}'
            with dpg.plot(tag=plt_tag, height=300, width=800):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label='t (s)')
                dpg.add_plot_axis(dpg.mvYAxis, label='Thrust (N)', tag=plt_tag+'_y_axis')
                dpg.add_line_series([], [], label='Total', parent=plt_tag+'_y_axis', tag=plt_tag+'_series')
        i, preview_win_tag = 1, 'previewR'
        with dpg.window(tag=preview_win_tag, label='Preview', **settings):
            plt_tag = f'preview_{i}'
            with dpg.plot(tag=plt_tag, height=300, width=800):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label='t (s)')
                dpg.add_plot_axis(dpg.mvYAxis, label='Pressure (Pa)', tag=plt_tag+'_y_axis')
                dpg.add_line_series([], [], label='Tank', parent=plt_tag+'_y_axis', tag=plt_tag+'_series')
    
    # Create initial internal motor
    recompile_motor()

    # part_configs = { 'cmbr': cmbr_config, 'noz': noz_config, 'tnk': tnk_config, 'grn': grain_config }
    # direct_tags = [ tag for tag in config if ('direct' in config[tag] and config[tag]['direct']) ]
    # init_deps()

    # resize_callback()
    dpg.set_viewport_resize_callback(resize_callback)
    
    upd_max_fps = 4
    upd_wall_dT = 1 / upd_max_fps # minimum time between relevant engine updates
    upd_wall_t = time.time() - 2*upd_wall_dT # time of last update
    
    max_fps = 24
    frame_wall_dT = 1/max_fps

    dpg.show_viewport()
    resize_callback()

    _unpack_engine = jax.jit(partial(core.unpack_engine, method=method))

    fps_wall_t = time.time()
    fps_i = 0
    while dpg.is_dearpygui_running():
        wall_t = time.time()

        if upd_due and wall_t - upd_wall_t >= upd_wall_dT:
            upd_due = False
            upd_wall_t = wall_t
        
            T = 10.0
            t10 = time.time()
            t, _, xstack = fire_engine(s, x, dt=1E-3, T=T)
            jax.block_until_ready(xstack)
            xstack = np.copy(xstack)
            # tnk, grn, cmbr, noz = _unpack_engine(s, xstack)
            
            N_t = xstack.shape[0]
            t2 = time.time()

            thrust = xstack[:,method['xmap']['noz_thrust']]
            tnk_P = xstack[:,method['xmap']['tnk_P']]
            # Copy is necessary as requires C-contiguous
            _t = np.copy(t[::10])
            dpg.set_value('preview_0_series', [_t, np.copy(thrust[::10])])
            dpg.set_value('preview_1_series', [_t, np.copy(tnk_P[::10])])
            print('max engine fps', 1/(t2-t10))
        
        # print('render')
        dpg.render_dearpygui_frame()
        # print('finish')
        
        wall_t_end = time.time()
        extra_time = frame_wall_dT - (wall_t_end - wall_t)
        if extra_time > 0.0:
            # print('sleep for', extra_time, frame_wall_dT, wall_t_end, wall_t)
            time.sleep(extra_time)

        # TODO: show on frame somewhere, or use dpg.get_frame_rate()
        # fps_i += 1
        # if wall_t >= fps_wall_t + 1.0:
            # print('FPS:', fps_i)#, '  freq', int(1/(t2-t1)))
            # fps_i = 0
            # fps_wall_t = wall_t # TODO: + modulus

    # dpg.start_dearpygui()
    dpg.destroy_context()
