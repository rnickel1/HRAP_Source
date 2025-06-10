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
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.sat_nos import *
from hrap.units   import _in, _ft

from hrap.gui.themes import create_babber_theme
from dearpygui_ext.themes import create_theme_imgui_light, create_theme_imgui_dark

from hrap.sat_nos import get_sat_nos_props
get_sat_props = get_sat_nos_props

hrap_version = version('hrap')

# Virtualized Python environments may redirect these to other locations
# Note that Windows Store Python redirects this to %APPDATA%\Local\Packages\PythonSoftwareFoundation.Python.[some garbage]\LocalCache\Roaming
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
fire_engine  = None

def clamped_param(val, props):
    if 'min' in props and val < props['min']:
        return [props['min']]*2
    elif 'max' in props and val > props['max']:
        return [props['max']]*2
    return val, None

def upd_direct_param(k): # TODO: no clam version
    global upd_due, x

    v = dpg.get_value(k)
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
    v, clam = clamped_param(dpg.get_value(tag), props)
    if clam != None: dpg.set_value(tag, clam)
    if props['direct']: upd_direct_param(tag)
    return clam

def set_param(tag, val):
    dpg.set_value(tag, float(val))
    return upd_param(tag)

# Callbacks for manual adjustments (gets sets the UI components, not the motor)
def man_call_tnk_D():
    D, L, T, fill = [dpg.get_value(tag) for tag in ['tnk_D', 'tnk_L', 'tnk_T', 'tnk_fill']]
    props = get_sat_props(T)
    V = np.pi/4 * D**2 * L
    print('TANK', V, D, L)
    set_param('tnk_V', V)
    set_param('tnk_m_ox', fill/100.0 * props['rho_l'] * V)

man_call_tnk_L = man_call_tnk_D

def man_call_tnk_V():
    V, D, T, fill = [dpg.get_value(tag) for tag in ['tnk_V', 'tnk_D', 'tnk_T', 'tnk_fill']]
    props = get_sat_props(T)
    set_param('tnk_L', V / (np.pi/4 * D**2))
    set_param('tnk_m_ox', fill/100.0 * props['rho_l'] * V)

def man_call_tnk_T():
    T, V, fill = [dpg.get_value(tag) for tag in ['tnk_T', 'tnk_V', 'tnk_fill']]
    props = get_sat_props(T)
    set_param('tnk_P', props['Pv'])
    set_param('tnk_m_ox', fill/100.0 * props['rho_l'] * V)

def man_call_tnk_P():
    pass

def man_call_tnk_m_ox():
    T, V, m = [dpg.get_value(tag) for tag in ['tnk_T', 'tnk_V', 'tnk_m_ox']]
    props = get_sat_props(T)
    fill_clam = set_param('tnk_fill', 100.0 * m / (props['rho_l'] * V))
    if fill_clam != None: set_param('tnk_m_ox', fill_clam/100.0 * props['rho_l'] * V)

def man_call_tnk_fill():
    T, V, fill = [dpg.get_value(tag) for tag in ['tnk_T', 'tnk_V', 'tnk_fill']]
    props = get_sat_props(T)
    set_param('tnk_m_ox', fill/100.0 * props['rho_l'] * V)

def man_call_noz_thrt():
    D_throat, ER = [dpg.get_value(tag) for tag in ['noz_thrt', 'noz_ER']]
    set_param('noz_exit', np.sqrt(ER * D_throat**2))

def man_call_noz_D_exit():
    D_throat, D_exit = [dpg.get_value(tag) for tag in ['noz_thrt', 'noz_exit']]
    ER_clam = set_param('noz_ER', D_exit**2/D_throat**2)
    if ER_clam != None: set_param('noz_exit', np.sqrt(ER_clam * D_throat**2))

def man_call_noz_ER():
    ER, D_throat = [dpg.get_value(tag) for tag in ['noz_ER', 'noz_thrt']]
    set_param('noz_exit', np.sqrt(ER * D_throat**2))

def init_deps(): # Called after init/load to verify consistency
    man_call_tnk_D()
    man_call_tnk_T()
    # man_call_ox_m()
    man_call_noz_ER()

def load_preset_chem(name):
    chem = scipy.io.loadmat(hrap_root/'resources'/'propellant_configs'/name)
    
    # print(chem.keys())
    # print(chem['s'][0][0]['opt_OF'])
    chem = chem['s'][0][0]
    # Plastisol: gamma, m, name, OF, temp, ?, ?, T
    chem_OF = chem['prop_OF'].ravel()
    chem_Pc = chem['prop_Pc'].ravel()
    chem_k, chem_M, chem_T = chem['prop_k'], chem['prop_M'], chem['prop_T']
    # opt_OF, prop_Rho, prop_nm, prop_Reg
    # print(chem)
    # print(chem_OF.shape, chem_Pc.shape, chem_k.shape, chem_M.shape, chem_T.shape, chem_k[0])
    if chem_k.size == 1: chem_k = np.full_like(chem_T, chem_k.item())

    chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k, fill_value=1.4)
    chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M, fill_value=29.0)
    chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T, fill_value=293.0)

    return chem_interp_k, chem_interp_M, chem_interp_T

def setup_motor(tnk_inj_vap_model, tnk_inj_liq_model, chem_interp_k, chem_interp_M, chem_interp_T):
    # Initialization
    tnk = make_sat_tank(
        get_sat_nos_props,
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
        thrt = 1.5 * _in, # Throat diameter
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
    global upd_due, s, x, method, fire_engine
    # TOOD: skip recompile if combo already in some dict
    gcm = dpg.get_value('select_grain_chem_mode')
    if gcm == 'HRAP Presets':
        print(dpg.get_value('select_grain_chem_hrap_presets')+'.mat')
        chem_info = load_preset_chem(dpg.get_value('select_grain_chem_hrap_presets')+'.mat')
    s, x, method, fire_engine = setup_motor(dpg.get_value('tnk_inj_vap_model'), dpg.get_value('tnk_inj_liq_model'), *chem_info)
    upd_due = True
    # Need to respecify all internal variables based on config
    for tag, props in config.items():
        if props['direct']: upd_direct_param(tag)
    init_deps()

def main():
    global hrap_root, config, upd_due #, s, x, method

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
        for tag in config:
            save_config[tag] = dpg.get_value(tag)
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
        set_whxy('grain',   vw // 2, vh // 3,      0,       vh // 3    )
        set_whxy('chamber', vw // 2, vh // 3,      0,       2 * vh // 3)
        set_whxy('nozzle',  vw // 2, vh // 3,      vw // 2, 2 * vh // 3)
        set_whxy('preview', vw // 2, vh // 3,      vw // 2, vh // 3    )

        set_wh('preview_1', vw // 2 - 18, vh // 3 - 36)

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
            dpg.add_input_float(label=title, step=props['step'], format=f'%.{decimal}f', tag=props['tag'], callback=callback)
            if 'default' in props:
                dpg.set_value(props['tag'], props['default'])
            
            # Add unit selector
            if 'units' in props
                
            dpg.add_combo(items=[k for k in all_units[].keys()]], default_value='Real Gas', callback=recompile_motor)
    
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
        core.save_rse(
            app_data['file_path_name'],
            t, noz['thrust'], noz['mdot'], t*0, t*0,
            OD=OD, L=L, D_throat=s['noz_thrt'], D_exit=np.sqrt(s['noz_ER'])*s['noz_thrt'],
            motor_type='hybrid', mfg=dpg.get_value('mfg'),
        )
    
    def export_eng_callback(sender, app_data):
        core.save_eng(
            app_data['file_path_name'],
            t, noz['thrust'], t*0,
            OD=OD, L=L,
            mfg=dpg.get_value('mfg'),
        )

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
        with dpg.file_dialog(tag='save_rse', default_filename='', directory_selector=False, show=False, width=700 ,height=400, callback=export_rse_callback):
            dpg.add_file_extension('.rse')
        with dpg.file_dialog(tag='save_eng', default_filename='', directory_selector=False, show=False, width=700 ,height=400, callback=export_eng_callback):
            dpg.add_file_extension('.eng')
        
        with dpg.menu_bar():
            with dpg.menu(label='File'):
                # https://dearpygui.readthedocs.io/en/latest/documentation/file-directory-selector.html
                dpg.add_menu_item(label='Load',       callback=lambda: dpg.show_item('load'))
                dpg.add_menu_item(label='Save',       callback=lambda: save_callback())
                dpg.add_menu_item(label='Save As',    callback=lambda: dpg.show_item('save_as'))
                dpg.add_menu_item(label='Export RSE', callback=lambda: dpg.show_item('save_rse'))
                dpg.add_menu_item(label='Export ENG', callback=lambda: dpg.show_item('save_eng'))
            with dpg.menu(label='Config'):
                dpg.add_input_text(label='Manufacturer', tag='mfg', default_value='HRAP')
            with dpg.menu(label='Theme'):
                def apply_theme_callback(sender, app_data, user_data): apply_theme(user_data, True)
                for theme in themes: dpg.add_menu_item(label=theme, callback=apply_theme_callback, user_data=theme)
        
        
        
        # Make tank window
        with dpg.window(tag='tank', label='Tank', **settings):
            dpg.add_combo(label='Injector Vapor Model', tag='tnk_inj_vap_model', items=['Real Gas', 'Incompressible'], default_value='Real Gas', callback=recompile_motor)
            dpg.add_combo(label='Injector Liquid Model', tag='tnk_inj_liq_model', items=['Incompressible'], default_value='Incompressible', callback=recompile_motor)
            make_param('Inner Diameter', {
                'type': float,
                'tag': 'tnk_D',
                'min': 0.0,
                'default': 4.75 * _in,
                'step': 1E-3,
                'decimal': 4,
                'man_call': man_call_tnk_D,
            })
            make_param('Length', {
                'type': float,
                'tag': 'tnk_L',
                'min': 0.0,
                'default': 7 * _ft,
                'step': 1E-2,
                'decimal': 4,
                'man_call': man_call_tnk_L,
            })
            make_param('Volume', {
                'type': float,
                'tag': 'tnk_V', 'direct': True,
                'min': 0.0,
                # 'default': (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
                'step': 1E-4,
                'decimal': 6,
                'man_call': man_call_tnk_V,
            })

            # 'Injector CdA': {
            #     'type': float,
            #     'key': 'inj_CdA',
            #     'min': 1E-9,
            #     'default': 0.5 * (np.pi/4 * 0.5**2 * _in**2),
            #     'step': 1E-6,
            #     'decimal': 6,
            # },
            # 
            make_param('Oxidizer Temperature', {
                'type': float,
                'tag': 'tnk_T', 'direct': True,
                'min': 240.0, # Generously high, yet leaves room for applicabiltiy
                'max': 305.0, # 309 is max applicability of sat nos
                'default': 293.0,
                'step': 1.0,
                'decimal': 0,
                'man_call': man_call_tnk_T,
            })
            make_param('Oxidizer Pressure', {
                'type': float,
                'tag': 'tnk_P',
                # 'key': 'P',
                # 'min': 1.0,
                # 'max': 1E+3,
                # 'default': 293.0,
                'step': 10.0E3,
                'decimal': 0,
                'man_call': man_call_tnk_P,
            })
            make_param('Oxidizer Mass', {
                'type': float,
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
                'default': 70.0,
                'step': 5E-1,
                'decimal': 1,
                'man_call': man_call_tnk_fill,
            })
            # V = (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
            # inj_CdA= 0.5 * (np.pi/4 * 0.5**2 * _in**2),
            # m_ox=14.0
        
        # Make grain window
        with dpg.window(tag='grain', label='Grain', **settings):
            dpg.add_text('Geometry')
            dpg.add_combo(label='Grain Shape', tag='select_shape', items=['Cylindrical', 'Star', 'Custom'], default_value='Cylindrical')

            dpg.add_text('Regression')
            dpg.add_combo(label='Rate Law', tag='select_regression', items=['Constant O/F', 'Regression Rate'], default_value='Constant O/F')

            dpg.add_text('Chemistry')
            dpg.add_combo(label='Mode', tag='select_grain_chem_mode', items=['HRAP Presets', 'Other Preset', 'Custom'], default_value='HRAP Presets')
            dpg.add_combo(label='Preset', tag='select_grain_chem_hrap_presets', items=[
                'ABS', 'Asphalt', 'HDPE', 'HTPB_Paraffin', 'HTPB', 'Metalized_Plastisol', 'Paraffin', 'Sorbitol',
            ], default_value='HTPB', callback=recompile_motor)
            # show_item, hide_item
            
            make_param('Fixed O/F ratio', {
                'type': float,
                'tag': 'grn_OF', 'direct': True,
                'min': 0.01, 'max': 100.0,
                'default': 5.0,
                'step': 1E-1,
                'decimal': 2,
            })
            make_param('Density', {
                'type': float,
                'tag': 'grn_rho', 'direct': True,
                'min': 100.0,
                'default': 1117.0,
                'step': 10.0,
                'decimal': 0,
            })
            make_param('Inner diamater', {
                'type': float,
                'tag': 'grn_shape_ID', 'direct': True,
                'min': 0.001,
                'default': 2.0 * _in,
                'step': 1E-3,
                'decimal': 4,
            })
            make_param('Outer diamater', {
                'type': float,
                'tag': 'grn_OD', 'direct': True,
                'min': 0.001,
                'default': 5.0 * _in,
                'step': 1E-3,
                'decimal': 4,
            })
            make_param('Length', {
                'type': float,
                'tag': 'grn_L', 'direct': True,
                'min': 0.001,
                'default': 5.0 * _ft,
                'step': 1E-2,
                'decimal': 4,
            })
        
        # Make chamber window
        with dpg.window(tag='chamber', label='Chamber', **settings):
            # 'Diameter': {
                # 'type': float,
                # 'min': 0.0,
                # 'step': 1E-3,
                # 'decimal': 4,
            # },
            # 'Length': {
                # 'type': float,
                # 'min': 0.0,
                # 'step': 1E-3,
                # 'decimal': 4,
            # },
            make_param('Volume [m^3]', {
                'type': float,
                'tag': 'cmbr_V0', 'key': 'V0',
                'min': 0.0,
                'step': 1E-4,
                'decimal': 6,
            })
            make_param('C* Efficiency', {
                'type': float,
                'tag': 'cstar_eff', 'key': 'cstar_eff',
                'min': 0.01, 'max': 1.0,
                'default': 0.95,
                'step': 1E-2,
                'decimal': 2,
            })
        
        # Make nozzle window
        with dpg.window(tag='nozzle', label='Nozzle', **settings):
            make_param('Discharge Coefficient', {
                'type': float,
                'tag': 'noz_Cd', 'direct': True,
                'min': 0.01, 'max': 1.0,
                'default': 0.9,
                'step': 1E-2,
                'decimal': 2,
            })
            make_param('Efficiency', {
                'type': float,
                'tag': 'noz_eff', 'direct': True,
                'min': 0.01, 'max': 1.0,
                'default': 0.9,
                'step': 1E-2,
                'decimal': 2,
            })
            make_param('Throat Diameter [m]', {
                'type': float,
                'tag': 'noz_thrt',
                'key': 'thrt',
                'min': 0.001,
                'default': 1.5 * _in,
                'step': 1E-3,
                'decimal': 3,
                'man_call': man_call_noz_thrt,
            })
            make_param('Exit Diameter', {
                'type': float,
                'tag': 'noz_exit',
                # 'key': None,
                # 'min': 0.001,
                # 'default': 5.0,
                'step': 1E-3,
                'decimal': 5,
                'man_call': man_call_noz_D_exit,
            })
            make_param('Exit/Throat Area Ratio', {
                'type': float,
                'tag': 'noz_ER', 'direct': True,
                'min': 1.001,
                'default': 5.0,
                'step': 1E-1,
                'decimal': 3,
                'man_call': man_call_noz_ER,
            })
            # TODO: atm pressure, button to optimize (based on ss, mid liq?)!
        
        with dpg.window(tag='preview', label='Preview', **settings):
            # dpg.add_text('Bottom Right Section')
            # dpg.add_simple_plot(label='Simple Plot', min_scale=-1.0, max_scale=1.0, height=300, tag='plot')
            # create plot
            with dpg.plot(tag='preview_1', height=300, width=800):
                # optionally create legend
                dpg.add_plot_legend()

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label='t (s)')
                dpg.add_plot_axis(dpg.mvYAxis, label='Thrust (N)', tag='y_axis')

                # series belong to a y axis
                dpg.add_line_series([], [], label='Thrust', parent='y_axis', tag='series_tag')
    
    # Create initial internal motor
    recompile_motor()

    # part_configs = { 'cmbr': cmbr_config, 'noz': noz_config, 'tnk': tnk_config, 'grn': grain_config }
    # direct_tags = [ tag for tag in config if ('direct' in config[tag] and config[tag]['direct']) ]
    init_deps()

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
            # print('run 1')
            # t, x1, xstack = fire_engine(s, x, dt=1E-3, T=T)
            _, _, xstack = fire_engine(s, x, dt=1E-3, T=T)
            # print('run 2')
            jax.block_until_ready(xstack)
            xstack = np.copy(xstack)
            # print('run 3')
            # tnk, grn, cmbr, noz = _unpack_engine(s, xstack)
            
            N_t = xstack.shape[0]
            t2 = time.time()

            thrust = xstack[:,method['xmap']['noz_thrust']]
            # print(t.shape) # What happened to arr?
            # dpg.set_value('series_tag', [np.asarray(t[::10]), np.asarray(thrust[::10])])
            dpg.set_value('series_tag', [np.linspace(0.0, T, N_t//10), np.copy(thrust[::10])])
            print('max engine fps', 1/(t2-t10))
            # dpg.set_value('series_tag', [np.linspace(0.0, T, N_t), np.asarray(noz['thrust'])])

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
