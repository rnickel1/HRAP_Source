import os
import sys
# sys.path.insert(1, '../HRAP/')
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

from hrap.examples.gui_themes import create_babber_theme
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
active_file = None

def main():
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
    dpg.create_viewport(title='HRAP', width=settings['view_w'], height=settings['view_h'], x_pos=settings['view_x'], y_pos=settings['view_y']) # small_icon='a.ico', large_icon='\a.ico'
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
    def resize_windows():
        # Get the size of the main window
        view_w, view_h = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        view_x, view_y = dpg.get_viewport_pos()
        menu_height = 20
        
        settings['view_w'] = view_w; settings['view_h'] = view_h
        settings['view_x'] = view_x; settings['view_y'] = view_y
        save_settings(settings)

        # Update the size and position of each window based on the main window's size
        dpg.set_item_width ('menu', view_w)
        # dpg.set_item_height('menu', menu_height)
        
        dpg.set_item_width ('Tank', view_w // 2)
        dpg.set_item_height('Tank', view_h // 3 - menu_height)
        dpg.set_item_pos   ('Tank', [0, menu_height])

        dpg.set_item_width ('Grain', view_w // 2)
        dpg.set_item_height('Grain', view_h // 3)
        dpg.set_item_pos   ('Grain', [0, view_h // 3])
        
        dpg.set_item_width ('Chamber', view_w // 2)
        dpg.set_item_height('Chamber', view_h // 3)
        dpg.set_item_pos   ('Chamber', [0, 2 * view_h // 3])

        dpg.set_item_width ('Preview', view_w // 2)
        dpg.set_item_height('Preview', view_h // 3)
        dpg.set_item_pos   ('Preview', [view_w // 2, view_h // 3])

        dpg.set_item_width ('preview_1', view_w // 2 - 18)
        dpg.set_item_height('preview_1', view_h // 3 - 36)
        
        dpg.set_item_width ('Nozzle', view_w // 2)
        dpg.set_item_height('Nozzle', view_h // 3)
        dpg.set_item_pos   ('Nozzle', [view_w // 2, 2 * view_h // 3])

    # First row
    settings = { 'no_move': True, 'no_collapse': True, 'no_resize': True, 'no_close': True }
    
    config = { }
    def set_param(tag, val):
        props = config[tag]
        if 'min' in props and val < props['min']:
            dpg.set_value(tag, props['min'])
        elif 'max' in props and val > props['max']:
            dpg.set_value(tag, props['max'])
        else:
            dpg.set_value(tag, float(val))
    
    # Callbacks for manual adjustments (gets sets the UI components, not the motor)
    def man_call_ox_D():
        D, L, T, fill = [dpg.get_value(tag) for tag in ['ox_D', 'ox_L', 'ox_T', 'ox_fill']]
        props = get_sat_props(T)
        V = np.pi/4 * D**2 * L
        set_param('ox_V', V)
        set_param('ox_m', fill/100.0 * props['rho_l'] * V)
    
    man_call_ox_L = man_call_ox_D
    
    def man_call_ox_V():
        V, D, T, fill = [dpg.get_value(tag) for tag in ['ox_V', 'ox_D', 'ox_T', 'ox_fill']]
        props = get_sat_props(T)
        set_param('ox_L', V / (np.pi/4 * D**2))
        set_param('ox_m', fill/100.0 * props['rho_l'] * V)
    
    def man_call_ox_T():
        T, V, fill = [dpg.get_value(tag) for tag in ['ox_T', 'ox_V', 'ox_fill']]
        props = get_sat_props(T)
        set_param('ox_P', props['Pv'])
        set_param('ox_m', fill/100.0 * props['rho_l'] * V)
    
    def man_call_ox_P():
        pass
    
    def man_call_ox_m():
        T, V, m = [dpg.get_value(tag) for tag in ['ox_T', 'ox_V', 'ox_m']]
        props = get_sat_props(T)
        set_param('ox_fill', 100.0 * m / (props['rho_l'] * V))
    
    def man_call_ox_fill():
        T, V, fill = [dpg.get_value(tag) for tag in ['ox_T', 'ox_V', 'ox_fill']]
        props = get_sat_props(T)
        set_param('ox_m', fill/100.0 * props['rho_l'] * V)
    
    def man_call_noz_D_exit():
        D_throat, D_exit = [dpg.get_value(tag) for tag in ['noz_throat', 'noz_exit']]
        set_param('noz_AR', D_exit**2/D_throat**2)
    
    def man_call_noz_AR():
        D_AR, D_throat = [dpg.get_value(tag) for tag in ['noz_AR', 'noz_throat']]
        set_param('noz_exit', np.sqrt(D_AR * D_throat**2))
    
    def init_deps(): # Called after init/load to verify consistency
        man_call_ox_D()
        man_call_ox_T()
        # man_call_ox_m()
        man_call_noz_AR()

    tnk_config = {
        'Inner Diameter': {
            'type': float,
            'tag': 'ox_D',
            'min': 0.0,
            'default': 4.75 * _in,
            'step': 1E-3,
            'decimal': 4,
            'man_call': man_call_ox_D,
        },
        'Length': {
            'type': float,
            'tag': 'ox_L',
            'min': 0.0,
            'default': 7 * _ft,
            'step': 1E-2,
            'decimal': 4,
            'man_call': man_call_ox_L,
        },
        'Volume': {
            'type': float,
            'tag': 'ox_V',
            'key': 'V',
            'min': 1E-9,
            'max': 1.0,
            # 'default': (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
            'step': 1E-4,
            'decimal': 6,
            'man_call': man_call_ox_V,
        },

        # 'Injector CdA': {
        #     'type': float,
        #     'key': 'inj_CdA',
        #     'min': 1E-9,
        #     'default': 0.5 * (np.pi/4 * 0.5**2 * _in**2),
        #     'step': 1E-6,
        #     'decimal': 6,
        # },
        # 
        'Oxidizer Temperature': {
            'type': float,
            'tag': 'ox_T',
            'key': 'T',
            'min': 240.0, # Generously high, yet leaves room for applicabiltiy
            'max': 305.0, # 309 is max applicability of sat nos
            'default': 293.0,
            'step': 1.0,
            'decimal': 0,
            'man_call': man_call_ox_T,
        },
        'Oxidizer Pressure': {
            'type': float,
            'tag': 'ox_P',
            # 'key': 'P',
            # 'min': 1.0,
            # 'max': 1E+3,
            # 'default': 293.0,
            'step': 10.0E3,
            'decimal': 0,
            'man_call': man_call_ox_P,
        },
        'Oxidizer Mass': {
            'type': float,
            'tag': 'ox_m', # TODO ..., actually change below
            'key': 'm_ox',
            'min': 1E-3,
            'max': 1E+3,
            # 'default': 14.0,
            'step': 1E-1,
            'decimal': 3,
            'man_call': man_call_ox_m,
        },
        'Oxidizer Fill [%]': {
            'type': float,
            'tag': 'ox_fill',
            # 'key': 'm_ox',
            'min': 1.0,
            'max': 100.0,
            'default': 70.0,
            'step': 5E-1,
            'decimal': 1,
            'man_call': man_call_ox_fill,
        },
        # V = (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
        # inj_CdA= 0.5 * (np.pi/4 * 0.5**2 * _in**2),
        # m_ox=14.0
    }
    
    # TODO: add info descriptions!
    cmbr_config = {
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
        'Volume [m^3]': {
            'type': float,
            'key': 'V0',
            'min': 0.0,
            'step': 1E-4,
            'decimal': 6,
        },
        'C* Efficiency': {
            'type': float,
            'key': 'cstar_eff',
            'min': 0.01,
            'max': 1.0,
            'default': 0.95,
            'step': 1E-2,
            'decimal': 2,
        }
    }
    
    grain_config = {
        'Fixed O/F ratio': {
            'type': float,
            'key': 'OF',
            'min': 0.01,
            'max': 100.0,
            'default': 5.0,
            'step': 1E-1,
            'decimal': 2,
        },
        'Inner diamater': {
            'type': float,
            # 'key': 'OD',
            'min': 0.001,
            'default': 2.0 * _in,
            'step': 1E-3,
            'decimal': 4,
        },
        'Outer diamater': {
            'type': float,
            'key': 'OD',
            'min': 0.001,
            'default': 5.0 * _in,
            'step': 1E-3,
            'decimal': 4,
        },
        'Length': {
            'type': float,
            'key': 'L',
            'min': 0.001,
            'default': 5.0 * _ft,
            'step': 1E-2,
            'decimal': 4,
        },
    }

    noz_config = {
        'Discharge Coefficient': {
            'type': float,
            'key': 'Cd',
            'min': 0.01,
            'max': 1.0,
            'default': 0.9,
            'step': 1E-2,
            'decimal': 2,
        },
        'Efficiency': {
            'type': float,
            'key': 'eff',
            'min': 0.01,
            'max': 1.0,
            'default': 0.9,
            'step': 1E-2,
            'decimal': 2,
        },
        'Throat Diameter [m]': {
            'type': float,
            'tag': 'noz_throat',
            'key': 'thrt',
            'min': 0.001,
            'default': 1.5 * _in,
            'step': 1E-3,
            'decimal': 3,
        },
        'Exit Diameter': {
            'type': float,
            'tag': 'noz_exit',
            # 'key': None,
            'min': 0.001,
            # 'default': 5.0,
            'step': 1E-3,
            'decimal': 5,
            'man_call': man_call_noz_D_exit,
        },
        'Exit/Throat Area Ratio': {
            'type': float,
            'tag': 'noz_AR',
            'key': 'ER',
            'min': 1.001,
            'default': 5.0,
            'step': 1E-1,
            'decimal': 3,
            'man_call': man_call_noz_AR,
        },
        # TODO: atm pressure, button to optimize (based on ss, mid liq?)!
    }

    def make_part_window(name, part_config):
        for key in part_config:
            if 'tag' in part_config[key]:
                part_config[key]['uuid'] = part_config[key]['tag']
                config[part_config[key]['tag']] = part_config[key]
            else:
                part_config[key]['uuid'] = dpg.generate_uuid() # TODO just tag
        # print(name)
        with dpg.window(tag=name, label=name, **settings):
            for title, props in part_config.items():
                if props['type'] == float:
                    decimal = props['decimal'] if 'decimal' in props else 3
                    callback = props['man_call'] if 'man_call' in props else None
                    dpg.add_input_float(label=title, step=props['step'], format=f'%.{decimal}f', tag=props['uuid'], callback=callback)
                    if 'default' in props:
                        dpg.set_value(props['uuid'], props['default'])
                    # dpg.add_text(key)
    
    def make_grain_window(name, part_config):
        for key in part_config:
            part_config[key]['uuid'] = dpg.generate_uuid()
        with dpg.window(tag=name, label=name, **settings):
            dpg.add_text('Geometry')
            dpg.add_combo(label='Grain Shape', tag='select_shape', items=['Cylindrical', 'Star', 'Custom'], default_value='Cylindrical')

            dpg.add_text('Regression')
            dpg.add_combo(label='Rate Law', tag='select_regression', items=['Constant O/F', 'Regression Rate'], default_value='Constant O/F')
            # show_item, hide_item

            for title, props in part_config.items():
                if props['type'] == float:
                    decimal = props['decimal'] if 'decimal' in props else 3
                    dpg.add_input_float(label=title, step=props['step'], format=f'%.{decimal}f', tag=props['uuid'])
                    if 'default' in props:
                        dpg.set_value(props['uuid'], props['default'])
    
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
        with dpg.file_dialog(tag='save_rse', default_filename='', directory_selector=False, show=False, width=700 ,height=400):
            dpg.add_file_extension('.rse')
        with dpg.file_dialog(tag='save_eng', default_filename='', directory_selector=False, show=False, width=700 ,height=400):
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
                dpg.add_input_text(label='Manufacturer')
            with dpg.menu(label='Theme'):
                def apply_theme_callback(sender, app_data, user_data): apply_theme(user_data, True)
                for theme in themes: dpg.add_menu_item(label=theme, callback=apply_theme_callback, user_data=theme)
        
        with dpg.window(tag='Preview', label='Preview', **settings):
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
                dpg.add_line_series([], [], label='Trust', parent='y_axis', tag='series_tag')
        
        make_part_window ('Tank',    tnk_config)
        make_grain_window('Grain',   grain_config)
        make_part_window ('Chamber', cmbr_config)
        make_part_window ('Nozzle',  noz_config)
    
    part_configs = { 'cmbr': cmbr_config, 'noz': noz_config, 'tnk': tnk_config, 'grn': grain_config }
    
    init_deps()


    # with dpg.window(tag='Nozzle', label='Nozzle', **settings):
        # dpg.add_text('Bottom Right Section')

    # chem = scipy.io.loadmat('../../propellant_configs/HTPB.mat')
    # import pkgutils
    # data_dir = Path(pkgutils.resolve_name('hrap.tank').__file__).parent
    # data_path = Path(data_dir , 'HTPB.mat')
    chem = scipy.io.loadmat(hrap_root/'resources'/'propellant_configs'/'HTPB.mat')
    
    chem = chem['s'][0][0]
    chem_OF = chem[1].ravel()
    chem_Pc = chem[0].ravel()
    chem_k = chem[2]
    chem_M = chem[3]
    chem_T = chem[4]

    chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k, fill_value=1.4)
    chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M, fill_value=29.0)
    chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T, fill_value=293.0)

    # Initialization
    tnk = make_sat_tank(
        get_sat_nos_props,
        V = (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
        inj_CdA= 0.5 * (np.pi/4 * 0.5**2 * _in**2),
        m_ox=1,#14.0, # TODO: init limit
        # m_ox = 3.0,
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

    fire_engine = core.make_integrator(
        # core.step_rk4,
        core.step_fe,
        method,
    )

    # resize_windows()
    dpg.set_viewport_resize_callback(resize_windows)
    
    upd_max_fps = 4
    upd_wall_dT = 1 / upd_max_fps # minimum time between relevant engine updates
    upd_wall_t = time.time() - 2*upd_wall_dT # time of last update
    upd_due = True
    
    max_fps = 24
    frame_wall_dT = 1/max_fps
    

    # dpg.add_text('Output')
    # dpg.add_input_text(label='file name')
    # dpg.add_button(label='Save', callback=save_callback)
    # dpg.add_slider_float(label='float')

    dpg.show_viewport()
    resize_windows()

    _unpack_engine = jax.jit(partial(core.unpack_engine, method=method))

    fps_wall_t = time.time()
    fps_i = 0
    while dpg.is_dearpygui_running():
        wall_t = time.time()
        # print('begin')
        
        # t1 = time.time()
        # TODO: can be done in callbacks?
        for part_name, part_config in part_configs.items():
            for key, props in part_config.items():
                val = dpg.get_value(props['uuid'])
                if 'min' in props and val < props['min']:
                    dpg.set_value(props['uuid'], props['min'])
                if 'max' in props and val > props['max']:
                    dpg.set_value(props['uuid'], props['max'])
            
            for value_config in part_config.values():
                if 'key' in value_config:
                    # print('set', part_name+'_'+value_config['key'], s[part_name+'_'+value_config['key']], '->', dpg.get_value(value_config['uuid']))
                    k = part_name+'_'+value_config['key']
                    v = dpg.get_value(value_config['uuid'])
                    if k in s:
                        if s[k] != v:
                            s[k] = v
                            upd_due = True
                            print('update due to s', k)
                    elif k in method['xmap']:
                        if x[method['xmap'][k]] != v:
                            print('update due to x', k)
                            x = x.at[method['xmap'][k]].set(v)
                            upd_due = True
                    else:
                        print('ERROR:', k, 'is nowhere!')
        # t2 = time.time()
        # print('v check took', t2-t1)
        
        
        # s['noz_eff'] = dpg.get_value(noz_config['Efficiency']['uuid'])
        # s['noz_thrt'] = dpg.get_value(noz_config['Throat Diameter [m]']['uuid'])
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
