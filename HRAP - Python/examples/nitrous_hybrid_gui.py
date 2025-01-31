import sys
sys.path.insert(1, '../HRAP/')
import time

import scipy
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

import dearpygui.dearpygui as dpg

from jax.scipy.interpolate import RegularGridInterpolator

import core
from tank    import *
from grain   import *
from chamber import *
from nozzle  import *
from sat_nos import *
from units   import _in, _ft

jax.config.update("jax_enable_x64", True)

# See https://github.com/hoffstadt/DearPyGui

def save_callback():
    print('Save Clicked')

dpg.create_context()
dpg.create_viewport(title='HRAP', width=800, height=600)
dpg.setup_dearpygui()
dpg.set_viewport_vsync(False)
# dpg.set_viewport_vsync(True)

def resize_windows():
    # Get the size of the main window
    main_width, main_height = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()

    # Update the size and position of each window based on the main window's size
    dpg.set_item_width ('Tank', main_width // 2)
    dpg.set_item_height('Tank', main_height // 3)
    dpg.set_item_pos   ('Tank', [0, 0])

    dpg.set_item_width ('Grain', main_width // 2)
    dpg.set_item_height('Grain', main_height // 3)
    dpg.set_item_pos   ('Grain', [0, main_height // 3])
    
    dpg.set_item_width ('Chamber', main_width // 2)
    dpg.set_item_height('Chamber', main_height // 3)
    dpg.set_item_pos   ('Chamber', [0, 2 * main_height // 3])
    
    dpg.set_item_width ('General', main_width // 2)
    dpg.set_item_height('General', main_height // 3)
    dpg.set_item_pos   ('General', [main_width // 2, 0])

    dpg.set_item_width ('Preview', main_width // 2)
    dpg.set_item_height('Preview', main_height // 3)
    dpg.set_item_pos   ('Preview', [main_width // 2, main_height // 3])
    
    dpg.set_item_width ('Nozzle', main_width // 2)
    dpg.set_item_height('Nozzle', main_height // 3)
    dpg.set_item_pos   ('Nozzle', [main_width // 2, 2 * main_height // 3])

# First row
settings = { 'no_move': True, 'no_collapse': True, 'no_resize': True, 'no_close': True }

with dpg.window(tag='General', label='General', **settings):
    dpg.add_input_text(label='file name')
    dpg.add_button(label='Save', callback=save_callback)

with dpg.window(tag='Preview', label='Preview', **settings):
    # dpg.add_text('Bottom Right Section')
    # dpg.add_simple_plot(label="Simple Plot", min_scale=-1.0, max_scale=1.0, height=300, tag="plot")
    # create plot
    with dpg.plot(label="Line Series", height=300, width=800):
        # optionally create legend
        dpg.add_plot_legend()

        # REQUIRED: create x and y axes
        dpg.add_plot_axis(dpg.mvXAxis, label="t (s)")
        dpg.add_plot_axis(dpg.mvYAxis, label="Thrust (N)", tag="y_axis")

        # series belong to a y axis
        dpg.add_line_series([], [], label="Trust", parent="y_axis", tag="series_tag")


with dpg.window(tag='Tank', label='Tank', **settings):
    dpg.add_text('Top Right Section')

with dpg.window(tag='Grain', label='Grain', **settings):
    dpg.add_text('Top Right Section')

cmbr_config = {
    'Base Volume [m^3]': {
        'type': float,
        'key': 'V',
        'min': 0.0,
        'step': 1E-5,
        'decimal': 6,
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
        'key': 'thrt',
        'min': 0.001,
        'default': 1.5 * _in,
        'step': 1E-3,
        'decimal': 3,
    },
    'Exit/Throat Area Ratio': {
        'type': float,
        'key': 'ER',
        'min': 1.001,
        'step': 1E-1,
        'decimal': 3,
    },
}

def make_part_window(name, part_config):
    for key in part_config:
        part_config[key]['uuid'] = dpg.generate_uuid()
    # print(name)
    with dpg.window(tag=name, label=name, **settings):
        for title, props in part_config.items():
            if props['type'] == float:
                decimal = props['decimal'] if 'decimal' in props else 3
                dpg.add_input_float(label=title, step=props['step'], format=f'%.{decimal}f', tag=props['uuid'])
                if 'default' in props:
                    dpg.set_value(props['uuid'], props['default'])
                # dpg.add_text(key)

make_part_window('Chamber', cmbr_config)
make_part_window('Nozzle', noz_config)
part_configs = [cmbr_config, noz_config]


# with dpg.window(tag='Nozzle', label='Nozzle', **settings):
    # dpg.add_text('Bottom Right Section')

chem = scipy.io.loadmat('../../propellant_configs/HTPB.mat')
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
    m_ox=14.0, # TODO: init limit
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

resize_windows()
dpg.set_viewport_resize_callback(resize_windows)

# dpg.add_text('Output')
# dpg.add_input_text(label='file name')
# dpg.add_button(label='Save', callback=save_callback)
# dpg.add_slider_float(label='float')

dpg.show_viewport()

_unpack_engine = jax.jit(partial(core.unpack_engine, method=method))

wall_t0 = time.time()
fps_i = 0
while dpg.is_dearpygui_running():
    
    t1 = time.time()
    for part_config in part_configs:
        for key, props in part_config.items():
            val = dpg.get_value(props['uuid'])
            if 'min' in props and val < props['min']:
                dpg.set_value(props['uuid'], props['min'])
            if 'max' in props and val > props['max']:
                dpg.set_value(props['uuid'], props['max'])
    # t2 = time.time()
    # print('v check took', t2-t1)
    
    
    s['noz_eff'] = dpg.get_value(noz_config['Efficiency']['uuid'])
    s['noz_thrt'] = dpg.get_value(noz_config['Throat Diameter [m]']['uuid'])
    T = 10.0
    t, x1, xstack = fire_engine(s, x, dt=1E-3, T=T)
    tnk, grn, cmbr, noz = _unpack_engine(s, xstack)
    # print(type(noz['thrust']))
    N_t = xstack.shape[0]
    t2 = time.time()
    # print('sim took', t2-t1)

    # 
    dpg.set_value('series_tag', [np.linspace(0.0, T, N_t), np.asarray(noz['thrust'])])

    # t1 = time.time()
    dpg.render_dearpygui_frame()
    t2 = time.time()
    # print('render took', t2-t1)

    fps_i += 1
    wall_t = time.time()
    if wall_t >= wall_t0 + 1.0:
        print('FPS:', fps_i, '  freq', int(1/(t2-t1)))
        fps_i = 0
        wall_t0 = wall_t # TODO: + modulus

# dpg.start_dearpygui()
dpg.destroy_context()
