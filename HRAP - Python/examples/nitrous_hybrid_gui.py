import dearpygui.dearpygui as dpg

# See https://github.com/hoffstadt/DearPyGui

def save_callback():
    print('Save Clicked')

dpg.create_context()
dpg.create_viewport(title='HRAP', width=800, height=600)
dpg.setup_dearpygui()
dpg.set_viewport_vsync(True)    

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
    dpg.add_text('Bottom Right Section')

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
        'step': 1E-2,
        'decimal': 2,
    },
    'Efficiency': {
        'type': float,
        'key': 'eff',
        'min': 0.01,
        'max': 1.0,
        'step': 1E-2,
        'decimal': 2,
    },
    'Throat Diameter [m]': {
        'type': float,
        'key': 'thrt',
        'min': 0.001,
        'step': 1E-2,
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
                # dpg.add_text(key)

make_part_window('Chamber', cmbr_config)
make_part_window('Nozzle', noz_config)
part_configs = [cmbr_config, noz_config]


# with dpg.window(tag='Nozzle', label='Nozzle', **settings):
    # dpg.add_text('Bottom Right Section')


resize_windows()
dpg.set_viewport_resize_callback(resize_windows)

# dpg.add_text('Output')
# dpg.add_input_text(label='file name')
# dpg.add_button(label='Save', callback=save_callback)
# dpg.add_slider_float(label='float')

dpg.show_viewport()

while dpg.is_dearpygui_running():
    for part_config in part_configs:
        for key, props in part_config.items():
            val = dpg.get_value(props['uuid'])
            if 'min' in props and val < props['min']:
                dpg.set_value(props['uuid'], props['min'])
            if 'max' in props and val > props['max']:
                dpg.set_value(props['uuid'], props['max'])
    
    dpg.render_dearpygui_frame()

# dpg.start_dearpygui()
dpg.destroy_context()
