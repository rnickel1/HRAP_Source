import sys
sys.path.insert(1, '../HRAP/')
import time

import scipy
import numpy as np
from pathlib import Path

# import matplotlib.pyplot as plt

import dearpygui.dearpygui as dpg

from jax.scipy.interpolate import RegularGridInterpolator

import hrap.core as core
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.sat_nos import *
from hrap.units   import _in, _ft

def create_babber_theme():# -> Union[str, int]:
    """
    Official HRAP babber theme
    The theme uses red, yellow, and white colors.
    """
    with dpg.theme() as theme_id:
        with dpg.theme_component(dpg.mvAll):
            # McDonald's Red (primary background and active elements)
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (220, 0, 0, 255)) # Main window background
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (200, 0, 0, 255)) # Child window background
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 50, 50, 255)) # Input fields, sliders, etc.
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (255, 80, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (200, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (200, 0, 0, 255)) # Inactive window title bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (255, 50, 50, 255)) # Active window title bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, (150, 0, 0, 255)) # Collapsed window title bar
            dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 50, 50, 255)) # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 80, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (200, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (255, 50, 50, 255)) # Collapsible headers, tree nodes
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (255, 80, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (200, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (100, 0, 0, 255)) # Separators
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, (150, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, (180, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, (255, 204, 0, 100)) # Resize grips (yellow for accent)
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, (255, 204, 0, 180))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, (255, 204, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (200, 0, 0, 255)) # Inactive tabs
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (255, 80, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (255, 50, 50, 255)) # Active tabs
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (150, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (180, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_DockingPreview, (255, 204, 0, 180)) # Docking preview (yellow)
            dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg, (100, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (255, 204, 0, 255)) # Slider grab (yellow)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (220, 180, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (255, 204, 0, 255)) # Checkbox mark (yellow)
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, (255, 204, 0, 90)) # Selected text background (yellow with transparency)
            dpg.add_theme_color(dpg.mvThemeCol_DragDropTarget, (255, 204, 0, 220)) # Drag and drop target (yellow)
            dpg.add_theme_color(dpg.mvThemeCol_NavHighlight, (255, 204, 0, 255)) # Keyboard navigation highlight (yellow)
            dpg.add_theme_color(dpg.mvThemeCol_NavWindowingHighlight, (255, 255, 255, 180)) # Windowing highlight (white)
            dpg.add_theme_color(dpg.mvThemeCol_NavWindowingDimBg, (255, 255, 255, 50)) # Windowing dim background (white)
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, (0, 0, 0, 150)) # Modal window dim background (darker)

            # McDonald's Yellow (text and highlights)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 204, 0, 255)) # Main text color
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (150, 100, 0, 255)) # Disabled text color

            # White (popups, menu bar, some borders)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (255, 255, 255, 255)) # Popup background
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (255, 255, 255, 255)) # Menu bar background
            dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 204, 0, 128)) # Borders (yellow with transparency)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, (0, 0, 0, 0)) # No border shadow

            # Scrollbar colors
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (200, 0, 0, 135)) # Scrollbar background (red with transparency)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (255, 204, 0, 255)) # Scrollbar grab (yellow)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (255, 220, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (220, 180, 0, 255))

            # Plot colors (adjusting to theme)
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg, (255, 255, 255, 20)) # Plot frame background (white with transparency)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (200, 0, 0, 128)) # Plot background (red with transparency)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (255, 204, 0, 128)) # Plot border (yellow with transparency)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, (255, 255, 255, 240)) # Legend background (white)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, (255, 204, 0, 128)) # Legend border (yellow)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText, (0, 0, 0, 255)) # Legend text (black for contrast)
            dpg.add_theme_color(dpg.mvPlotCol_TitleText, (255, 204, 0, 255)) # Plot title text (yellow)
            dpg.add_theme_color(dpg.mvPlotCol_InlayText, (255, 204, 0, 255)) # Plot inlay text (yellow)
            dpg.add_theme_color(dpg.mvPlotCol_AxisBg, (220, 0, 0, 0)) # Axis background (transparent red)
            dpg.add_theme_color(dpg.mvPlotCol_AxisBgActive, (255, 204, 0, 255)) # Active axis background (yellow)
            dpg.add_theme_color(dpg.mvPlotCol_AxisBgHovered, (255, 220, 50, 255)) # Hovered axis background (lighter yellow)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, (255, 255, 255, 128)) # Axis grid (white with transparency)
            dpg.add_theme_color(dpg.mvPlotCol_AxisText, (255, 204, 0, 255)) # Axis text (yellow)
            dpg.add_theme_color(dpg.mvPlotCol_Selection, (255, 204, 0, 255)) # Plot selection (yellow)
            dpg.add_theme_color(dpg.mvPlotCol_Crosshairs, (255, 255, 255, 128)) # Crosshairs (white with transparency)
            # dpg.add_theme_color(dpg.mvPlotCol_PlotLines, (255, 204, 0, 255)) # Plot lines (yellow)
            # dpg.add_theme_color(dpg.mvPlotCol_PlotLinesHovered, (255, 255, 50, 255))
            # dpg.add_theme_color(dpg.mvPlotCol_PlotHistogram, (255, 204, 0, 255)) # Plot histogram (yellow)
            # dpg.add_theme_color(dpg.mvPlotCol_PlotHistogramHovered, (255, 220, 50, 255))

            # Node Editor colors (adjusting to theme)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (220, 0, 0, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (255, 50, 50, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (255, 80, 80, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (255, 204, 0, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (255, 50, 50, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (255, 80, 80, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (200, 0, 0, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Link, (255, 204, 0, 200), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (255, 220, 50, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (220, 180, 0, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Pin, (255, 204, 0, 180), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (255, 220, 50, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelector, (255, 204, 0, 30), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelectorOutline, (255, 204, 0, 150), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridBackground, (180, 0, 0, 200), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridLine, (255, 255, 255, 40), category=dpg.mvThemeCat_Nodes)

            # Table colors (adjusting to theme)
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (255, 204, 0, 255)) # Table header background (yellow)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, (200, 0, 0, 255)) # Strong table border (red)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, (255, 50, 50, 255)) # Light table border (lighter red)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, (220, 0, 0, 0)) # Table row background (transparent red)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, (255, 255, 255, 15)) # Alternating table row background (slight white tint)

    return theme_id

def main():
    jax.config.update("jax_enable_x64", True)
    
    # See https://github.com/hoffstadt/DearPyGui

    dpg.create_context()
    dpg.create_viewport(title='HRAP', width=800, height=600) # small_icon='a.ico', large_icon='\a.ico'
    dpg.setup_dearpygui()
    dpg.set_viewport_vsync(False)

    babber_theme = create_babber_theme()
    dpg.bind_theme(babber_theme)

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

        dpg.set_item_width ('preview_1', main_width // 2 - 18)
        dpg.set_item_height('preview_1', main_height // 3 - 36)
        
        dpg.set_item_width ('Nozzle', main_width // 2)
        dpg.set_item_height('Nozzle', main_height // 3)
        dpg.set_item_pos   ('Nozzle', [main_width // 2, 2 * main_height // 3])

    # First row
    settings = { 'no_move': True, 'no_collapse': True, 'no_resize': True, 'no_close': True }

    with dpg.window(tag='General', label='General', **settings):
        dpg.add_input_text(label='Manufacturer')
        # def save_callback():
        #     print('Save Clicked')
        # dpg.add_button(label='Save', callback=save_callback)

        
        # https://dearpygui.readthedocs.io/en/latest/documentation/file-directory-selector.html
        dpg.add_file_dialog(directory_selector=True, show=False, tag="load")
        dpg.add_file_dialog(directory_selector=True, show=False, tag="save")
        dpg.add_file_dialog(directory_selector=True, show=False, tag="save_rse")
        dpg.add_file_dialog(directory_selector=True, show=False, tag="save_eng")

        with dpg.group(horizontal=True):
            dpg.add_button(label="Load", callback=lambda: dpg.show_item("load"))
            dpg.add_button(label="Save", callback=lambda: dpg.show_item("save"))
            dpg.add_button(label="Save As", callback=lambda: dpg.show_item("save"))
            dpg.add_button(label="Export RSE", callback=lambda: dpg.show_item("save_rse"))
            dpg.add_button(label="Export ENG", callback=lambda: dpg.show_item("save_eng"))

    with dpg.window(tag='Preview', label='Preview', **settings):
        # dpg.add_text('Bottom Right Section')
        # dpg.add_simple_plot(label="Simple Plot", min_scale=-1.0, max_scale=1.0, height=300, tag="plot")
        # create plot
        with dpg.plot(tag='preview_1', height=300, width=800):
            # optionally create legend
            dpg.add_plot_legend()

            # REQUIRED: create x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label="t (s)")
            dpg.add_plot_axis(dpg.mvYAxis, label="Thrust (N)", tag="y_axis")

            # series belong to a y axis
            dpg.add_line_series([], [], label="Thrust", parent="y_axis", tag="series_tag")


    # with dpg.window(tag='Tank', label='Tank', **settings):
        # dpg.add_text('Top Right Section')

    # with dpg.window(tag='Grain', label='Grain', **settings):
        # dpg.add_text('Top Right Section')

    tnk_config = {
        'Diameter': {
            'type': float,
            'min': 0.0,
            'step': 1E-3,
            'decimal': 4,
        },
        'Length': {
            'type': float,
            'min': 0.0,
            'step': 1E-3,
            'decimal': 4,
        },
        'Volume': {
            'type': float,
            'key': 'V',
            'min': 1E-9,
            'max': 1.0,
            'default': (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
            'step': 1E-4,
            'decimal': 6,
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
            'key': 'T',
            'min': 240.0, # Generously high, yet leaves room for applicabiltiy
            'max': 309.0, # Max applicability of sat nos
            'default': 293.0,
            'step': 5.0,
            'decimal': 0,
        },
        'Oxidizer Pressure': {
            'type': float,
            # 'key': 'P',
            # 'min': 1.0,
            # 'max': 1E+3,
            # 'default': 293.0,
            'step': 10.0E3,
            'decimal': 0,
        },
        'Oxidizer Mass': {
            'type': float,
            'key': 'm_ox',
            'min': 1E-3,
            'max': 1E+3,
            'default': 14.0,
            'step': 1E-1,
            'decimal': 3,
        },
        'Oxidizer Fill [%]': {
            'type': float,
            # 'key': 'm_ox',
            'min': 1.0,
            'max': 100.0,
            # 'default': 14.0,
            'step': 1E-1,
            'decimal': 1,
        },
        # V = (np.pi/4 * 5.0**2 * _in**2) * (10 * _ft),
        # inj_CdA= 0.5 * (np.pi/4 * 0.5**2 * _in**2),
        # m_ox=14.0
    }
    
    # TODO: add info descriptions!
    cmbr_config = {
        'Diameter': {
            'type': float,
            'min': 0.0,
            'step': 1E-3,
            'decimal': 4,
        },
        'Length': {
            'type': float,
            'min': 0.0,
            'step': 1E-3,
            'decimal': 4,
        },
        'Volume [m^3]': {
            'type': float,
            'key': 'V0',
            'min': 0.0,
            'step': 1E-4,
            'decimal': 6,
        },
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
            'key': 'thrt',
            'min': 0.001,
            'default': 1.5 * _in,
            'step': 1E-3,
            'decimal': 3,
        },
        'Throat Diameter': {
            'type': float,
            # 'key': None,
            'min': 0.001,
            # 'default': 5.0,
            'step': 1E-3,
            'decimal': 5,
        },
        'Exit/Throat Area Ratio': {
            'type': float,
            'key': 'ER',
            'min': 1.001,
            'default': 5.0,
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

    make_part_window ('Tank',    tnk_config)
    make_grain_window('Grain',   grain_config)
    make_part_window ('Chamber', cmbr_config)
    make_part_window ('Nozzle',  noz_config)
    part_configs = { 'cmbr': cmbr_config, 'noz': noz_config, 'tnk': tnk_config, 'grn': grain_config }


    # with dpg.window(tag='Nozzle', label='Nozzle', **settings):
        # dpg.add_text('Bottom Right Section')

    # chem = scipy.io.loadmat('../../propellant_configs/HTPB.mat')
    # import pkgutils
    # data_dir = Path(pkgutils.resolve_name('hrap.tank').__file__).parent
    # data_path = Path(data_dir , 'HTPB.mat')
    from importlib.resources import files as imp_files
    chem = scipy.io.loadmat(str(imp_files('hrap').joinpath('HTPB.mat')))
    
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

    resize_windows()
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
            t, x1, xstack = fire_engine(s, x, dt=1E-3, T=T)
            jax.block_until_ready(xstack)
            # tnk, grn, cmbr, noz = _unpack_engine(s, xstack)
            
            N_t = xstack.shape[0]
            t2 = time.time()

            thrust = xstack[:,method['xmap']['noz_thrust']]
            # print(t.shape) # What happened to arr?
            # dpg.set_value('series_tag', [np.asarray(t[::10]), np.asarray(thrust[::10])])
            dpg.set_value('series_tag', [np.linspace(0.0, T, N_t//10), np.asarray(thrust[::10])])
            print('max engine fps', 1/(t2-t10))
            # dpg.set_value('series_tag', [np.linspace(0.0, T, N_t), np.asarray(noz['thrust'])])

        dpg.render_dearpygui_frame()
        
        wall_t_end = time.time()
        extra_time = frame_wall_dT - (wall_t_end - wall_t)
        if extra_time > 0.0:
            time.sleep(extra_time)

        # TODO: show on frame somewhere, or use dpg.get_frame_rate()
        # fps_i += 1
        # if wall_t >= fps_wall_t + 1.0:
            # print('FPS:', fps_i)#, '  freq', int(1/(t2-t1)))
            # fps_i = 0
            # fps_wall_t = wall_t # TODO: + modulus

    # dpg.start_dearpygui()
    dpg.destroy_context()
