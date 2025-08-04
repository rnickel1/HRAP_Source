import dearpygui.dearpygui as dpg

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
