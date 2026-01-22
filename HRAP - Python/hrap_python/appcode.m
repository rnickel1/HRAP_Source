classdef HRAP < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        HRAP_v2022_06_04                matlab.ui.Figure
        ReleaseLabel                    matlab.ui.control.Label
        AuthorLabel                     matlab.ui.control.Label
        AppTabs                         matlab.ui.container.TabGroup
        MotorTab                        matlab.ui.container.Tab
        SaveMotor                       matlab.ui.control.Button
        LoadMotor                       matlab.ui.control.Button
        RunSimulation                   matlab.ui.control.Button
        InitialConditionsPanel          matlab.ui.container.Panel
        m_O_unit                        matlab.ui.control.DropDown
        Pa_unit                         matlab.ui.control.DropDown
        P_cmbr_unit                     matlab.ui.control.DropDown
        T_tnk_unit                      matlab.ui.control.DropDown
        OxidizerDropDown                matlab.ui.control.DropDown
        TankDropDown                    matlab.ui.control.DropDown
        AmbientPressure                 matlab.ui.control.NumericEditField
        AmbientPressureEditField_2Label  matlab.ui.control.Label
        OxidizerFill                    matlab.ui.control.NumericEditField
        ChamberPressure                 matlab.ui.control.NumericEditField
        StartingChamberPressureEditFieldLabel  matlab.ui.control.Label
        TankCond                        matlab.ui.control.NumericEditField
        SimulationConfigurationPanel    matlab.ui.container.Panel
        timestep_label                  matlab.ui.control.Label
        burntime_label                  matlab.ui.control.Label
        runtime_label                   matlab.ui.control.Label
        RegressionModel                 matlab.ui.control.DropDown
        RegressionModelDropDownLabel    matlab.ui.control.Label
        Timestep                        matlab.ui.control.NumericEditField
        SimulationTimestepEditFieldLabel  matlab.ui.control.Label
        BurnTime                        matlab.ui.control.NumericEditField
        MaxBurnTimeEditFieldLabel       matlab.ui.control.Label
        RunTime                         matlab.ui.control.NumericEditField
        MaxSimulationRunTimeEditFieldLabel  matlab.ui.control.Label
        MassPropertiesPanel             matlab.ui.container.Panel
        mtr_m_unit                      matlab.ui.control.DropDown
        mtr_cg_unit                     matlab.ui.control.DropDown
        cmbr_X_unit                     matlab.ui.control.DropDown
        tnk_X_unit                      matlab.ui.control.DropDown
        MassProperties                  matlab.ui.control.CheckBox
        MotorMass                       matlab.ui.control.NumericEditField
        EmptyMotorMassEditFieldLabel    matlab.ui.control.Label
        MotorCG                         matlab.ui.control.NumericEditField
        EmptyMotorCenterofMassEditFieldLabel  matlab.ui.control.Label
        GrainLocation                   matlab.ui.control.NumericEditField
        FuelGrainLocationEditFieldLabel  matlab.ui.control.Label
        TankLocation                    matlab.ui.control.NumericEditField
        OxidizerTankLocationEditFieldLabel  matlab.ui.control.Label
        InjectorConfigurationPanel      matlab.ui.container.Panel
        inj_D_unit                      matlab.ui.control.DropDown
        vnt_D_unit                      matlab.ui.control.DropDown
        VentState                       matlab.ui.control.DropDown
        VentStateDropDownLabel          matlab.ui.control.Label
        VentCd                          matlab.ui.control.NumericEditField
        VentDischargeCoefficientEditFieldLabel  matlab.ui.control.Label
        VentDiameter                    matlab.ui.control.NumericEditField
        VentDiameterEditFieldLabel      matlab.ui.control.Label
        NumberofInjectors               matlab.ui.control.NumericEditField
        NumberofInjectorsEditFieldLabel  matlab.ui.control.Label
        InjectorCd                      matlab.ui.control.NumericEditField
        InjectorDischargeCoefficientEditFieldLabel  matlab.ui.control.Label
        InjectorDiameter                matlab.ui.control.NumericEditField
        InjectorDiameterEditFieldLabel  matlab.ui.control.Label
        PropellantConfigurationPanel    matlab.ui.container.Panel
        m_unit                          matlab.ui.control.Label
        n_unit                          matlab.ui.control.Label
        a_unit                          matlab.ui.control.Label
        grnL_unit                       matlab.ui.control.DropDown
        grnOD_unit                      matlab.ui.control.DropDown
        grnID_unit                      matlab.ui.control.DropDown
        density_unit                    matlab.ui.control.DropDown
        cstar_percent                   matlab.ui.control.Label
        ConstantOF                      matlab.ui.control.NumericEditField
        ConstantOFRatioLabel            matlab.ui.control.Label
        LoadPropellantConfig            matlab.ui.control.Button
        GrainLength                     matlab.ui.control.NumericEditField
        GrainLengthEditFieldLabel       matlab.ui.control.Label
        GrainOD                         matlab.ui.control.NumericEditField
        GrainODEditFieldLabel           matlab.ui.control.Label
        GrainID                         matlab.ui.control.NumericEditField
        GrainIDEditFieldLabel           matlab.ui.control.Label
        CEfficiency                     matlab.ui.control.NumericEditField
        CEfficiencyEditFieldLabel       matlab.ui.control.Label
        LengthExponent                  matlab.ui.control.NumericEditField
        LengthExponentmEditFieldLabel   matlab.ui.control.Label
        RegressionExponent              matlab.ui.control.NumericEditField
        RegressionExponentnEditFieldLabel  matlab.ui.control.Label
        RegressionCoefficient           matlab.ui.control.NumericEditField
        RegressionCoefficientaEditFieldLabel  matlab.ui.control.Label
        PropellantDensity               matlab.ui.control.NumericEditField
        PropellantDensityEditFieldLabel  matlab.ui.control.Label
        PropellantName                  matlab.ui.control.EditField
        PropellantNameEditFieldLabel    matlab.ui.control.Label
        NozzleConfigurationPanel        matlab.ui.container.Panel
        NozzleDropDown                  matlab.ui.control.DropDown
        noz_ext_unit                    matlab.ui.control.DropDown
        noz_percent                     matlab.ui.control.Label
        noz_thrt_unit                   matlab.ui.control.DropDown
        NozzleCd                        matlab.ui.control.NumericEditField
        NozzleDischargeCoefficientEditFieldLabel  matlab.ui.control.Label
        NozzleEfficiency                matlab.ui.control.NumericEditField
        NozzleEfficiencyEditFieldLabel  matlab.ui.control.Label
        NozzleExit                      matlab.ui.control.NumericEditField
        ThroatDiameter                  matlab.ui.control.NumericEditField
        NozzleThroatDiameterEditFieldLabel  matlab.ui.control.Label
        TankDimensionsPanel             matlab.ui.container.Panel
        tnk_V_unit                      matlab.ui.control.DropDown
        cmbr_V_unit                     matlab.ui.control.DropDown
        tnk_D_unit                      matlab.ui.control.DropDown
        tnk_L_unit                      matlab.ui.control.DropDown
        ChamberVolumeByDimensions       matlab.ui.control.CheckBox
        ChamberVolume                   matlab.ui.control.NumericEditField
        CombustionChamberVolumeEditFieldLabel  matlab.ui.control.Label
        TankVolume                      matlab.ui.control.NumericEditField
        OxidizerTankVolumeEditFieldLabel  matlab.ui.control.Label
        TankDiameter                    matlab.ui.control.NumericEditField
        OxidizerTankDiameterEditFieldLabel  matlab.ui.control.Label
        TankLength                      matlab.ui.control.NumericEditField
        OxidizerTankLengthEditFieldLabel  matlab.ui.control.Label
        TankVolumeByDimensions          matlab.ui.control.CheckBox
        MotorConfiguration              matlab.ui.control.EditField
        ConfigurationNameLabel          matlab.ui.control.Label
        ResultsTab                      matlab.ui.container.Tab
        Legend                          matlab.ui.control.EditField
        LegendEditFieldLabel            matlab.ui.control.Label
        YaxisUnits                      matlab.ui.control.DropDown
        YaxisUnitsDropDownLabel         matlab.ui.control.Label
        XaxisUnits                      matlab.ui.control.DropDown
        XaxisUnitsDropDownLabel         matlab.ui.control.Label
        ExportCSV                       matlab.ui.control.Button
        ExportRSE                       matlab.ui.control.Button
        SaveResults                     matlab.ui.control.Button
        SavePlot                        matlab.ui.control.Button
        ClearPlot                       matlab.ui.control.Button
        AddPlot                         matlab.ui.control.Button
        PerformanceSummary              matlab.ui.control.TextArea
        PerformanceSummaryTextAreaLabel  matlab.ui.control.Label
        Yaxis                           matlab.ui.control.DropDown
        YaxisDropDownLabel              matlab.ui.control.Label
        PlotTitle                       matlab.ui.control.EditField
        PlotTitleEditFieldLabel         matlab.ui.control.Label
        Xaxis                           matlab.ui.control.DropDown
        XaxisDropDownLabel              matlab.ui.control.Label
        UIAxes                          matlab.ui.control.UIAxes
        AboutTab                        matlab.ui.container.Tab
        Image2                          matlab.ui.control.Image
        Label                           matlab.ui.control.Label
        Image                           matlab.ui.control.Image
        ClickhereformoreinformationregardingHRAPitsusageandButton  matlab.ui.control.Button
        TextArea                        matlab.ui.control.TextArea
        AppTitle                        matlab.ui.control.Label
    end

    
    properties (Access = public)
        s = struct(); % input structure
        o = struct(); % output structure
        x = struct(); % state structure
        u = struct(); % selected units structure
        plt = struct(); % plotting structure
        appDir % user install location
        t % time variable
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            movegui(app.HRAP_v2022_06_04, 'center')
            [~, result] = system('path');
            app.appDir = convertCharsToStrings(char(regexpi(result, 'Path=(.*?);', 'tokens', 'once')));
        end

        % Button pushed function: LoadPropellantConfig
        function load_propellant(app, event)
            [file,path,~] = uigetfile('*.mat','Select a propellant file',sprintf('%s\\propellant_configs',app.appDir));

            if isequal(file,0)
            else
                load(fullfile(path, file)); %#ok<LOAD> 
                app.s.prop_file = convertCharsToStrings(fullfile(path,file));
                app.s.prop_nm = s.prop_nm; %#ok<*ADPROPLC> 
                app.s.prop_k = s.prop_k;
                app.s.prop_M = s.prop_M;
                app.s.prop_OF = s.prop_OF;
                app.s.prop_Pc = s.prop_Pc;
                app.s.prop_Reg = s.prop_Reg;
                app.s.prop_Rho = s.prop_Rho;
                app.s.prop_T = s.prop_T;
                app.s.opt_OF = s.opt_OF;
                
                app.PropellantName.Value = app.s.prop_nm;
                app.PropellantDensity.Value = app.s.prop_Rho;
                app.RegressionCoefficient.Value = app.s.prop_Reg(1);
                app.RegressionExponent.Value = app.s.prop_Reg(2);
                app.LengthExponent.Value = app.s.prop_Reg(3);
                app.ConstantOF.Value = app.s.opt_OF;
            end
        end

        % Value changed function: MassProperties
        function enable_mass_properties(app, event)
            value = app.MassProperties.Value;
            
            if value == 1
                app.TankLocation.Enable = 1;
                app.TankLocation.Editable = 1;
                app.OxidizerTankLocationEditFieldLabel.Enable = 1;
                app.tnk_X_unit.Enable = 1;
                app.tnk_X_unit.Editable = 1;
                app.GrainLocation.Enable = 1;
                app.GrainLocation.Editable = 1;
                app.FuelGrainLocationEditFieldLabel.Enable = 1;
                app.cmbr_X_unit.Enable = 1;
                app.cmbr_X_unit.Editable = 1;
                app.MotorCG.Enable = 1;
                app.MotorCG.Editable = 1;
                app.EmptyMotorCenterofMassEditFieldLabel.Enable = 1;
                app.mtr_cg_unit.Enable = 1;
                app.mtr_cg_unit.Editable = 1;
                app.MotorMass.Enable = 1;
                app.MotorMass.Editable = 1;
                app.EmptyMotorMassEditFieldLabel.Enable = 1;
                app.mtr_m_unit.Enable = 1;
                app.mtr_m_unit.Editable = 1;
                if app.TankVolumeByDimensions.Value == 0
                    app.TankDiameter.Enable = 1;
                    app.TankDiameter.Editable = 1;
                    app.OxidizerTankDiameterEditFieldLabel.Enable = 1;
                    app.tnk_D_unit.Enable = 1;
                    app.tnk_D_unit.Editable = 1;
                end
            else
                app.TankLocation.Enable = 0;
                app.TankLocation.Editable = 0;
                app.OxidizerTankLocationEditFieldLabel.Enable = 0;
                app.tnk_X_unit.Enable = 0;
                app.tnk_X_unit.Editable = 0;
                app.GrainLocation.Enable = 0;
                app.GrainLocation.Editable = 0;
                app.FuelGrainLocationEditFieldLabel.Enable = 0;
                app.cmbr_X_unit.Enable = 0;
                app.cmbr_X_unit.Editable = 0;
                app.MotorCG.Enable = 0;
                app.MotorCG.Editable = 0;
                app.EmptyMotorCenterofMassEditFieldLabel.Enable = 0;
                app.mtr_cg_unit.Enable = 0;
                app.mtr_cg_unit.Editable = 0;
                app.MotorMass.Enable = 0;
                app.MotorMass.Editable = 0;
                app.EmptyMotorMassEditFieldLabel.Enable = 0;
                app.mtr_m_unit.Enable = 0;
                app.mtr_m_unit.Editable = 0;
                if app.TankVolumeByDimensions.Value == 0
                    app.TankDiameter.Enable = 0;
                    app.TankDiameter.Editable = 0;
                    app.OxidizerTankDiameterEditFieldLabel.Enable = 0;
                    app.tnk_D_unit.Enable = 0;
                    app.tnk_D_unit.Editable = 0;
                end
            end
        end

        % Value changed function: TankVolumeByDimensions
        function tank_dimensions(app, event)
            value = app.TankVolumeByDimensions.Value;
            
            if value == 1
                app.TankLength.Enable = 1;
                app.TankLength.Editable = 1;
                app.OxidizerTankLengthEditFieldLabel.Enable = 1;
                app.tnk_L_unit.Enable = 1;
                app.tnk_L_unit.Editable = 1;
                if app.MassProperties.Value == 0
                    app.TankDiameter.Enable = 1;
                    app.TankDiameter.Editable = 1;
                    app.OxidizerTankDiameterEditFieldLabel.Enable = 1;
                    app.tnk_D_unit.Enable = 1;
                    app.tnk_D_unit.Editable = 1;
                end
                app.TankVolume.Enable = 0;
                app.TankVolume.Editable = 0;
                app.OxidizerTankVolumeEditFieldLabel.Enable = 0;
                app.tnk_V_unit.Enable = 0;
                app.tnk_V_unit.Editable = 0;
            else
                app.TankLength.Enable = 0;
                app.TankLength.Editable = 0;
                app.OxidizerTankLengthEditFieldLabel.Enable = 0;
                app.tnk_L_unit.Enable = 0;
                app.tnk_L_unit.Editable = 0;
                if app.MassProperties.Value == 0
                    app.TankDiameter.Enable = 0;
                    app.TankDiameter.Editable = 0;
                    app.OxidizerTankDiameterEditFieldLabel.Enable = 0;
                    app.tnk_D_unit.Enable = 0;
                    app.tnk_D_unit.Editable = 0;
                end
                app.TankVolume.Enable = 1;
                app.TankVolume.Editable = 1;
                app.OxidizerTankVolumeEditFieldLabel.Enable = 1;
                app.tnk_V_unit.Enable = 1;
                app.tnk_V_unit.Editable = 1;
            end
        end

        % Value changed function: ChamberVolumeByDimensions
        function combustion_chamber_volume(app, event)
            value = app.ChamberVolumeByDimensions.Value;
            
            if value == 0
                app.ChamberVolume.Enable = 1;
                app.ChamberVolume.Editable = 1;
                app.CombustionChamberVolumeEditFieldLabel.Enable = 1;
                app.cmbr_V_unit.Enable = 1;
                app.cmbr_V_unit.Editable = 1;
            else
                app.ChamberVolume.Enable = 0;
                app.ChamberVolume.Editable = 0;
                app.CombustionChamberVolumeEditFieldLabel.Enable = 0;
                app.cmbr_V_unit.Enable = 0;
                app.cmbr_V_unit.Editable = 0;
            end
        end

        % Value changed function: RegressionModel
        function reg_model(app, event)
            value = app.RegressionModel.Value;
            
            if value == "Shifting OF"
                app.RegressionCoefficient.Enable = 1;
                app.RegressionCoefficient.Editable = 1;
                app.RegressionCoefficientaEditFieldLabel.Enable = 1;
                app.a_unit.Enable = 1;
                app.RegressionExponent.Enable = 1;
                app.RegressionExponent.Editable = 1;
                app.RegressionExponentnEditFieldLabel.Enable = 1;
                app.n_unit.Enable = 1;
                app.LengthExponent.Enable = 1;
                app.LengthExponent.Editable = 1;
                app.LengthExponentmEditFieldLabel.Enable = 1;
                app.m_unit.Enable = 1;
                app.ConstantOF.Enable = 0;
                app.ConstantOF.Editable = 0;
                app.ConstantOFRatioLabel.Enable = 0;
            else
                app.RegressionCoefficient.Enable = 0;
                app.RegressionCoefficient.Editable = 0;
                app.RegressionCoefficientaEditFieldLabel.Enable = 0;
                app.a_unit.Enable = 0;
                app.RegressionExponent.Enable = 0;
                app.RegressionExponent.Editable = 0;
                app.RegressionExponentnEditFieldLabel.Enable = 0;
                app.n_unit.Enable = 0;
                app.LengthExponent.Enable = 0;
                app.LengthExponent.Editable = 0;
                app.LengthExponentmEditFieldLabel.Enable = 0;
                app.m_unit.Enable = 0;
                app.ConstantOF.Enable = 1;
                app.ConstantOF.Editable = 1;
                app.ConstantOFRatioLabel.Enable = 1;
            end
        end

        % Value changed function: NozzleDropDown
        function noz_definition(app, event)
            value = app.NozzleDropDown.Value;
            if value == "Nozzle Expansion Ratio"
                app.noz_ext_unit.Enable = 0;
                app.noz_ext_unit.Editable = 0;
                app.noz_ext_unit.Visible = 0;
            else
                app.noz_ext_unit.Enable = 1;
                app.noz_ext_unit.Editable = 1;
                app.noz_ext_unit.Visible = 1;
            end
        end

        % Value changed function: TankDropDown
        function tank_condition(app, event)
            value = app.TankDropDown.Value;
            if value == "Starting Tank Temperature"
                app.T_tnk_unit.Items = {'F','C','K','R'};
            else
                app.T_tnk_unit.Items = {'psi','psf','atm','Pa','kPa','Bar','MPa'};
            end
        end

        % Value changed function: OxidizerDropDown
        function tank_fill(app, event)
            value = app.OxidizerDropDown.Value;
            if value == "Starting Oxidizer Mass"
                app.m_O_unit.Items = {'lbm','kg','g','oz'};
            else
                app.m_O_unit.Items = {'%'};
            end
        end

        % Value changed function: VentState
        function vent_state(app, event)
            value = app.VentState.Value;
            if value == "None"
                app.VentCd.Enable = 0;
                app.VentCd.Editable = 0;
                app.VentDischargeCoefficientEditFieldLabel.Enable = 0;
                app.VentDiameter.Enable = 0;
                app.VentDiameter.Editable = 0;
                app.VentDiameterEditFieldLabel.Enable = 0;
                app.vnt_D_unit.Enable = 0;
                app.vnt_D_unit.Editable = 0;
            else
                app.VentCd.Enable = 1;
                app.VentCd.Editable = 1;
                app.VentDischargeCoefficientEditFieldLabel.Enable = 1;
                app.VentDiameter.Enable = 1;
                app.VentDiameter.Editable = 1;
                app.VentDiameterEditFieldLabel.Enable = 1;
                app.vnt_D_unit.Enable = 1;
                app.vnt_D_unit.Editable = 0;
            end
        end

        % Button pushed function: RunSimulation
        function run_sim(app, event)
            app.RunSimulation.BackgroundColor = [0.8,0.8,0.8];
            
            % Pull in units
            
            value = app.tnk_V_unit.Value;
            
            if value == "in^3"
                app.u.tnk_V = 0.0254^3;
            elseif value == "ft^3"
                app.u.tnk_V = 0.3048^3;
            elseif value == "cm^3"
                app.u.tnk_V = 0.01^3;
            elseif value == "L"
                app.u.tnk_V = 0.001;
            elseif value == "Gal"
                app.u.tnk_V = 0.00378541;
            elseif value == "m^3"
                app.u.tnk_V = 1;
            end

            value = app.tnk_L_unit.Value;
            
            if value == "in"
                app.u.tnk_L = 0.0254;
            elseif value == "ft"
                app.u.tnk_L = 0.3048;
            elseif value == "cm"
                app.u.tnk_L = 0.01;
            elseif value == "mm"
                app.u.tnk_L = 0.001;
            elseif value == "m"
                app.u.tnk_L = 1;
            end

            value = app.tnk_D_unit.Value;
            
            if value == "in"
                app.u.tnk_D = 0.0254;
            elseif value == "ft"
                app.u.tnk_D = 0.3048;
            elseif value == "cm"
                app.u.tnk_D = 0.01;
            elseif value == "mm"
                app.u.tnk_D = 0.001;
            elseif value == "m"
                app.u.tnk_D = 1;
            end

            value = app.cmbr_V_unit.Value;
            
            if value == "in^3"
                app.u.cmbr_V = 0.0254^3;
            elseif value == "ft^3"
                app.u.cmbr_V = 0.3048^3;
            elseif value == "cm^3"
                app.u.cmbr_V = 0.01^3;
            elseif value == "L"
                app.u.cmbr_V = 0.001;
            elseif value == "Gal"
                app.u.cmbr_V = 0.00378541;
            elseif value == "m^3"
                app.u.cmbr_V = 1;
            end

            value = app.noz_thrt_unit.Value;
            
            if value == "in"
                app.u.noz_thrt = 0.0254;
            elseif value == "ft"
                app.u.noz_thrt = 0.3048;
            elseif value == "cm"
                app.u.noz_thrt = 0.01;
            elseif value == "mm"
                app.u.noz_thrt = 0.001;
            elseif value == "m"
                app.u.noz_thrt = 1;
            end

            value = app.noz_ext_unit.Value;
            
            if value == "in"
                app.u.noz_ext = 0.0254;
            elseif value == "ft"
                app.u.noz_ext = 0.3048;
            elseif value == "cm"
                app.u.noz_ext = 0.01;
            elseif value == "mm"
                app.u.noz_ext = 0.001;
            elseif value == "m"
                app.u.noz_ext = 1;
            end

            value = app.tnk_X_unit.Value;
            
            if value == "in"
                app.u.tnk_X = 0.0254;
            elseif value == "ft"
                app.u.tnk_X = 0.3048;
            elseif value == "cm"
                app.u.tnk_X = 0.01;
            elseif value == "mm"
                app.u.tnk_X = 0.001;
            elseif value == "m"
                app.u.tnk_X = 1;
            end

            value = app.cmbr_X_unit.Value;
            
            if value == "in"
                app.u.cmbr_X = 0.0254;
            elseif value == "ft"
                app.u.cmbr_X = 0.3048;
            elseif value == "cm"
                app.u.cmbr_X = 0.01;
            elseif value == "mm"
                app.u.cmbr_X = 0.001;
            elseif value == "m"
                app.u.cmbr_X = 1;
            end

            value = app.mtr_cg_unit.Value;
            
            if value == "in"
                app.u.mtr_cg = 0.0254;
            elseif value == "ft"
                app.u.mtr_cg = 0.3048;
            elseif value == "cm"
                app.u.mtr_cg = 0.01;
            elseif value == "mm"
                app.u.mtr_cg = 0.001;
            elseif value == "m"
                app.u.mtr_cg = 1;
            end

            value = app.mtr_m_unit.Value;
            
            if value == "lbm"
                app.u.mtr_m = 0.453592;
            elseif value == "oz"
                app.u.mtr_m = 0.0283495;
            elseif value == "g"
                app.u.mtr_m = 0.001;
            elseif value == "kg"
                app.u.mtr_m = 1;
            end

            value = app.P_cmbr_unit.Value;
            
            if value == "psi"
                app.u.P_cmbr = 101325/14.696;
            elseif value == "psf"
                app.u.P_cmbr = 101325/14.696*144;
            elseif value == "atm"
                app.u.P_cmbr = 101325;
            elseif value == "MPa"
                app.u.P_cmbr = 1000000;
            elseif value == "kPa"
                app.u.P_cmbr = 1000;
            elseif value == "Bar"
                app.u.P_cmbr = 100000;
            elseif value == "Pa"
                app.u.P_cmbr = 1;
            end

            value = app.Pa_unit.Value;
            
            if value == "psi"
                app.u.Pa = 101325/14.696;
            elseif value == "psf"
                app.u.Pa = 101325/14.696*144;
            elseif value == "atm"
                app.u.Pa = 101325;
            elseif value == "MPa"
                app.u.Pa = 1000000;
            elseif value == "kPa"
                app.u.Pa = 1000;
            elseif value == "Bar"
                app.u.Pa = 100000;
            elseif value == "Pa"
                app.u.Pa = 1;
            end

            value = app.density_unit.Value;
            
            if value == "lb/in^3"
                app.u.prop_Rho = 1/(2.205*0.0254^3);
            elseif value == "lb/ft^3"
                app.u.prop_Rho = 1/(2.205*0.3048^3);
            elseif value == "g/cm^3"
                app.u.prop_Rho = 1000;
            elseif value == "kg/m^3"
                app.u.prop_Rho = 1;
            end

            value = app.inj_D_unit.Value;
            
            if value == "in"
                app.u.inj_D = 0.0254;
            elseif value == "ft"
                app.u.inj_D = 0.3048;
            elseif value == "cm"
                app.u.inj_D = 0.01;
            elseif value == "mm"
                app.u.inj_D = 0.001;
            elseif value == "m"
                app.u.inj_D = 1;
            end

            value = app.vnt_D_unit.Value;
            
            if value == "in"
                app.u.vnt_D = 0.0254;
            elseif value == "ft"
                app.u.vnt_D = 0.3048;
            elseif value == "cm"
                app.u.vnt_D = 0.01;
            elseif value == "mm"
                app.u.vnt_D = 0.001;
            elseif value == "m"
                app.u.vnt_D = 1;
            end

            value = app.grnID_unit.Value;
            
            if value == "in"
                app.u.grn_ID = 0.0254;
            elseif value == "ft"
                app.u.grn_ID = 0.3048;
            elseif value == "cm"
                app.u.grn_ID = 0.01;
            elseif value == "mm"
                app.u.grn_ID = 0.001;
            elseif value == "m"
                app.u.grn_ID = 1;
            end

            value = app.grnOD_unit.Value;
            
            if value == "in"
                app.u.grn_OD = 0.0254;
            elseif value == "ft"
                app.u.grn_OD = 0.3048;
            elseif value == "cm"
                app.u.grn_OD = 0.01;
            elseif value == "mm"
                app.u.grn_OD = 0.001;
            elseif value == "m"
                app.u.grn_OD = 1;
            end

            value = app.grnL_unit.Value;
            
            if value == "in"
                app.u.grn_L = 0.0254;
            elseif value == "ft"
                app.u.grn_L = 0.3048;
            elseif value == "cm"
                app.u.grn_L = 0.01;
            elseif value == "mm"
                app.u.grn_L = 0.001;
            elseif value == "kg"
                app.u.grn_L = 1;
            end

            value = app.m_O_unit.Value;
            
            if value == "lbm"
                app.u.m_o = 0.453592;
            elseif value == "oz"
                app.u.m_o = 0.0283495;
            elseif value == "g"
                app.u.m_o = 0.001;
            elseif value == "kg"
                app.u.m_o = 1;
            end
            
            % initialize variables for HRAP
            
            app.s.grn_OD = app.GrainOD.Value*app.u.grn_OD;
            app.s.grn_ID = app.GrainID.Value*app.u.grn_ID;
            app.s.grn_L = app.GrainLength.Value*app.u.grn_L;
            app.s.cstar_eff = app.CEfficiency.Value/100;
            if app.RegressionModel.Value == "Constant OF"
                app.s.const_OF = app.ConstantOF.Value;
            end
            app.s.mtr_nm = app.MotorConfiguration.Value;
            app.s.tmax = app.RunTime.Value;
            app.s.tburn = app.BurnTime.Value;
            app.s.dt = app.Timestep.Value;
            app.s.Pa = app.AmbientPressure.Value*app.u.Pa;
            if app.MassProperties.Value == 1 || app.TankVolumeByDimensions.Value == 1
                app.s.tnk_D = app.TankDiameter.Value*app.u.tnk_D;
            end
            if app.TankVolumeByDimensions.Value == 1
                app.s.tnk_V = app.TankLength.Value*app.u.tnk_L*0.25*pi*(app.TankDiameter.Value*app.u.tnk_D)^2;
            else
                app.s.tnk_V = app.TankVolume.Value*app.u.tnk_V;
            end
            if app.ChamberVolumeByDimensions.Value == 1
                app.s.cmbr_V = app.GrainLength.Value*app.u.grn_L*0.25*pi*(app.GrainOD.Value*app.u.grn_OD)^2;
            else
                app.s.cmbr_V = app.ChamberVolume.Value*app.u.cmbr_V;
            end
            if app.RegressionModel.Value == "Constant OF"
                app.s.regression_model = @(s,x) const_OF(s,x);
            elseif app.RegressionModel.Value == "Shifting OF"
                app.s.regression_model = @(s,x) shift_OF(s,x);
                app.s.prop_Reg(1) = app.RegressionCoefficient.Value;
                app.s.prop_Reg(2) = app.RegressionExponent.Value;
                app.s.prop_Reg(3) = app.LengthExponent.Value;
            end
            app.s.noz_Cd = app.NozzleCd.Value;
            app.s.noz_thrt = app.ThroatDiameter.Value*app.u.noz_thrt;
            if app.NozzleDropDown.Value == "Nozzle Expansion Ratio"
                app.s.noz_ER = app.NozzleExit.Value;
            elseif app.NozzleDropDown.Value == "Nozzle Exit Diameter"
                app.s.noz_ER = (app.NozzleExit.Value*app.u.noz_ext)^2/(app.s.noz_thrt)^2;
            end
            app.s.noz_eff = app.NozzleEfficiency.Value/100;
            app.s.inj_CdA = 0.25*pi*(app.InjectorDiameter.Value*app.u.inj_D)^2*app.InjectorCd.Value;
            app.s.inj_N = app.NumberofInjectors.Value;
            if app.VentState.Value == "None"
                app.s.vnt_S = 0;
            elseif app.VentState.Value == "External"
                app.s.vnt_S = 1;
                app.s.vnt_CdA = 0.25*pi*(app.VentDiameter.Value*app.u.vnt_D)^2*app.VentCd.Value;
            else
                app.s.vnt_S = 2;
                app.s.vnt_CdA = 0.25*pi*(app.VentDiameter.Value*app.u.vnt_D)^2*app.VentCd.Value;
            end
            if app.MassProperties.Value == 1
                app.s.mtr_m = app.MotorMass.Value*app.u.mtr_m;
                app.s.mtr_cg = app.MotorCG.Value*app.u.mtr_cg;
                app.s.tnk_X = app.TankLocation.Value*app.u.tnk_X;
                app.s.cmbr_X = app.GrainLocation.Value*app.u.cmbr_X;
                app.s.mp_calc = 1;
            else
                app.s.mp_calc = 0;
            end
            if app.TankDropDown.Value == "Starting Tank Temperature"
                value = app.T_tnk_unit.Value;
                if value == "C"
                    app.x.T_tnk = app.TankCond.Value+273.15;
                elseif value == "R"
                    app.x.T_tnk = app.TankCond.Value/1.8;
                elseif value == "F"
                    app.x.T_tnk = ((app.TankCond.Value-32)/1.8)+273.15;
                else
                    app.x.T_tnk = app.TankCond.Value;
                end
            elseif app.TankDropDown.Value == "Starting Tank Pressure"
                value = app.T_tnk_unit.Value;
                if value == "psi"
                    app.u.T_tnk = 101325/14.696;
                elseif value == "psf"
                    app.u.T_tnk = 101325/14.696*144;
                elseif value == "atm"
                    app.u.T_tnk = 101325;
                elseif value == "MPa"
                    app.u.T_tnk = 1000000;
                elseif value == "kPa"
                    app.u.T_tnk = 1000;
                elseif value == "Bar"
                    app.u.T_tnk = 100000;
                elseif value == "Pa"
                    app.u.T_tnk = 1;
                end
                a1 = -6.71893;
                a2 = 1.35966;
                a3 = -1.3779;
                a4 = -4.051;
                Pc = 7251000;
                Tc = 309.57;
                Pv = @(T) Pc*exp((1/(T/Tc))*(a1*(1-T/Tc) + a2*(1-(T/Tc))^(3/2) + a3*(1-(T/Tc))^(5/2) + a4*(1-(T/Tc))^5)) - app.TankCond.Value*app.u.T_tnk;
                app.x.T_tnk = fzero(Pv,273.15);
            end

            app.x.ox_props                  = NOX(app.x.T_tnk);
            
            if app.OxidizerDropDown.Value == "Tank Fill Percentage"
                app.x.m_o                   = (app.OxidizerFill.Value/100)*app.s.tnk_V*app.x.ox_props.rho_l + (1-app.OxidizerFill.Value/100)*app.s.tnk_V*app.x.ox_props.rho_v;
            else
                app.x.m_o                   = app.OxidizerFill.Value*app.u.m_o;
            end
            app.x.P_tnk                     = app.x.ox_props.Pv;
            app.x.P_cmbr                    = app.ChamberPressure.Value*app.u.P_cmbr;
            app.x.mdot_o                    = 0;
            app.x.mLiq_new                  = (app.s.tnk_V - (app.x.m_o/app.x.ox_props.rho_v))/((1/app.x.ox_props.rho_l)-(1/app.x.ox_props.rho_v));
            app.x.mLiq_old                  = app.x.mLiq_new + 1;
            app.x.m_f                       = 0.25*pi*(app.s.grn_OD^2 - app.s.grn_ID^2)*app.s.prop_Rho*app.s.grn_L;
            app.x.m_g                       = 1.225*(app.s.cmbr_V - 0.25*pi*(app.s.grn_OD^2 - app.s.grn_ID^2)*app.s.grn_L);
            if app.RegressionModel.Value == "Constant OF"
                app.x.OF                    = app.s.const_OF;
            elseif app.RegressionModel.Value == "Shifting OF"
                app.x.OF                        = 0;
            end
            app.x.mdot_f                    = 0;
            app.x.mdot_n                    = 0;
            app.x.rdot                      = 0;
            app.x.grn_ID                    = app.s.grn_ID;
            
            app.t                           = 0;
            
            app.o.t                         = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.m_o                       = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.P_tnk                     = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.P_cmbr                    = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.mdot_o                    = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.mdot_f                    = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.OF                        = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.grn_ID                    = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.mdot_n                    = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.rdot                      = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.m_f                       = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.F_thr                     = zeros(1,app.s.tmax/app.s.dt + 1);
            app.o.dP                        = zeros(1,app.s.tmax/app.s.dt + 1);
            
            app.o.m_o(1)                    = app.x.m_o;
            app.o.P_tnk(1)                  = app.x.P_tnk;
            app.o.P_cmbr(1)                 = app.x.P_cmbr;
            app.o.mdot_o(1)                 = app.x.mdot_o;
            app.o.mdot_f(1)                 = app.x.mdot_f;
            app.o.OF(1)                     = app.x.OF;
            app.o.grn_ID(1)                 = app.x.grn_ID;
            app.o.mdot_n(1)                 = app.x.mdot_n;
            app.o.rdot(1)                   = app.x.rdot;
            app.o.m_f(1)                    = app.x.m_f;
            
            if app.s.mp_calc == 1
                app.o.m_t                   = zeros(1,app.s.tmax/app.s.dt + 1);
                app.o.cg                    = zeros(1,app.s.tmax/app.s.dt + 1);
            
                mp                          = mass(app.s,app.x);
            
                app.o.m_t(1)                = mp(1);
                app.o.cg(1)                 = mp(2);
            end
            
            % RUN HRAP SIMULATION
            
            [app.s,app.x,app.o,app.t] = sim_loop(app.s,app.x,app.o,app.t);
            
            % Sim Results
            
            totalImpulse = trapz(app.o.t(app.o.F_thr>0),app.o.F_thr(app.o.F_thr>0));

            [motorClass,percent] = impulse(totalImpulse);
            
            %Display Motor Performance
            
            burnTime = app.o.t(sum(app.o.F_thr>0));
            peakF_thr = max(app.o.F_thr);
            averageF_thr = mean(app.o.F_thr(app.o.F_thr>0));
            peakPressure = max(app.o.P_cmbr)/10^5;
            averagePressure = mean(app.o.P_cmbr(app.o.P_cmbr>0))/10^5;
            fuelConsumed = (app.o.m_f(1)-app.o.m_f(end));
            oxidizerConsumed = (app.o.m_o(1)-app.o.m_o(end));
            averageOF = oxidizerConsumed/fuelConsumed;
            intPressure = trapz(app.o.t,app.o.P_cmbr);
            Cstar = intPressure*0.25*pi*app.s.noz_thrt^2/(fuelConsumed+oxidizerConsumed);
            specificImpulse = totalImpulse/(((oxidizerConsumed+fuelConsumed))*9.81);
            fuelLeft = app.x.grn_ID*100;
            
            app.PerformanceSummary.Value = sprintf('Motor Name: %s\n    Propellant: %s\n    Oxidizer Tank Volume: %0.0f cc\n    Burn Time: %0.3f s\n    Peak Thrust: %0.1f N\n    Average Thrust: %0.1f N\n    Total Impulse: %0.2f N-s\n    Peak Chamber Pressure: %0.3f bar\n    Average Chamber Pressure: %0.3f bar\n    Port Diameter at Burnout: %0.3f cm\n    Fuel Consumed: %0.3f kg\n    Oxidizer Consumed: %0.3f kg\n    Average OF Ratio: %0.3f\n    Characteristic Velocity: %0.1f m/s\n    Specific Impulse: %0.1f s\n    Motor Classification: %3.0f%% %s%0.0f', app.s.mtr_nm,app.s.prop_nm,app.s.tnk_V*10^6,burnTime,peakF_thr,averageF_thr,totalImpulse,peakPressure,averagePressure,fuelLeft,fuelConsumed,oxidizerConsumed,averageOF,Cstar,specificImpulse,percent,motorClass,averageF_thr);
            
            app.PlotTitle.Enable = 1;
            app.PlotTitle.Editable = 1;
            app.PlotTitleEditFieldLabel.Enable = 1;
            app.Xaxis.Enable = 1;
            app.Xaxis.Editable = 1;
            app.XaxisDropDownLabel.Enable = 1;
            app.Yaxis.Enable = 1;
            app.Yaxis.Editable = 1;
            app.YaxisDropDownLabel.Enable = 1;
            app.XaxisUnits.Enable = 1;
            app.XaxisUnits.Editable = 1;
            app.XaxisUnitsDropDownLabel.Enable = 1;
            app.YaxisUnits.Enable = 1;
            app.YaxisUnits.Editable = 1;
            app.YaxisUnitsDropDownLabel.Enable = 1;
            app.Legend.Enable = 1;
            app.Legend.Editable = 1;
            app.LegendEditFieldLabel.Enable = 1;
            app.AddPlot.Enable = 1;
            app.ClearPlot.Enable = 1;
            app.SavePlot.Enable = 1;
            app.SaveResults.Enable = 1;
            if app.MassProperties.Value == 1
                app.ExportRSE.Enable = 1;
            end
            app.ExportCSV.Enable = 1;
            
            app.RunSimulation.BackgroundColor = [0.96,0.96,0.96];
            
        end

        % Value changed function: Xaxis
        function x_axis(app, event)
            value = app.Xaxis.Value;
            
            if value == "Time"
                app.XaxisUnits.Items = {'s'};
            elseif value == "Oxidizer Mass"
                app.XaxisUnits.Items = {'lbm','kg','g','oz'};
            elseif value == "Fuel Mass"
                app.XaxisUnits.Items = {'lbm','kg','g','oz'};
            elseif value == "Total Motor Mass"
                app.XaxisUnits.Items = {'lbm','kg','g','oz'};
            elseif value == "Center of Mass"
                app.XaxisUnits.Items = {'in','ft','mm','cm','m'};
            elseif value == "Chamber Pressure"
                app.XaxisUnits.Items = {'psi','psf','atm','Pa','kPa','Bar','MPa'};
            elseif value == "Total Propellant Mass"
                app.XaxisUnits.Items = {'lbm','kg','g','oz'};
            elseif value == "Tank Pressure"
                app.XaxisUnits.Items = {'psi','psf','atm','Pa','kPa','Bar','MPa'};
            elseif value == "Oxidizer Mass Flow Rate"
                app.XaxisUnits.Items = {'lbm/s','kg/s','g/s','oz/s'};
            elseif value == "Fuel Mass Flow Rate"
                app.XaxisUnits.Items = {'lbm/s','kg/s','g/s','oz/s'};
            elseif value == "Total Mass Flow Rate"
                app.XaxisUnits.Items = {'lbm/s','kg/s','g/s','oz/s'};
            elseif value == "OF Ratio"
                app.XaxisUnits.Items = {''};
            elseif value == "Regression Rate"
                app.XaxisUnits.Items = {'in/s','ft/s','mm/s','cm/s','m/s'};
            elseif value == "Thrust"
                app.XaxisUnits.Items = {'N','lbf','kgf'};
            elseif value == "Injector Pressure Drop"
                app.XaxisUnits.Items = {'psi','psf','atm','Pa','kPa','Bar','MPa'};
            end
        end

        % Value changed function: Yaxis
        function y_axis(app, event)
            value = app.Yaxis.Value;
            
            if value == "Time"
                app.YaxisUnits.Items = {'s'};
                app.Legend.Value = 'Time';
            elseif value == "Oxidizer Mass"
                app.YaxisUnits.Items = {'lbm','kg','g','oz'};
                app.Legend.Value = 'Oxidizer Mass';
            elseif value == "Fuel Mass"
                app.YaxisUnits.Items = {'lbm','kg','g','oz'};
                app.Legend.Value = 'Fuel Mass';
            elseif value == "Total Motor Mass"
                app.YaxisUnits.Items = {'lbm','kg','g','oz'};
                app.Legend.Value = 'Total Motor Mass';
            elseif value == "Center of Mass"
                app.YaxisUnits.Items = {'in','ft','mm','cm','m'};
                app.Legend.Value = 'Center of Mass';
            elseif value == "Chamber Pressure"
                app.YaxisUnits.Items = {'psi','psf','atm','Pa','kPa','Bar','MPa'};
                app.Legend.Value = 'Chamber Pressure';
            elseif value == "Total Propellant Mass"
                app.YaxisUnits.Items = {'lbm','kg','g','oz'};
                app.Legend.Value = 'Total Propellant Mass';
            elseif value == "Tank Pressure"
                app.YaxisUnits.Items = {'psi','psf','atm','Pa','kPa','Bar','MPa'};
                app.Legend.Value = 'Tank Pressure';
            elseif value == "Oxidizer Mass Flow Rate"
                app.YaxisUnits.Items = {'lbm/s','kg/s','g/s','oz/s'};
                app.Legend.Value = 'Oxidizer Mass Flow Rate';
            elseif value == "Fuel Mass Flow Rate"
                app.YaxisUnits.Items = {'lbm/s','kg/s','g/s','oz/s'};
                app.Legend.Value = 'Fuel Mass Flow Rate';
            elseif value == "Total Mass Flow Rate"
                app.YaxisUnits.Items = {'lbm/s','kg/s','g/s','oz/s'};
                app.Legend.Value = 'Total Mass Flow Rate';
            elseif value == "OF Ratio"
                app.YaxisUnits.Items = {''};
                app.Legend.Value = 'OF Ratio';
            elseif value == "Regression Rate"
                app.YaxisUnits.Items = {'in/s','ft/s','mm/s','cm/s','m/s'};
                app.Legend.Value = 'Regression Rate';
            elseif value == "Thrust"
                app.YaxisUnits.Items = {'N','lbf','kgf'};
                app.Legend.Value = 'Thrust';
            elseif value == "Injector Pressure Drop"
                app.YaxisUnits.Items = {'psi','psf','atm','Pa','kPa','Bar','MPa'};
                app.Legend.Value = 'Injector Pressure Drop';
            end
        end

        % Button pushed function: AddPlot
        function plot(app, event)
            value = app.Xaxis.Value;
            
            if value == "Time"
                app.plt.x = app.o.t;
            elseif value == "Oxidizer Mass"
                app.plt.x = app.o.m_o;
            elseif value == "Fuel Mass"
                app.plt.x = app.o.m_f;
            elseif value == "Total Motor Mass"
                app.plt.x = app.o.m_f+app.o.m_o+app.o.m_t;
            elseif value == "Center of Mass"
                app.plt.x = app.o.cg;
            elseif value == "Chamber Pressure"
                app.plt.x = app.o.P_cmbr;
            elseif value == "Total Propellant Mass"
                app.plt.x = app.o.m_f+app.o.m_o;
            elseif value == "Tank Pressure"
                app.plt.x = app.o.P_tnk;
            elseif value == "Oxidizer Mass Flow Rate"
                app.plt.x = app.o.mdot_o;
            elseif value == "Fuel Mass Flow Rate"
                app.plt.x = app.o.mdot_f;
            elseif value == "Total Mass Flow Rate"
                app.plt.x = app.o.mdot_n;
            elseif value == "OF Ratio"
                app.plt.x = app.o.OF;
            elseif value == "Regression Rate"
                app.plt.x = app.o.rdot;
            elseif value == "Thrust"
                app.plt.x = app.o.F_thr;
            elseif value == "Injector Pressure Drop"
                app.plt.x = app.o.P_tnk - app.o.P_cmbr;
            end
            
            value = app.Yaxis.Value;
            
            if value == "Time"
                app.plt.y = app.o.t;
            elseif value == "Oxidizer Mass"
                app.plt.y = app.o.m_o;
            elseif value == "Fuel Mass"
                app.plt.y = app.o.m_f;
            elseif value == "Total Motor Mass"
                app.plt.y = app.o.m_f+app.o.m_o+app.o.m_t;
            elseif value == "Center of Mass"
                app.plt.y = app.o.cg;
            elseif value == "Chamber Pressure"
                app.plt.y = app.o.P_cmbr;
            elseif value == "Total Propellant Mass"
                app.plt.y = app.o.m_f+app.o.m_o;
            elseif value == "Tank Pressure"
                app.plt.y = app.o.P_tnk;
            elseif value == "Oxidizer Mass Flow Rate"
                app.plt.y = app.o.mdot_o;
            elseif value == "Fuel Mass Flow Rate"
                app.plt.y = app.o.mdot_f;
            elseif value == "Total Mass Flow Rate"
                app.plt.y = app.o.mdot_n;
            elseif value == "OF Ratio"
                app.plt.y = app.o.OF;
            elseif value == "Regression Rate"
                app.plt.y = app.o.rdot;
            elseif value == "Thrust"
                app.plt.y = app.o.F_thr;
            elseif value == "Injector Pressure Drop"
                app.plt.y = app.o.P_tnk - app.o.P_cmbr;
            end
            
            value = app.XaxisUnits.Value;
            
            if value == "lbm"
            elseif value == "kg"
                app.u.x_axis = 1;
            elseif value == "g"
                app.u.x_axis = 1/0.001;
            elseif value == "oz"
                app.u.x_axis = 1/0.0283495;
            elseif value == "in"
                app.u.x_axis = 1/0.0254;
            elseif value == "ft"
                app.u.x_axis = 1/0.3048;
            elseif value == "mm"
                app.u.x_axis = 1/0.001;
            elseif value == "cm"
                app.u.x_axis = 1/0.01';
            elseif value == "m"
                app.u.x_axis = 1;
            elseif value == "psi"
                app.u.x_axis = 1/6894.76;
            elseif value == "psf"
                app.u.x_axis = 1/147.8802777777778;
            elseif value == "atm"
                app.u.x_axis = 1/101325;
            elseif value == "Pa"
                app.u.x_axis = 1;
            elseif value == "kPa"
                app.u.x_axis = 1/1000;
            elseif value == "Bar"
                app.u.x_axis = 1/100000;
            elseif value == "MPa"
                app.u.x_axis = 1/1000000;
            elseif value == "lbm/s"
                app.u.x_axis = 1/0.453592;
            elseif value == "kg/s"
                app.u.x_axis = 1;
            elseif value == "g/s"
                app.u.x_axis = 1/0.001;
            elseif value == "oz/s"
                app.u.x_axis = 1/0.0283495;
            elseif value == "in/s"
                app.u.x_axis = 1/0.0254;
            elseif value == "ft/s"
                app.u.x_axis = 1/0.3048;
            elseif value == "mm/s"
                app.u.x_axis = 1/0.001;
            elseif value == "cm/s"
                app.u.x_axis = 1/0.01;
            elseif value == "m/s"
                app.u.x_axis = 1;
            elseif value == "N"
                app.u.x_axis = 1;
            elseif value == "lbf"
                app.u.x_axis = 1/4.44822;
            elseif value == "kgf"
                app.u.x_axis = 1/9.80665;
            else
                app.u.x_axis = 1;
            end
            
            value = app.YaxisUnits.Value;
            
            if value == "lbm"
            elseif value == "kg"
                app.u.y_axis = 1;
            elseif value == "g"
                app.u.y_axis = 1000;
            elseif value == "oz"
                app.u.y_axis = 35.28;
            elseif value == "in"
                app.u.y_axis = 1/0.0254;
            elseif value == "ft"
                app.u.y_axis = 1/0.3048;
            elseif value == "mm"
                app.u.y_axis = 1000;
            elseif value == "cm"
                app.u.y_axis = 100;
            elseif value == "m"
                app.u.y_axis = 1;
            elseif value == "psi"
                app.u.y_axis = 1/6894.76;
            elseif value == "psf"
                app.u.y_axis = 1/147.8802777777778;
            elseif value == "atm"
                app.u.y_axis = 1/101325;
            elseif value == "Pa"
                app.u.y_axis = 1;
            elseif value == "kPa"
                app.u.y_axis = 0.001;
            elseif value == "Bar"
                app.u.y_axis = 0.00001;
            elseif value == "MPa"
                app.u.y_axis = 0.000001;
            elseif value == "lbm/s"
                app.u.y_axis = 1/0.453592;
            elseif value == "kg/s"
                app.u.y_axis = 1;
            elseif value == "g/s"
                app.u.y_axis = 1/0.001;
            elseif value == "oz/s"
                app.u.y_axis = 1/0.0283495;
            elseif value == "in/s"
                app.u.y_axis = 1/0.0254;
            elseif value == "ft/s"
                app.u.y_axis = 1/0.3048;
            elseif value == "mm/s"
                app.u.y_axis = 1/0.001;
            elseif value == "cm/s"
                app.u.y_axis = 1/0.01;
            elseif value == "m/s"
                app.u.y_axis = 1;
            elseif value == "N"
                app.u.y_axis = 1;
            elseif value == "lbf"
                app.u.y_axis = 1/4.44822;
            elseif value == "kgf"
                app.u.y_axis = 1/9.80665;
            else
                app.u.y_axis = 1;
            end
            
            plot(app.UIAxes,(app.plt.x.*app.u.x_axis),(app.plt.y.*app.u.y_axis),'DisplayName',app.Legend.Value) %#ok<ADMTHDINV> 
            app.UIAxes.Title.String = app.PlotTitle.Value;
   
            xvalue = app.Xaxis.Value;
            if xvalue == "Time"
                app.plt.xlabel = append('Time (',app.XaxisUnits.Value,')');
            elseif xvalue == "Oxidizer Mass"
                app.plt.xlabel = append('Mass (',app.XaxisUnits.Value,')');
            elseif xvalue == "Fuel Mass"
                app.plt.xlabel = append('Mass (',app.XaxisUnits.Value,')');
            elseif xvalue == "Total Mass"
                app.plt.xlabel = append('Mass (',app.XaxisUnits.Value,')');
            elseif xvalue == "Center of Mass"
                app.plt.xlabel = append('Center of Mass (',app.XaxisUnits.Value,')');
            elseif xvalue == "Chamber Pressure"
                app.plt.xlabel = append('Pressure (',app.XaxisUnits.Value,')');
            elseif xvalue == "Total Propellant Mass"
                app.plt.xlabel = append('Mass (',app.XaxisUnits.Value,')');
            elseif xvalue == "Tank Pressure"
                app.plt.xlabel = append('Pressure (',app.XaxisUnits.Value,')');
            elseif xvalue == "Oxidizer Mass Flow Rate"
                app.plt.xlabel = append('Mass Flow Rate (',app.XaxisUnits.Value,')');
            elseif xvalue == "Fuel Mass Flow Rate"
                app.plt.xlabel = append('Mass Flow Rate (',app.XaxisUnits.Value,')');
            elseif xvalue == "Total Mass Flow Rate"
                app.plt.xlabel = append('Mass Flow Rate (',app.XaxisUnits.Value,')');
            elseif xvalue == "OF Ratio"
                app.plt.xlabel = 'OF Ratio';
            elseif xvalue == "Regression Rate"
                app.plt.xlabel = append('Regression Rate (',app.XaxisUnits.Value,')');
            elseif xvalue == "Thrust"
                app.plt.xlabel = append('Thrust (',app.XaxisUnits.Value,')');
            elseif xvalue == "Injector Pressure Drop"
                app.plt.xlabel = append('Pressure (',app.XaxisUnits.Value,')');
            end
            
            yvalue = app.Yaxis.Value;
            if yvalue == "Time"
                app.plt.ylabel = append('Time (',app.YaxisUnits.Value,')');
            elseif yvalue == "Oxidizer Mass"
                app.plt.ylabel = append('Mass (',app.YaxisUnits.Value,')');
            elseif yvalue == "Fuel Mass"
                app.plt.ylabel = append('Mass (',app.YaxisUnits.Value,')');
            elseif yvalue == "Total Mass"
                app.plt.ylabel = append('Mass (',app.YaxisUnits.Value,')');
            elseif yvalue == "Center of Mass"
                app.plt.ylabel = append('Center of Mass (',app.YaxisUnits.Value,')');
            elseif yvalue == "Chamber Pressure"
                app.plt.ylabel = append('Pressure (',app.YaxisUnits.Value,')');
            elseif yvalue == "Total Propellant Mass"
                app.plt.ylabel = append('Mass (',app.YaxisUnits.Value,')');
            elseif yvalue == "Tank Pressure"
                app.plt.ylabel = append('Pressure (',app.YaxisUnits.Value,')');
            elseif yvalue == "Oxidizer Mass Flow Rate"
                app.plt.ylabel = append('Mass Flow Rate (',app.YaxisUnits.Value,')');
            elseif yvalue == "Fuel Mass Flow Rate"
                app.plt.ylabel = append('Mass Flow Rate (',app.YaxisUnits.Value,')');
            elseif yvalue == "Total Mass Flow Rate"
                app.plt.ylabel = append('Mass Flow Rate (',app.YaxisUnits.Value,')');
            elseif yvalue == "OF Ratio"
                app.plt.ylabel = 'OF Ratio';
            elseif yvalue == "Regression Rate"
                app.plt.ylabel = append('Regression Rate (',app.YaxisUnits.Value,')');
            elseif yvalue == "Thrust"
                app.plt.ylabel = append('Thrust (',app.YaxisUnits.Value,')');
            elseif yvalue == "Injector Pressure Drop"
                app.plt.ylabel = append('Pressure (',app.YaxisUnits.Value,')');
            end
            
            app.UIAxes.XLabel.String = app.plt.xlabel;
            app.UIAxes.YLabel.String = app.plt.ylabel;
            hold(app.UIAxes,"on")
            legend(app.UIAxes)
        end

        % Button pushed function: ClearPlot
        function clear_plot(app, event)
            reset(app.UIAxes);
            cla(app.UIAxes);
        end

        % Button pushed function: SavePlot
        function save_plot(app, event)
            [file,path] = uiputfile('*.png','Save Plot',sprintf('%s\\output\\%s.png',app.appDir,app.PlotTitle.Value));

            if isequal(file,0)
            else
                filepath = convertCharsToStrings(fullfile(path,file));
                exportgraphics(app.UIAxes,filepath);
            end
        end

        % Button pushed function: SaveResults
        function save_results(app, event)
            totalImpulse = trapz(app.o.t(app.o.F_thr>0),app.o.F_thr(app.o.F_thr>0));
            [motorClass,percent] = impulse(totalImpulse);
            burnTime = app.o.t(sum(app.o.F_thr>0));
            peakF_thr = max(app.o.F_thr);
            averageF_thr = mean(app.o.F_thr(app.o.F_thr>0));
            peakPressure = max(app.o.P_cmbr)/10^5;
            averagePressure = mean(app.o.P_cmbr(app.o.P_cmbr>0))/10^5;
            fuelConsumed = (app.o.m_f(1)-app.o.m_f(end));
            oxidizerConsumed = (app.o.m_o(1)-app.o.m_o(end));
            averageOF = oxidizerConsumed/fuelConsumed;
            intPressure = trapz(app.o.t,app.o.P_cmbr);
            Cstar = intPressure*0.25*pi*app.s.noz_thrt^2/(fuelConsumed+oxidizerConsumed);
            specificImpulse = totalImpulse/(((oxidizerConsumed+fuelConsumed))*9.81);
            fuelLeft = app.x.grn_ID*100;

            [file,path] = uiputfile('*.txt','Save Results Summary',sprintf('%s\\output\\sim_results.txt',app.appDir));

            if isequal(file,0)
            else
                fileID = fopen(convertCharsToStrings(fullfile(path,file)),'w');
                fprintf(fileID,'Motor Name: %s\n    Propellant: %s\n    Oxidizer Tank Volume: %0.0f cc\n    Burn Time: %0.3f s\n    Peak Thrust: %0.1f N\n    Average Thrust: %0.1f N\n    Total Impulse: %0.2f N-s\n    Peak Chamber Pressure: %0.3f bar\n    Average Chamber Pressure: %0.3f bar\n    Port Diameter at Burnout: %0.3f cm\n    Fuel Consumed: %0.3f kg\n    Oxidizer Consumed: %0.3f kg\n    Average OF Ratio: %0.3f\n    Characteristic Velocity: %0.1f m/s\n    Specific Impulse: %0.1f s\n    Motor Classification: %3.0f%% %s%0.0f', app.s.mtr_nm,app.s.prop_nm,app.s.tnk_V*10^6,burnTime,peakF_thr,averageF_thr,totalImpulse,peakPressure,averagePressure,fuelLeft,fuelConsumed,oxidizerConsumed,averageOF,Cstar,specificImpulse,percent,motorClass,averageF_thr);
                fclose('all');
            end
        end

        % Button pushed function: ExportRSE
        function save_rse(app, event)
            if app.MassProperties.Value == 1
                %Input template
                cg = app.o.cg*1000; %array of cg (millimeters from forward end of motor) as function of time
                f = app.o.F_thr; %array of force (newtons) as function of time
                m = app.o.m_t*1000; %array of mass (grams) as function of time
                t = app.o.t; %array of time (seconds) for other variables
                Itot = trapz(app.o.t(app.o.F_thr>0),app.o.F_thr(app.o.F_thr>0)); %total impulse, newton-seconds
                propWt = ((app.o.m_f(1)-app.o.m_f(end)) + (app.o.m_o(1)-app.o.m_o(end)))*1000; %propellant weight, grams
                Isp = Itot/((propWt*1000)*9.81); %specific impulse, seconds
                Type = 'Hybrid';
                avgThrust = mean(app.o.F_thr); %average thrust, newtons
                burn_time = max(app.o.t); %burn time, seconds
                motorClass = impulse(Itot);
                prompt = {'Enter Motor Diameter in mm:','Enter Motor Length in mm:','Enter Motor Manufacturer:','Input Motor Notes:'};
                dlgtitle = 'RSE Export';
                dims = [1 12; 1 12; 1 50; 5 50];
                definput = {'0','0','Research',''};
                answer = inputdlg(prompt,dlgtitle,dims,definput);
                code = sprintf('%s%0.0f',motorClass,avgThrust);
                dia = str2double(answer{1}); %motor diameter, millimeters
                exitDia = (app.s.noz_thrt*app.s.noz_ER^0.5)*1000; %nozzle exit diameter, millimeters
                initWt = app.o.m_t(1)*1000; %loaded motor weight before ignition, grams
                len = str2double(answer{2}); %motor length, millimeters
                massFrac = (app.o.m_t(1)-app.o.m_t(length(app.o.m_t)))/app.o.m_t(1); %mass fraction, ratio of propellant mass to total mass
                mfg = answer{3};
                peakThrust = max(app.o.F_thr); %peak thrust, newtons
                throatDia = app.s.noz_thrt*1000;
                comments = answer{4};

                [file,path] = uiputfile('*.rse','Save RSE File',sprintf('%s\\output\\%s.rse',app.appDir,code));

                if isequal(file,0)
                else
                    fileID = fopen(convertCharsToStrings(fullfile(path,file)),'w');
                    fprintf(fileID,'<engine-database>\n	<engine-list>\n');
                    fprintf(fileID,'	<engine FDiv="10" FFix="1" FStep="-1." Isp="%f" Itot="%f" Type="%s" auto-calc-cg="0" auto-calc-mass="0" avgThrust="%f" burn-time="%f" cgDiv="10" cgFix="1" cgStep="-1." code="%s" delays="0" dia="%f" exitDia="%f" initWt="%f" len="%f" mDiv="10" mFix="1" mStep="-1." massFrac="%f" mfg="%s" peakThrust="%f" propWt="%f" tDiv="10" tFix="1" tStep="-1." throatDia="%f">',Isp, Itot, Type, avgThrust, burn_time, code, dia, exitDia, initWt, len, massFrac, mfg, peakThrust, propWt, throatDia);
                    fprintf(fileID,'\n	<comments> \n');
                    fprintf(fileID,'		%s\n',comments);
                    fprintf(fileID,'	</comments>\n');
                    fprintf(fileID,'	<data>\n');
            
                    for i = 1:length(cg)
                        fprintf(fileID,'		<eng-data cg="%f" f="%f" m="%f" t="%f"/>\n',cg(i),f(i),m(i),t(i));
                    end
            
                    fprintf(fileID,'	</data>\n');
                    fprintf(fileID,'	</engine>\n');
                    fprintf(fileID,' 	</engine-list>\n');
                    fprintf(fileID,'</engine-database>\n');
                    fclose('all');
                end
            end
        end

        % Button pushed function: ExportCSV
        function save_csv(app, event)
            [file,path] = uiputfile('*.csv','Save CSV Output',sprintf('%s\\output\\HRAP_output.csv',app.appDir));

            if isequal(file,0)
            else
                filename = convertCharsToStrings(fullfile(path,file));
                if app.MassProperties.Value == 1
                    M = ["Time (s)","Thrust (N)","Oxidizer Tank Pressure (kPa)","Combustion Chamber Pressure (kPa)","Injector Pressure Drop (kPa)","Oxidizer Mass (kg)","Fuel Mass (kg)","Total Motor Mass (kg)","Oxidizer Mass Flow Rate (kg/s)","Fuel Mass Flow Rate (kg/s)","Exhaust Mass Flow Rate (kg/s)","Oxidizer to Fuel Ratio","Grain ID (cm)","Regression Rate (mm/s)","Motor Center of Mass (cm)";
                        app.o.t', app.o.F_thr', app.o.P_tnk'./1000, app.o.P_cmbr'./1000, app.o.dP'./1000, app.o.m_o', app.o.m_f', app.o.m_t', app.o.mdot_o', app.o.mdot_f', app.o.mdot_n', app.o.OF', app.o.grn_ID'.*100, app.o.rdot', app.o.cg'.*100];
                else
                    M = ["Time (s)","Thrust (N)","Oxidizer Tank Pressure (kPa)","Combustion Chamber Pressure (kPa)","Injector Pressure Drop (kPa)","Oxidizer Mass (kg)","Fuel Mass (kg)","Oxidizer Mass Flow Rate (kg/s)","Fuel Mass Flow Rate (kg/s)","Exhaust Mass Flow Rate (kg/s)","Oxidizer to Fuel Ratio","Grain ID (cm)","Regression Rate (mm/s)";
                        app.o.t', app.o.F_thr', app.o.P_tnk'./1000, app.o.P_cmbr'./1000, app.o.dP'./1000, app.o.m_o', app.o.m_f', app.o.mdot_o', app.o.mdot_f', app.o.mdot_n', app.o.OF', app.o.grn_ID'.*100, app.o.rdot'];
                end
                
                writematrix(M,filename);
                fclose('all');
            end
            
        end

        % Button pushed function: SaveMotor
        function save_config(app, event)

            cfg.mtr_nm = app.MotorConfiguration.Value;
            
            % Tank Dimensions Tab
            cfg.tnk_V = app.TankVolume.Value;
            cfg.tnk_V_unit = app.tnk_V_unit.Value;
            cfg.tnk_V_state = app.TankVolumeByDimensions.Value;
            cfg.tnk_L = app.TankLength.Value;
            cfg.tnk_L_unit = app.tnk_L_unit.Value;
            cfg.tnk_D = app.TankDiameter.Value;
            cfg.tnk_D_unit = app.tnk_D_unit.Value;
            cfg.cmbr_V_state = app.ChamberVolumeByDimensions.Value;
            cfg.cmbr_V = app.ChamberVolume.Value;
            cfg.cmbr_V_unit = app.cmbr_V_unit.Value;
            
            % Nozzle Configuration Tab
            cfg.noz_thrt = app.ThroatDiameter.Value;
            cfg.noz_thrt_unit = app.noz_thrt_unit.Value;
            cfg.noz_def = app.NozzleDropDown.Value;
            cfg.noz_ex = app.NozzleExit.Value;
            cfg.noz_ex_unit = app.noz_ext_unit.Value;
            cfg.noz_eff = app.NozzleEfficiency.Value;
            cfg.noz_Cd = app.NozzleCd.Value;
            
            % Mass Properties
            cfg.mp_state = app.MassProperties.Value;
            cfg.tnk_X = app.TankLocation.Value;
            cfg.tnk_X_unit = app.tnk_X_unit.Value;
            cfg.cmbr_X = app.GrainLocation.Value;
            cfg.cmbr_X_unit = app.cmbr_X_unit.Value;
            cfg.mtr_cg = app.MotorCG.Value;
            cfg.mtr_cg_unit = app.mtr_cg_unit.Value;
            cfg.mtr_m = app.MotorMass.Value;
            cfg.mtr_m_unit = app.mtr_m_unit.Value;
            
            % Initial Conditions
            cfg.tnk_dd = app.TankDropDown.Value;
            cfg.tnk_cond = app.TankCond.Value;
            if app.TankDropDown.Value == "Starting Tank Temperature"
                cfg.T_tnk_unit = app.T_tnk_unit.Value;
            elseif app.TankDropDown.Value == "Starting Tank Pressure"
                cfg.T_tnk_unit = app.T_tnk_unit.Value;
            end
            cfg.P_cmbr = app.ChamberPressure.Value;
            cfg.P_cmbr_unit = app.P_cmbr_unit.Value;
            cfg.fill_dd = app.OxidizerDropDown.Value;
            cfg.fill = app.OxidizerFill.Value;
            cfg.fill_unit = app.m_O_unit.Value;
            cfg.Pa = app.AmbientPressure.Value;
            cfg.Pa_unit = app.Pa_unit.Value;
            
            % Propellant Configuration
            cfg.prop_file = app.s.prop_file;
            cfg.prop_nm = app.PropellantName.Value;
            cfg.prop_rho = app.PropellantDensity.Value;
            cfg.prop_rho_unit = app.density_unit.Value;
            cfg.prop_a = app.RegressionCoefficient.Value;
            cfg.prop_n = app.RegressionExponent.Value;
            cfg.prop_m = app.LengthExponent.Value;
            cfg.const_OF = app.ConstantOF.Value;
            cfg.cstar_eff = app.CEfficiency.Value;
            cfg.grn_ID = app.GrainID.Value;
            cfg.grn_ID_unit = app.grnID_unit.Value;
            cfg.grn_OD = app.GrainOD.Value;
            cfg.grn_OD_unit = app.grnOD_unit.Value;
            cfg.grn_L = app.GrainLength.Value;
            cfg.grn_L_unit = app.grnL_unit.Value;
            
            % Injector Configuration
            cfg.inj_D = app.InjectorDiameter.Value;
            cfg.inj_D_unit = app.inj_D_unit.Value;
            cfg.inj_N = app.NumberofInjectors.Value;
            cfg.inj_Cd = app.InjectorCd.Value;
            cfg.vnt_state = app.VentState.Value;
            cfg.vnt_D = app.VentDiameter.Value;
            cfg.vnt_D_unit = app.vnt_D_unit.Value;
            cfg.vnt_Cd = app.VentCd.Value;
            
            % Simulation Configuration
            cfg.t_max = app.RunTime.Value;
            cfg.t_burn = app.BurnTime.Value;
            cfg.dt = app.Timestep.Value;
            cfg.reg_model = app.RegressionModel.Value;
            
            [file,path] = uiputfile('*.mat','Save Motor Configuration',sprintf('%s\\motor_configs\\%s.mat',app.appDir,app.MotorConfiguration.Value));

            if isequal(file,0)
            else
                filename = convertCharsToStrings(fullfile(path,file));
                save(filename,'cfg');
                fclose('all');
            end
        end

        % Button pushed function: 
        % ClickhereformoreinformationregardingHRAPitsusageandButton
        function open_hrap_document(app, event)
            winopen('Theory and Application of the Hybrid Rocket Analysis Program (HRAP).pdf');
        end

        % Button pushed function: LoadMotor
        function load_config(app, event)
            [file,path,~] = uigetfile('*.mat','Select a configuration file',sprintf('%s\\motor_configs',app.appDir));
            
            if isequal(file,0)
            else
                load(fullfile(path, file)); %#ok<LOAD> 
                
                app.MotorConfiguration.Value = cfg.mtr_nm; %#ok<*NODEF> 
                
                % Tank Dimensions Tab
                app.TankVolumeByDimensions.Value = cfg.tnk_V_state;
                app.TankLength.Value = cfg.tnk_L;
                app.tnk_L_unit.Value = cfg.tnk_L_unit;
                app.TankDiameter.Value = cfg.tnk_D;
                app.tnk_D_unit.Value = cfg.tnk_D_unit;
                app.TankVolumeByDimensions.Value = cfg.tnk_V_state;
                app.TankVolume.Value = cfg.tnk_V;
                app.tnk_V_unit.Value = cfg.tnk_V_unit;
                app.ChamberVolumeByDimensions.Value = cfg.cmbr_V_state;
                app.ChamberVolume.Value = cfg.cmbr_V;
                app.cmbr_V_unit.Value = cfg.cmbr_V_unit;
                
                % Nozzle Configuration Tab
                app.ThroatDiameter.Value = cfg.noz_thrt;
                app.noz_thrt_unit.Value = cfg.noz_thrt_unit;
                app.NozzleDropDown.Value = cfg.noz_def;
                app.NozzleExit.Value = cfg.noz_ex;
                app.noz_ext_unit.Value = cfg.noz_ex_unit;
                app.NozzleEfficiency.Value = cfg.noz_eff;
                app.NozzleCd.Value = cfg.noz_Cd;
                
                % Mass Properties
                app.MassProperties.Value = cfg.mp_state;
                app.TankLocation.Value = cfg.tnk_X;
                app.tnk_X_unit.Value = cfg.tnk_X_unit;
                app.GrainLocation.Value = cfg.cmbr_X;
                app.cmbr_X_unit.Value = cfg.cmbr_X_unit;
                app.MotorCG.Value = cfg.mtr_cg;
                app.mtr_cg_unit.Value = cfg.mtr_cg_unit;
                app.MotorMass.Value = cfg.mtr_m;
                app.mtr_m_unit.Value = cfg.mtr_m_unit;
                
                % Initial Conditions
                app.TankDropDown.Value = cfg.tnk_dd;
                app.TankCond.Value = cfg.tnk_cond;
                if cfg.tnk_dd == "Starting Tank Temperature"
                    app.T_tnk_unit.Items = {'F','C','K','R'};
                    app.T_tnk_unit.Value = cfg.T_tnk_unit;
                else
                    app.T_tnk_unit.Items = {'psi','psf','atm','Pa','kPa','Bar','MPa'};
                    app.T_tnk_unit.Value = cfg.T_tnk_unit;
                end
                app.ChamberPressure.Value = cfg.P_cmbr;
                app.P_cmbr_unit.Value = cfg.P_cmbr_unit;
                app.OxidizerDropDown.Value = cfg.fill_dd;
                app.OxidizerFill.Value = cfg.fill;
                app.m_O_unit.Value = cfg.fill_unit;
                if app.OxidizerDropDown.Value == "Starting Oxidizer Mass"
                    app.m_O_unit.Items = {'lbm','kg','g','oz'};
                else
                    app.m_O_unit.Items = {'%'};
                end
                app.AmbientPressure.Value = cfg.Pa;
                app.Pa_unit.Value = cfg.Pa_unit;
                
                % Propellant Configuration
                load(cfg.prop_file);
                app.s.prop_file = cfg.prop_file;
                app.s.prop_nm = s.prop_nm; %#ok<*ADPROPLC> 
                app.s.prop_k = s.prop_k;
                app.s.prop_M = s.prop_M;
                app.s.prop_OF = s.prop_OF;
                app.s.prop_Pc = s.prop_Pc;
                app.s.prop_Reg = s.prop_Reg;
                app.s.prop_Rho = s.prop_Rho;
                app.s.prop_T = s.prop_T;
                app.s.opt_OF = s.opt_OF;
                
                app.PropellantName.Value = cfg.prop_nm;
                app.PropellantDensity.Value = cfg.prop_rho;
                app.density_unit.Value = cfg.prop_rho_unit;
                app.RegressionCoefficient.Value = cfg.prop_a;
                app.RegressionExponent.Value = cfg.prop_n;
                app.LengthExponent.Value = cfg.prop_m;
                app.ConstantOF.Value = cfg.const_OF;
                app.CEfficiency.Value = cfg.cstar_eff;
                app.GrainID.Value = cfg.grn_ID;
                app.grnID_unit.Value = cfg.grn_ID_unit;
                app.GrainOD.Value = cfg.grn_OD;
                app.grnOD_unit.Value = cfg.grn_OD_unit;
                app.GrainLength.Value = cfg.grn_L;
                app.grnL_unit.Value = cfg.grn_L_unit;
                
                % Injector Configuration
                app.InjectorDiameter.Value = cfg.inj_D;
                app.inj_D_unit.Value = cfg.inj_D_unit;
                app.NumberofInjectors.Value = cfg.inj_N;
                app.InjectorCd.Value = cfg.inj_Cd;
                app.VentState.Value = cfg.vnt_state;
                app.VentDiameter.Value = cfg.vnt_D;
                app.vnt_D_unit.Value = cfg.vnt_D_unit;
                app.VentCd.Value = cfg.vnt_Cd;
                
                % Simulation Configuration
                app.RunTime.Value = cfg.t_max;
                app.BurnTime.Value = cfg.t_burn;
                app.Timestep.Value = cfg.dt;
                app.RegressionModel.Value = cfg.reg_model;
            end
            
            value = app.MassProperties.Value;
            
            if value == 1
                app.TankLocation.Enable = 1;
                app.TankLocation.Editable = 1;
                app.OxidizerTankLocationEditFieldLabel.Enable = 1;
                app.tnk_X_unit.Enable = 1;
                app.tnk_X_unit.Editable = 1;
                app.GrainLocation.Enable = 1;
                app.GrainLocation.Editable = 1;
                app.FuelGrainLocationEditFieldLabel.Enable = 1;
                app.cmbr_X_unit.Enable = 1;
                app.cmbr_X_unit.Editable = 1;
                app.MotorCG.Enable = 1;
                app.MotorCG.Editable = 1;
                app.EmptyMotorCenterofMassEditFieldLabel.Enable = 1;
                app.mtr_cg_unit.Enable = 1;
                app.mtr_cg_unit.Editable = 1;
                app.MotorMass.Enable = 1;
                app.MotorMass.Editable = 1;
                app.EmptyMotorMassEditFieldLabel.Enable = 1;
                app.mtr_m_unit.Enable = 1;
                app.mtr_m_unit.Editable = 1;
                if app.TankVolumeByDimensions.Value == 0
                    app.TankDiameter.Enable = 1;
                    app.TankDiameter.Editable = 1;
                    app.OxidizerTankDiameterEditFieldLabel.Enable = 1;
                    app.tnk_D_unit.Enable = 1;
                    app.tnk_D_unit.Editable = 1;
                end
            else
                app.TankLocation.Enable = 0;
                app.TankLocation.Editable = 0;
                app.OxidizerTankLocationEditFieldLabel.Enable = 0;
                app.tnk_X_unit.Enable = 0;
                app.tnk_X_unit.Editable = 0;
                app.GrainLocation.Enable = 0;
                app.GrainLocation.Editable = 0;
                app.FuelGrainLocationEditFieldLabel.Enable = 0;
                app.cmbr_X_unit.Enable = 0;
                app.cmbr_X_unit.Editable = 0;
                app.MotorCG.Enable = 0;
                app.MotorCG.Editable = 0;
                app.EmptyMotorCenterofMassEditFieldLabel.Enable = 0;
                app.mtr_cg_unit.Enable = 0;
                app.mtr_cg_unit.Editable = 0;
                app.MotorMass.Enable = 0;
                app.MotorMass.Editable = 0;
                app.EmptyMotorMassEditFieldLabel.Enable = 0;
                app.mtr_m_unit.Enable = 0;
                app.mtr_m_unit.Editable = 0;
                if app.TankVolumeByDimensions.Value == 0
                    app.TankDiameter.Enable = 0;
                    app.TankDiameter.Editable = 0;
                    app.OxidizerTankDiameterEditFieldLabel.Enable = 0;
                    app.tnk_D_unit.Enable = 0;
                    app.tnk_D_unit.Editable = 0;
                end
            end
            
            value = app.TankVolumeByDimensions.Value;
            
            if value == 1
                app.TankLength.Enable = 1;
                app.TankLength.Editable = 1;
                app.OxidizerTankLengthEditFieldLabel.Enable = 1;
                app.tnk_L_unit.Enable = 1;
                app.tnk_L_unit.Editable = 1;
                if app.MassProperties.Value == 0
                    app.TankDiameter.Enable = 1;
                    app.TankDiameter.Editable = 1;
                    app.OxidizerTankDiameterEditFieldLabel.Enable = 1;
                    app.tnk_D_unit.Enable = 1;
                    app.tnk_D_unit.Editable = 1;
                end
                app.TankVolume.Enable = 0;
                app.TankVolume.Editable = 0;
                app.OxidizerTankVolumeEditFieldLabel.Enable = 0;
                app.tnk_V_unit.Enable = 0;
                app.tnk_V_unit.Editable = 0;
            else
                app.TankLength.Enable = 0;
                app.TankLength.Editable = 0;
                app.OxidizerTankLengthEditFieldLabel.Enable = 0;
                app.tnk_L_unit.Enable = 0;
                app.tnk_L_unit.Editable = 0;
                if app.MassProperties.Value == 0
                    app.TankDiameter.Enable = 0;
                    app.TankDiameter.Editable = 0;
                    app.OxidizerTankDiameterEditFieldLabel.Enable = 0;
                    app.tnk_D_unit.Enable = 0;
                    app.tnk_D_unit.Editable = 0;
                end
                app.TankVolume.Enable = 1;
                app.TankVolume.Editable = 1;
                app.OxidizerTankVolumeEditFieldLabel.Enable = 1;
                app.tnk_V_unit.Enable = 1;
                app.tnk_V_unit.Editable = 1;
            end
            
            value = app.ChamberVolumeByDimensions.Value;
            
            if value == 0
                app.ChamberVolume.Enable = 1;
                app.ChamberVolume.Editable = 1;
                app.CombustionChamberVolumeEditFieldLabel.Enable = 1;
                app.cmbr_V_unit.Enable = 1;
                app.cmbr_V_unit.Editable = 1;
            else
                app.ChamberVolume.Enable = 0;
                app.ChamberVolume.Editable = 0;
                app.CombustionChamberVolumeEditFieldLabel.Enable = 0;
                app.cmbr_V_unit.Enable = 0;
                app.cmbr_V_unit.Editable = 0;
            end
            
            value = app.RegressionModel.Value;
            
            if value == "Shifting OF"
                app.RegressionCoefficient.Enable = 1;
                app.RegressionCoefficient.Editable = 1;
                app.RegressionCoefficientaEditFieldLabel.Enable = 1;
                app.a_unit.Enable = 1;
                app.RegressionExponent.Enable = 1;
                app.RegressionExponent.Editable = 1;
                app.RegressionExponentnEditFieldLabel.Enable = 1;
                app.n_unit.Enable = 1;
                app.LengthExponent.Enable = 1;
                app.LengthExponent.Editable = 1;
                app.LengthExponentmEditFieldLabel.Enable = 1;
                app.m_unit.Enable = 1;
                app.ConstantOF.Enable = 0;
                app.ConstantOF.Editable = 0;
                app.ConstantOFRatioLabel.Enable = 0;
            else
                app.RegressionCoefficient.Enable = 0;
                app.RegressionCoefficient.Editable = 0;
                app.RegressionCoefficientaEditFieldLabel.Enable = 0;
                app.a_unit.Enable = 0;
                app.RegressionExponent.Enable = 0;
                app.RegressionExponent.Editable = 0;
                app.RegressionExponentnEditFieldLabel.Enable = 0;
                app.n_unit.Enable = 0;
                app.LengthExponent.Enable = 0;
                app.LengthExponent.Editable = 0;
                app.LengthExponentmEditFieldLabel.Enable = 0;
                app.m_unit.Enable = 0;
                app.ConstantOF.Enable = 1;
                app.ConstantOF.Editable = 1;
                app.ConstantOFRatioLabel.Enable = 1;
            end
            
            value = app.NozzleDropDown.Value;
            if value == "Nozzle Expansion Ratio"
                app.noz_ext_unit.Enable = 0;
                app.noz_ext_unit.Editable = 0;
                app.noz_ext_unit.Visible = 0;
            else
                app.noz_ext_unit.Enable = 1;
                app.noz_ext_unit.Editable = 1;
                app.noz_ext_unit.Visible = 1;
            end
            
            value = app.VentState.Value;
            if value == "None"
                app.VentCd.Enable = 0;
                app.VentCd.Editable = 0;
                app.VentDischargeCoefficientEditFieldLabel.Enable = 0;
                app.VentDiameter.Enable = 0;
                app.VentDiameter.Editable = 0;
                app.VentDiameterEditFieldLabel.Enable = 0;
            else
                app.VentCd.Enable = 1;
                app.VentCd.Editable = 1;
                app.VentDischargeCoefficientEditFieldLabel.Enable = 1;
                app.VentDiameter.Enable = 1;
                app.VentDiameter.Editable = 1;
                app.VentDiameterEditFieldLabel.Enable = 1;
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create HRAP_v2022_06_04 and hide until all components are created
            app.HRAP_v2022_06_04 = uifigure('Visible', 'off');
            app.HRAP_v2022_06_04.Position = [100 100 850 889];
            app.HRAP_v2022_06_04.Name = 'MATLAB App';
            app.HRAP_v2022_06_04.Resize = 'off';

            % Create AppTitle
            app.AppTitle = uilabel(app.HRAP_v2022_06_04);
            app.AppTitle.HorizontalAlignment = 'center';
            app.AppTitle.FontName = 'Times New Roman';
            app.AppTitle.FontSize = 24;
            app.AppTitle.FontWeight = 'bold';
            app.AppTitle.Position = [250 850 353 40];
            app.AppTitle.Text = 'Hybrid Rocket Analysis Program';

            % Create AppTabs
            app.AppTabs = uitabgroup(app.HRAP_v2022_06_04);
            app.AppTabs.Position = [11 20 830 830];

            % Create MotorTab
            app.MotorTab = uitab(app.AppTabs);
            app.MotorTab.Title = 'Simulation Configuration';

            % Create ConfigurationNameLabel
            app.ConfigurationNameLabel = uilabel(app.MotorTab);
            app.ConfigurationNameLabel.HorizontalAlignment = 'right';
            app.ConfigurationNameLabel.Position = [12 773 112 22];
            app.ConfigurationNameLabel.Text = 'Configuration Name';

            % Create MotorConfiguration
            app.MotorConfiguration = uieditfield(app.MotorTab, 'text');
            app.MotorConfiguration.Position = [143 773 268 22];
            app.MotorConfiguration.Value = 'mtr_cfg';

            % Create TankDimensionsPanel
            app.TankDimensionsPanel = uipanel(app.MotorTab);
            app.TankDimensionsPanel.Title = 'Tank Dimensions';
            app.TankDimensionsPanel.Position = [11 566 400 200];

            % Create TankVolumeByDimensions
            app.TankVolumeByDimensions = uicheckbox(app.TankDimensionsPanel);
            app.TankVolumeByDimensions.ValueChangedFcn = createCallbackFcn(app, @tank_dimensions, true);
            app.TankVolumeByDimensions.Text = 'Calculate Oxidizer Tank Volume By Dimensions';
            app.TankVolumeByDimensions.Position = [11 122 278 22];

            % Create OxidizerTankLengthEditFieldLabel
            app.OxidizerTankLengthEditFieldLabel = uilabel(app.TankDimensionsPanel);
            app.OxidizerTankLengthEditFieldLabel.HorizontalAlignment = 'right';
            app.OxidizerTankLengthEditFieldLabel.Enable = 'off';
            app.OxidizerTankLengthEditFieldLabel.Position = [1 92 118 22];
            app.OxidizerTankLengthEditFieldLabel.Text = 'Oxidizer Tank Length';

            % Create TankLength
            app.TankLength = uieditfield(app.TankDimensionsPanel, 'numeric');
            app.TankLength.Limits = [0 Inf];
            app.TankLength.Editable = 'off';
            app.TankLength.Enable = 'off';
            app.TankLength.Position = [230 92 100 22];

            % Create OxidizerTankDiameterEditFieldLabel
            app.OxidizerTankDiameterEditFieldLabel = uilabel(app.TankDimensionsPanel);
            app.OxidizerTankDiameterEditFieldLabel.HorizontalAlignment = 'right';
            app.OxidizerTankDiameterEditFieldLabel.Enable = 'off';
            app.OxidizerTankDiameterEditFieldLabel.Position = [1 62 130 22];
            app.OxidizerTankDiameterEditFieldLabel.Text = 'Oxidizer Tank Diameter';

            % Create TankDiameter
            app.TankDiameter = uieditfield(app.TankDimensionsPanel, 'numeric');
            app.TankDiameter.Limits = [0 Inf];
            app.TankDiameter.Editable = 'off';
            app.TankDiameter.Enable = 'off';
            app.TankDiameter.Position = [230 62 100 22];

            % Create OxidizerTankVolumeEditFieldLabel
            app.OxidizerTankVolumeEditFieldLabel = uilabel(app.TankDimensionsPanel);
            app.OxidizerTankVolumeEditFieldLabel.HorizontalAlignment = 'right';
            app.OxidizerTankVolumeEditFieldLabel.Position = [1 152 122 22];
            app.OxidizerTankVolumeEditFieldLabel.Text = 'Oxidizer Tank Volume';

            % Create TankVolume
            app.TankVolume = uieditfield(app.TankDimensionsPanel, 'numeric');
            app.TankVolume.Limits = [0 Inf];
            app.TankVolume.Position = [230 152 100 22];

            % Create CombustionChamberVolumeEditFieldLabel
            app.CombustionChamberVolumeEditFieldLabel = uilabel(app.TankDimensionsPanel);
            app.CombustionChamberVolumeEditFieldLabel.HorizontalAlignment = 'right';
            app.CombustionChamberVolumeEditFieldLabel.Enable = 'off';
            app.CombustionChamberVolumeEditFieldLabel.Position = [1 32 166 22];
            app.CombustionChamberVolumeEditFieldLabel.Text = 'Combustion Chamber Volume';

            % Create ChamberVolume
            app.ChamberVolume = uieditfield(app.TankDimensionsPanel, 'numeric');
            app.ChamberVolume.Limits = [0 Inf];
            app.ChamberVolume.Editable = 'off';
            app.ChamberVolume.Enable = 'off';
            app.ChamberVolume.Position = [230 32 100 22];

            % Create ChamberVolumeByDimensions
            app.ChamberVolumeByDimensions = uicheckbox(app.TankDimensionsPanel);
            app.ChamberVolumeByDimensions.ValueChangedFcn = createCallbackFcn(app, @combustion_chamber_volume, true);
            app.ChamberVolumeByDimensions.Text = 'Calculate Combustion Chamber Volume By Grain Dimensions';
            app.ChamberVolumeByDimensions.Position = [11 6 356 22];
            app.ChamberVolumeByDimensions.Value = true;

            % Create tnk_L_unit
            app.tnk_L_unit = uidropdown(app.TankDimensionsPanel);
            app.tnk_L_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.tnk_L_unit.Enable = 'off';
            app.tnk_L_unit.Position = [329 92 62 22];
            app.tnk_L_unit.Value = 'cm';

            % Create tnk_D_unit
            app.tnk_D_unit = uidropdown(app.TankDimensionsPanel);
            app.tnk_D_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.tnk_D_unit.Enable = 'off';
            app.tnk_D_unit.Position = [329 62 62 22];
            app.tnk_D_unit.Value = 'cm';

            % Create cmbr_V_unit
            app.cmbr_V_unit = uidropdown(app.TankDimensionsPanel);
            app.cmbr_V_unit.Items = {'in^3', 'ft^3', 'cm^3', 'L', 'Gal', 'm^3'};
            app.cmbr_V_unit.Enable = 'off';
            app.cmbr_V_unit.Position = [329 32 62 22];
            app.cmbr_V_unit.Value = 'cm^3';

            % Create tnk_V_unit
            app.tnk_V_unit = uidropdown(app.TankDimensionsPanel);
            app.tnk_V_unit.Items = {'in^3', 'ft^3', 'cm^3', 'L', 'Gal', 'm^3'};
            app.tnk_V_unit.Editable = 'on';
            app.tnk_V_unit.BackgroundColor = [1 1 1];
            app.tnk_V_unit.Position = [329 152 62 22];
            app.tnk_V_unit.Value = 'cm^3';

            % Create NozzleConfigurationPanel
            app.NozzleConfigurationPanel = uipanel(app.MotorTab);
            app.NozzleConfigurationPanel.Title = 'Nozzle Configuration';
            app.NozzleConfigurationPanel.Position = [11 406 400 150];

            % Create NozzleThroatDiameterEditFieldLabel
            app.NozzleThroatDiameterEditFieldLabel = uilabel(app.NozzleConfigurationPanel);
            app.NozzleThroatDiameterEditFieldLabel.HorizontalAlignment = 'right';
            app.NozzleThroatDiameterEditFieldLabel.Position = [1 99 132 22];
            app.NozzleThroatDiameterEditFieldLabel.Text = 'Nozzle Throat Diameter';

            % Create ThroatDiameter
            app.ThroatDiameter = uieditfield(app.NozzleConfigurationPanel, 'numeric');
            app.ThroatDiameter.Limits = [0 Inf];
            app.ThroatDiameter.Position = [230 99 100 22];

            % Create NozzleExit
            app.NozzleExit = uieditfield(app.NozzleConfigurationPanel, 'numeric');
            app.NozzleExit.Limits = [0 Inf];
            app.NozzleExit.Position = [230 69 100 22];
            app.NozzleExit.Value = 1;

            % Create NozzleEfficiencyEditFieldLabel
            app.NozzleEfficiencyEditFieldLabel = uilabel(app.NozzleConfigurationPanel);
            app.NozzleEfficiencyEditFieldLabel.HorizontalAlignment = 'right';
            app.NozzleEfficiencyEditFieldLabel.Position = [1 39 97 22];
            app.NozzleEfficiencyEditFieldLabel.Text = 'Nozzle Efficiency';

            % Create NozzleEfficiency
            app.NozzleEfficiency = uieditfield(app.NozzleConfigurationPanel, 'numeric');
            app.NozzleEfficiency.Limits = [0 100];
            app.NozzleEfficiency.ValueDisplayFormat = '%11.3g';
            app.NozzleEfficiency.Position = [230 39 100 22];
            app.NozzleEfficiency.Value = 100;

            % Create NozzleDischargeCoefficientEditFieldLabel
            app.NozzleDischargeCoefficientEditFieldLabel = uilabel(app.NozzleConfigurationPanel);
            app.NozzleDischargeCoefficientEditFieldLabel.HorizontalAlignment = 'right';
            app.NozzleDischargeCoefficientEditFieldLabel.Position = [1 9 159 22];
            app.NozzleDischargeCoefficientEditFieldLabel.Text = 'Nozzle Discharge Coefficient';

            % Create NozzleCd
            app.NozzleCd = uieditfield(app.NozzleConfigurationPanel, 'numeric');
            app.NozzleCd.Limits = [0 1];
            app.NozzleCd.Position = [230 9 100 22];
            app.NozzleCd.Value = 1;

            % Create noz_thrt_unit
            app.noz_thrt_unit = uidropdown(app.NozzleConfigurationPanel);
            app.noz_thrt_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.noz_thrt_unit.Editable = 'on';
            app.noz_thrt_unit.BackgroundColor = [1 1 1];
            app.noz_thrt_unit.Position = [329 99 60 22];
            app.noz_thrt_unit.Value = 'cm';

            % Create noz_percent
            app.noz_percent = uilabel(app.NozzleConfigurationPanel);
            app.noz_percent.HorizontalAlignment = 'center';
            app.noz_percent.Position = [329 39 25 22];
            app.noz_percent.Text = '%';

            % Create noz_ext_unit
            app.noz_ext_unit = uidropdown(app.NozzleConfigurationPanel);
            app.noz_ext_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.noz_ext_unit.Editable = 'on';
            app.noz_ext_unit.Visible = 'off';
            app.noz_ext_unit.BackgroundColor = [1 1 1];
            app.noz_ext_unit.Position = [329 69 60 22];
            app.noz_ext_unit.Value = 'in';

            % Create NozzleDropDown
            app.NozzleDropDown = uidropdown(app.NozzleConfigurationPanel);
            app.NozzleDropDown.Items = {'Nozzle Expansion Ratio', 'Nozzle Exit Diameter'};
            app.NozzleDropDown.ValueChangedFcn = createCallbackFcn(app, @noz_definition, true);
            app.NozzleDropDown.Position = [5 69 171 22];
            app.NozzleDropDown.Value = 'Nozzle Expansion Ratio';

            % Create PropellantConfigurationPanel
            app.PropellantConfigurationPanel = uipanel(app.MotorTab);
            app.PropellantConfigurationPanel.Title = 'Propellant Configuration';
            app.PropellantConfigurationPanel.Position = [421 435 400 360];

            % Create PropellantNameEditFieldLabel
            app.PropellantNameEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.PropellantNameEditFieldLabel.HorizontalAlignment = 'right';
            app.PropellantNameEditFieldLabel.Position = [1 278 95 22];
            app.PropellantNameEditFieldLabel.Text = 'Propellant Name';

            % Create PropellantName
            app.PropellantName = uieditfield(app.PropellantConfigurationPanel, 'text');
            app.PropellantName.Editable = 'off';
            app.PropellantName.Position = [215 278 106 22];

            % Create PropellantDensityEditFieldLabel
            app.PropellantDensityEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.PropellantDensityEditFieldLabel.HorizontalAlignment = 'right';
            app.PropellantDensityEditFieldLabel.Position = [1 248 103 22];
            app.PropellantDensityEditFieldLabel.Text = 'Propellant Density';

            % Create PropellantDensity
            app.PropellantDensity = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.PropellantDensity.Limits = [0 Inf];
            app.PropellantDensity.Position = [215 248 106 22];
            app.PropellantDensity.Value = 1000;

            % Create RegressionCoefficientaEditFieldLabel
            app.RegressionCoefficientaEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.RegressionCoefficientaEditFieldLabel.HorizontalAlignment = 'right';
            app.RegressionCoefficientaEditFieldLabel.Enable = 'off';
            app.RegressionCoefficientaEditFieldLabel.Position = [1 218 144 22];
            app.RegressionCoefficientaEditFieldLabel.Text = 'Regression Coefficient (a)';

            % Create RegressionCoefficient
            app.RegressionCoefficient = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.RegressionCoefficient.Limits = [0 Inf];
            app.RegressionCoefficient.Editable = 'off';
            app.RegressionCoefficient.Enable = 'off';
            app.RegressionCoefficient.Position = [215 218 106 22];

            % Create RegressionExponentnEditFieldLabel
            app.RegressionExponentnEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.RegressionExponentnEditFieldLabel.HorizontalAlignment = 'right';
            app.RegressionExponentnEditFieldLabel.Enable = 'off';
            app.RegressionExponentnEditFieldLabel.Position = [2 189 138 22];
            app.RegressionExponentnEditFieldLabel.Text = 'Regression Exponent (n)';

            % Create RegressionExponent
            app.RegressionExponent = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.RegressionExponent.Editable = 'off';
            app.RegressionExponent.Enable = 'off';
            app.RegressionExponent.Position = [215 189 106 22];

            % Create LengthExponentmEditFieldLabel
            app.LengthExponentmEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.LengthExponentmEditFieldLabel.HorizontalAlignment = 'right';
            app.LengthExponentmEditFieldLabel.Enable = 'off';
            app.LengthExponentmEditFieldLabel.Position = [2 159 118 22];
            app.LengthExponentmEditFieldLabel.Text = 'Length Exponent (m)';

            % Create LengthExponent
            app.LengthExponent = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.LengthExponent.Editable = 'off';
            app.LengthExponent.Enable = 'off';
            app.LengthExponent.Position = [215 159 106 22];

            % Create CEfficiencyEditFieldLabel
            app.CEfficiencyEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.CEfficiencyEditFieldLabel.HorizontalAlignment = 'right';
            app.CEfficiencyEditFieldLabel.Position = [4 98 73 22];
            app.CEfficiencyEditFieldLabel.Text = 'C* Efficiency';

            % Create CEfficiency
            app.CEfficiency = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.CEfficiency.Limits = [0 100];
            app.CEfficiency.Position = [215 98 106 22];
            app.CEfficiency.Value = 100;

            % Create GrainIDEditFieldLabel
            app.GrainIDEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.GrainIDEditFieldLabel.HorizontalAlignment = 'right';
            app.GrainIDEditFieldLabel.Position = [4 68 50 22];
            app.GrainIDEditFieldLabel.Text = 'Grain ID';

            % Create GrainID
            app.GrainID = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.GrainID.Limits = [0 Inf];
            app.GrainID.Position = [215 68 106 22];

            % Create GrainODEditFieldLabel
            app.GrainODEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.GrainODEditFieldLabel.HorizontalAlignment = 'right';
            app.GrainODEditFieldLabel.Position = [4 38 56 22];
            app.GrainODEditFieldLabel.Text = 'Grain OD';

            % Create GrainOD
            app.GrainOD = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.GrainOD.Limits = [0 Inf];
            app.GrainOD.Position = [215 38 106 22];

            % Create GrainLengthEditFieldLabel
            app.GrainLengthEditFieldLabel = uilabel(app.PropellantConfigurationPanel);
            app.GrainLengthEditFieldLabel.HorizontalAlignment = 'right';
            app.GrainLengthEditFieldLabel.Position = [4 8 75 22];
            app.GrainLengthEditFieldLabel.Text = 'Grain Length';

            % Create GrainLength
            app.GrainLength = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.GrainLength.Limits = [0 Inf];
            app.GrainLength.Position = [215 8 106 22];

            % Create LoadPropellantConfig
            app.LoadPropellantConfig = uibutton(app.PropellantConfigurationPanel, 'push');
            app.LoadPropellantConfig.ButtonPushedFcn = createCallbackFcn(app, @load_propellant, true);
            app.LoadPropellantConfig.Position = [108 312 174 22];
            app.LoadPropellantConfig.Text = 'Load Propellant Configuration';

            % Create ConstantOFRatioLabel
            app.ConstantOFRatioLabel = uilabel(app.PropellantConfigurationPanel);
            app.ConstantOFRatioLabel.HorizontalAlignment = 'right';
            app.ConstantOFRatioLabel.Position = [4 128 105 22];
            app.ConstantOFRatioLabel.Text = 'Constant OF Ratio';

            % Create ConstantOF
            app.ConstantOF = uieditfield(app.PropellantConfigurationPanel, 'numeric');
            app.ConstantOF.Limits = [0 Inf];
            app.ConstantOF.Position = [215 128 106 22];

            % Create cstar_percent
            app.cstar_percent = uilabel(app.PropellantConfigurationPanel);
            app.cstar_percent.HorizontalAlignment = 'center';
            app.cstar_percent.Position = [320 98 25 22];
            app.cstar_percent.Text = '%';

            % Create density_unit
            app.density_unit = uidropdown(app.PropellantConfigurationPanel);
            app.density_unit.Items = {'lb/in^3', 'lb/ft^3', 'g/cm^3', 'kg/m^3'};
            app.density_unit.Editable = 'on';
            app.density_unit.BackgroundColor = [1 1 1];
            app.density_unit.Position = [320 248 75 22];
            app.density_unit.Value = 'kg/m^3';

            % Create grnID_unit
            app.grnID_unit = uidropdown(app.PropellantConfigurationPanel);
            app.grnID_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.grnID_unit.Editable = 'on';
            app.grnID_unit.BackgroundColor = [1 1 1];
            app.grnID_unit.Position = [319 68 75 22];
            app.grnID_unit.Value = 'cm';

            % Create grnOD_unit
            app.grnOD_unit = uidropdown(app.PropellantConfigurationPanel);
            app.grnOD_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.grnOD_unit.Editable = 'on';
            app.grnOD_unit.BackgroundColor = [1 1 1];
            app.grnOD_unit.Position = [319 38 75 22];
            app.grnOD_unit.Value = 'cm';

            % Create grnL_unit
            app.grnL_unit = uidropdown(app.PropellantConfigurationPanel);
            app.grnL_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.grnL_unit.Editable = 'on';
            app.grnL_unit.BackgroundColor = [1 1 1];
            app.grnL_unit.Position = [319 8 75 22];
            app.grnL_unit.Value = 'cm';

            % Create a_unit
            app.a_unit = uilabel(app.PropellantConfigurationPanel);
            app.a_unit.Enable = 'off';
            app.a_unit.Position = [320 218 38 22];
            app.a_unit.Text = ' mm/s';

            % Create n_unit
            app.n_unit = uilabel(app.PropellantConfigurationPanel);
            app.n_unit.Enable = 'off';
            app.n_unit.Position = [320 189 57 22];
            app.n_unit.Text = ' kg/m^2-s';

            % Create m_unit
            app.m_unit = uilabel(app.PropellantConfigurationPanel);
            app.m_unit.Enable = 'off';
            app.m_unit.Position = [320 159 25 22];
            app.m_unit.Text = ' m';

            % Create InjectorConfigurationPanel
            app.InjectorConfigurationPanel = uipanel(app.MotorTab);
            app.InjectorConfigurationPanel.Title = 'Injector Configuration';
            app.InjectorConfigurationPanel.Position = [421 216 398 210];

            % Create InjectorDiameterEditFieldLabel
            app.InjectorDiameterEditFieldLabel = uilabel(app.InjectorConfigurationPanel);
            app.InjectorDiameterEditFieldLabel.HorizontalAlignment = 'right';
            app.InjectorDiameterEditFieldLabel.Position = [1 158 97 22];
            app.InjectorDiameterEditFieldLabel.Text = 'Injector Diameter';

            % Create InjectorDiameter
            app.InjectorDiameter = uieditfield(app.InjectorConfigurationPanel, 'numeric');
            app.InjectorDiameter.Limits = [0 Inf];
            app.InjectorDiameter.Position = [215 158 105 22];

            % Create InjectorDischargeCoefficientEditFieldLabel
            app.InjectorDischargeCoefficientEditFieldLabel = uilabel(app.InjectorConfigurationPanel);
            app.InjectorDischargeCoefficientEditFieldLabel.HorizontalAlignment = 'right';
            app.InjectorDischargeCoefficientEditFieldLabel.Position = [1 128 162 22];
            app.InjectorDischargeCoefficientEditFieldLabel.Text = 'Injector Discharge Coefficient';

            % Create InjectorCd
            app.InjectorCd = uieditfield(app.InjectorConfigurationPanel, 'numeric');
            app.InjectorCd.Limits = [0 1];
            app.InjectorCd.Position = [215 128 105 22];
            app.InjectorCd.Value = 1;

            % Create NumberofInjectorsEditFieldLabel
            app.NumberofInjectorsEditFieldLabel = uilabel(app.InjectorConfigurationPanel);
            app.NumberofInjectorsEditFieldLabel.HorizontalAlignment = 'right';
            app.NumberofInjectorsEditFieldLabel.Position = [1 98 110 22];
            app.NumberofInjectorsEditFieldLabel.Text = 'Number of Injectors';

            % Create NumberofInjectors
            app.NumberofInjectors = uieditfield(app.InjectorConfigurationPanel, 'numeric');
            app.NumberofInjectors.Limits = [1 Inf];
            app.NumberofInjectors.RoundFractionalValues = 'on';
            app.NumberofInjectors.ValueDisplayFormat = '%.0f';
            app.NumberofInjectors.Position = [215 98 105 22];
            app.NumberofInjectors.Value = 1;

            % Create VentDiameterEditFieldLabel
            app.VentDiameterEditFieldLabel = uilabel(app.InjectorConfigurationPanel);
            app.VentDiameterEditFieldLabel.HorizontalAlignment = 'right';
            app.VentDiameterEditFieldLabel.Enable = 'off';
            app.VentDiameterEditFieldLabel.Position = [1 38 82 22];
            app.VentDiameterEditFieldLabel.Text = 'Vent Diameter';

            % Create VentDiameter
            app.VentDiameter = uieditfield(app.InjectorConfigurationPanel, 'numeric');
            app.VentDiameter.Limits = [0 Inf];
            app.VentDiameter.Editable = 'off';
            app.VentDiameter.Enable = 'off';
            app.VentDiameter.Position = [215 38 105 22];

            % Create VentDischargeCoefficientEditFieldLabel
            app.VentDischargeCoefficientEditFieldLabel = uilabel(app.InjectorConfigurationPanel);
            app.VentDischargeCoefficientEditFieldLabel.HorizontalAlignment = 'right';
            app.VentDischargeCoefficientEditFieldLabel.Enable = 'off';
            app.VentDischargeCoefficientEditFieldLabel.Position = [1 8 147 22];
            app.VentDischargeCoefficientEditFieldLabel.Text = 'Vent Discharge Coefficient';

            % Create VentCd
            app.VentCd = uieditfield(app.InjectorConfigurationPanel, 'numeric');
            app.VentCd.Limits = [0 1];
            app.VentCd.Editable = 'off';
            app.VentCd.Enable = 'off';
            app.VentCd.Position = [215 8 105 22];

            % Create VentStateDropDownLabel
            app.VentStateDropDownLabel = uilabel(app.InjectorConfigurationPanel);
            app.VentStateDropDownLabel.HorizontalAlignment = 'right';
            app.VentStateDropDownLabel.Position = [1 68 61 22];
            app.VentStateDropDownLabel.Text = 'Vent State';

            % Create VentState
            app.VentState = uidropdown(app.InjectorConfigurationPanel);
            app.VentState.Items = {'None', 'External', 'Internal'};
            app.VentState.ValueChangedFcn = createCallbackFcn(app, @vent_state, true);
            app.VentState.Position = [215 68 105 22];
            app.VentState.Value = 'None';

            % Create vnt_D_unit
            app.vnt_D_unit = uidropdown(app.InjectorConfigurationPanel);
            app.vnt_D_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.vnt_D_unit.Enable = 'off';
            app.vnt_D_unit.Position = [319 38 75 22];
            app.vnt_D_unit.Value = 'cm';

            % Create inj_D_unit
            app.inj_D_unit = uidropdown(app.InjectorConfigurationPanel);
            app.inj_D_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.inj_D_unit.Editable = 'on';
            app.inj_D_unit.BackgroundColor = [1 1 1];
            app.inj_D_unit.Position = [319 158 75 22];
            app.inj_D_unit.Value = 'cm';

            % Create MassPropertiesPanel
            app.MassPropertiesPanel = uipanel(app.MotorTab);
            app.MassPropertiesPanel.Title = 'Mass Properties';
            app.MassPropertiesPanel.Position = [11 216 401 180];

            % Create OxidizerTankLocationEditFieldLabel
            app.OxidizerTankLocationEditFieldLabel = uilabel(app.MassPropertiesPanel);
            app.OxidizerTankLocationEditFieldLabel.HorizontalAlignment = 'right';
            app.OxidizerTankLocationEditFieldLabel.Enable = 'off';
            app.OxidizerTankLocationEditFieldLabel.Position = [5 127 127 22];
            app.OxidizerTankLocationEditFieldLabel.Text = 'Oxidizer Tank Location';

            % Create TankLocation
            app.TankLocation = uieditfield(app.MassPropertiesPanel, 'numeric');
            app.TankLocation.Limits = [0 Inf];
            app.TankLocation.Editable = 'off';
            app.TankLocation.Enable = 'off';
            app.TankLocation.Position = [234 127 96 22];

            % Create FuelGrainLocationEditFieldLabel
            app.FuelGrainLocationEditFieldLabel = uilabel(app.MassPropertiesPanel);
            app.FuelGrainLocationEditFieldLabel.HorizontalAlignment = 'right';
            app.FuelGrainLocationEditFieldLabel.Enable = 'off';
            app.FuelGrainLocationEditFieldLabel.Position = [5 98 110 22];
            app.FuelGrainLocationEditFieldLabel.Text = 'Fuel Grain Location';

            % Create GrainLocation
            app.GrainLocation = uieditfield(app.MassPropertiesPanel, 'numeric');
            app.GrainLocation.Limits = [0 Inf];
            app.GrainLocation.Editable = 'off';
            app.GrainLocation.Enable = 'off';
            app.GrainLocation.Position = [234 98 96 22];

            % Create EmptyMotorCenterofMassEditFieldLabel
            app.EmptyMotorCenterofMassEditFieldLabel = uilabel(app.MassPropertiesPanel);
            app.EmptyMotorCenterofMassEditFieldLabel.HorizontalAlignment = 'right';
            app.EmptyMotorCenterofMassEditFieldLabel.Enable = 'off';
            app.EmptyMotorCenterofMassEditFieldLabel.Position = [5 68 158 22];
            app.EmptyMotorCenterofMassEditFieldLabel.Text = 'Empty Motor Center of Mass';

            % Create MotorCG
            app.MotorCG = uieditfield(app.MassPropertiesPanel, 'numeric');
            app.MotorCG.Limits = [0 Inf];
            app.MotorCG.Editable = 'off';
            app.MotorCG.Enable = 'off';
            app.MotorCG.Position = [234 68 96 22];

            % Create EmptyMotorMassEditFieldLabel
            app.EmptyMotorMassEditFieldLabel = uilabel(app.MassPropertiesPanel);
            app.EmptyMotorMassEditFieldLabel.HorizontalAlignment = 'right';
            app.EmptyMotorMassEditFieldLabel.Enable = 'off';
            app.EmptyMotorMassEditFieldLabel.Position = [5 38 105 22];
            app.EmptyMotorMassEditFieldLabel.Text = 'Empty Motor Mass';

            % Create MotorMass
            app.MotorMass = uieditfield(app.MassPropertiesPanel, 'numeric');
            app.MotorMass.Limits = [0 Inf];
            app.MotorMass.Editable = 'off';
            app.MotorMass.Enable = 'off';
            app.MotorMass.Position = [234 38 96 22];

            % Create MassProperties
            app.MassProperties = uicheckbox(app.MassPropertiesPanel);
            app.MassProperties.ValueChangedFcn = createCallbackFcn(app, @enable_mass_properties, true);
            app.MassProperties.Text = 'Calculate Mass Properties';
            app.MassProperties.Position = [21 8 269 22];

            % Create tnk_X_unit
            app.tnk_X_unit = uidropdown(app.MassPropertiesPanel);
            app.tnk_X_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.tnk_X_unit.Enable = 'off';
            app.tnk_X_unit.Position = [329 127 60 22];
            app.tnk_X_unit.Value = 'cm';

            % Create cmbr_X_unit
            app.cmbr_X_unit = uidropdown(app.MassPropertiesPanel);
            app.cmbr_X_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.cmbr_X_unit.Enable = 'off';
            app.cmbr_X_unit.Position = [329 98 60 22];
            app.cmbr_X_unit.Value = 'cm';

            % Create mtr_cg_unit
            app.mtr_cg_unit = uidropdown(app.MassPropertiesPanel);
            app.mtr_cg_unit.Items = {'in', 'ft', 'mm', 'cm', 'm'};
            app.mtr_cg_unit.Enable = 'off';
            app.mtr_cg_unit.Position = [329 68 60 22];
            app.mtr_cg_unit.Value = 'cm';

            % Create mtr_m_unit
            app.mtr_m_unit = uidropdown(app.MassPropertiesPanel);
            app.mtr_m_unit.Items = {'lbm', 'kg', 'g', 'oz'};
            app.mtr_m_unit.Enable = 'off';
            app.mtr_m_unit.Position = [329 38 60 22];
            app.mtr_m_unit.Value = 'kg';

            % Create SimulationConfigurationPanel
            app.SimulationConfigurationPanel = uipanel(app.MotorTab);
            app.SimulationConfigurationPanel.Title = 'Simulation Configuration';
            app.SimulationConfigurationPanel.Position = [421 56 398 150];

            % Create MaxSimulationRunTimeEditFieldLabel
            app.MaxSimulationRunTimeEditFieldLabel = uilabel(app.SimulationConfigurationPanel);
            app.MaxSimulationRunTimeEditFieldLabel.HorizontalAlignment = 'right';
            app.MaxSimulationRunTimeEditFieldLabel.Position = [10 99 142 22];
            app.MaxSimulationRunTimeEditFieldLabel.Text = 'Max Simulation Run Time';

            % Create RunTime
            app.RunTime = uieditfield(app.SimulationConfigurationPanel, 'numeric');
            app.RunTime.Limits = [0 Inf];
            app.RunTime.Position = [215 99 105 22];
            app.RunTime.Value = 10;

            % Create MaxBurnTimeEditFieldLabel
            app.MaxBurnTimeEditFieldLabel = uilabel(app.SimulationConfigurationPanel);
            app.MaxBurnTimeEditFieldLabel.HorizontalAlignment = 'right';
            app.MaxBurnTimeEditFieldLabel.Position = [10 70 86 22];
            app.MaxBurnTimeEditFieldLabel.Text = 'Max Burn Time';

            % Create BurnTime
            app.BurnTime = uieditfield(app.SimulationConfigurationPanel, 'numeric');
            app.BurnTime.Limits = [0 Inf];
            app.BurnTime.Position = [215 70 105 22];

            % Create SimulationTimestepEditFieldLabel
            app.SimulationTimestepEditFieldLabel = uilabel(app.SimulationConfigurationPanel);
            app.SimulationTimestepEditFieldLabel.HorizontalAlignment = 'right';
            app.SimulationTimestepEditFieldLabel.Position = [10 40 114 22];
            app.SimulationTimestepEditFieldLabel.Text = 'Simulation Timestep';

            % Create Timestep
            app.Timestep = uieditfield(app.SimulationConfigurationPanel, 'numeric');
            app.Timestep.Limits = [0 Inf];
            app.Timestep.Position = [215 40 105 22];
            app.Timestep.Value = 0.001;

            % Create RegressionModelDropDownLabel
            app.RegressionModelDropDownLabel = uilabel(app.SimulationConfigurationPanel);
            app.RegressionModelDropDownLabel.HorizontalAlignment = 'right';
            app.RegressionModelDropDownLabel.Position = [10 9 102 22];
            app.RegressionModelDropDownLabel.Text = 'Regression Model';

            % Create RegressionModel
            app.RegressionModel = uidropdown(app.SimulationConfigurationPanel);
            app.RegressionModel.Items = {'Constant OF', 'Shifting OF'};
            app.RegressionModel.ValueChangedFcn = createCallbackFcn(app, @reg_model, true);
            app.RegressionModel.Position = [215 9 105 22];
            app.RegressionModel.Value = 'Constant OF';

            % Create runtime_label
            app.runtime_label = uilabel(app.SimulationConfigurationPanel);
            app.runtime_label.HorizontalAlignment = 'center';
            app.runtime_label.Position = [319 99 25 22];
            app.runtime_label.Text = 's';

            % Create burntime_label
            app.burntime_label = uilabel(app.SimulationConfigurationPanel);
            app.burntime_label.HorizontalAlignment = 'center';
            app.burntime_label.Position = [319 70 25 22];
            app.burntime_label.Text = 's';

            % Create timestep_label
            app.timestep_label = uilabel(app.SimulationConfigurationPanel);
            app.timestep_label.HorizontalAlignment = 'center';
            app.timestep_label.Position = [319 40 25 22];
            app.timestep_label.Text = 's';

            % Create InitialConditionsPanel
            app.InitialConditionsPanel = uipanel(app.MotorTab);
            app.InitialConditionsPanel.Title = 'Initial Conditions';
            app.InitialConditionsPanel.Position = [11 56 401 150];

            % Create TankCond
            app.TankCond = uieditfield(app.InitialConditionsPanel, 'numeric');
            app.TankCond.Limits = [0 Inf];
            app.TankCond.Position = [230 98 100 22];
            app.TankCond.Value = 293.15;

            % Create StartingChamberPressureEditFieldLabel
            app.StartingChamberPressureEditFieldLabel = uilabel(app.InitialConditionsPanel);
            app.StartingChamberPressureEditFieldLabel.HorizontalAlignment = 'right';
            app.StartingChamberPressureEditFieldLabel.Position = [5 69 151 22];
            app.StartingChamberPressureEditFieldLabel.Text = 'Starting Chamber Pressure';

            % Create ChamberPressure
            app.ChamberPressure = uieditfield(app.InitialConditionsPanel, 'numeric');
            app.ChamberPressure.Limits = [0 7251000];
            app.ChamberPressure.Position = [230 69 100 22];
            app.ChamberPressure.Value = 1;

            % Create OxidizerFill
            app.OxidizerFill = uieditfield(app.InitialConditionsPanel, 'numeric');
            app.OxidizerFill.Limits = [0 Inf];
            app.OxidizerFill.ValueDisplayFormat = '%11.3g';
            app.OxidizerFill.Position = [230 39 100 22];

            % Create AmbientPressureEditField_2Label
            app.AmbientPressureEditField_2Label = uilabel(app.InitialConditionsPanel);
            app.AmbientPressureEditField_2Label.HorizontalAlignment = 'right';
            app.AmbientPressureEditField_2Label.Position = [10 9 101 22];
            app.AmbientPressureEditField_2Label.Text = 'Ambient Pressure';

            % Create AmbientPressure
            app.AmbientPressure = uieditfield(app.InitialConditionsPanel, 'numeric');
            app.AmbientPressure.Limits = [0 Inf];
            app.AmbientPressure.Position = [230 9 100 22];
            app.AmbientPressure.Value = 1;

            % Create TankDropDown
            app.TankDropDown = uidropdown(app.InitialConditionsPanel);
            app.TankDropDown.Items = {'Starting Tank Temperature', 'Starting Tank Pressure'};
            app.TankDropDown.ValueChangedFcn = createCallbackFcn(app, @tank_condition, true);
            app.TankDropDown.Position = [10 97 181 22];
            app.TankDropDown.Value = 'Starting Tank Temperature';

            % Create OxidizerDropDown
            app.OxidizerDropDown = uidropdown(app.InitialConditionsPanel);
            app.OxidizerDropDown.Items = {'Starting Oxidizer Mass', 'Tank Fill Percentage'};
            app.OxidizerDropDown.ValueChangedFcn = createCallbackFcn(app, @tank_fill, true);
            app.OxidizerDropDown.Position = [11 38 181 22];
            app.OxidizerDropDown.Value = 'Starting Oxidizer Mass';

            % Create T_tnk_unit
            app.T_tnk_unit = uidropdown(app.InitialConditionsPanel);
            app.T_tnk_unit.Items = {'F', 'C', 'K', 'R'};
            app.T_tnk_unit.Editable = 'on';
            app.T_tnk_unit.BackgroundColor = [1 1 1];
            app.T_tnk_unit.Position = [329 98 60 22];
            app.T_tnk_unit.Value = 'K';

            % Create P_cmbr_unit
            app.P_cmbr_unit = uidropdown(app.InitialConditionsPanel);
            app.P_cmbr_unit.Items = {'psi', 'psf', 'atm', 'Pa', 'kPa', 'Bar', 'MPa'};
            app.P_cmbr_unit.Editable = 'on';
            app.P_cmbr_unit.BackgroundColor = [1 1 1];
            app.P_cmbr_unit.Position = [329 69 60 22];
            app.P_cmbr_unit.Value = 'atm';

            % Create Pa_unit
            app.Pa_unit = uidropdown(app.InitialConditionsPanel);
            app.Pa_unit.Items = {'psi', 'psf', 'atm', 'Pa', 'kPa', 'Bar', 'MPa'};
            app.Pa_unit.Editable = 'on';
            app.Pa_unit.BackgroundColor = [1 1 1];
            app.Pa_unit.Position = [329 9 60 22];
            app.Pa_unit.Value = 'atm';

            % Create m_O_unit
            app.m_O_unit = uidropdown(app.InitialConditionsPanel);
            app.m_O_unit.Items = {'lbm', 'kg', 'g', 'oz'};
            app.m_O_unit.Editable = 'on';
            app.m_O_unit.BackgroundColor = [1 1 1];
            app.m_O_unit.Position = [329 39 60 22];
            app.m_O_unit.Value = 'kg';

            % Create RunSimulation
            app.RunSimulation = uibutton(app.MotorTab, 'push');
            app.RunSimulation.ButtonPushedFcn = createCallbackFcn(app, @run_sim, true);
            app.RunSimulation.BackgroundColor = [0.9608 0.9608 0.9608];
            app.RunSimulation.FontSize = 24;
            app.RunSimulation.Position = [331 9 178 37];
            app.RunSimulation.Text = 'Run Simulation';

            % Create LoadMotor
            app.LoadMotor = uibutton(app.MotorTab, 'push');
            app.LoadMotor.ButtonPushedFcn = createCallbackFcn(app, @load_config, true);
            app.LoadMotor.Position = [11 24 176 22];
            app.LoadMotor.Text = 'Load Simulation Configuration';

            % Create SaveMotor
            app.SaveMotor = uibutton(app.MotorTab, 'push');
            app.SaveMotor.ButtonPushedFcn = createCallbackFcn(app, @save_config, true);
            app.SaveMotor.Position = [641 24 176 22];
            app.SaveMotor.Text = 'Save Simulation Configuration';

            % Create ResultsTab
            app.ResultsTab = uitab(app.AppTabs);
            app.ResultsTab.Title = 'Results';

            % Create UIAxes
            app.UIAxes = uiaxes(app.ResultsTab);
            title(app.UIAxes, 'Thrust vs Time')
            app.UIAxes.XTickLabelRotation = 0;
            app.UIAxes.YTickLabelRotation = 0;
            app.UIAxes.ZTickLabelRotation = 0;
            app.UIAxes.XGrid = 'on';
            app.UIAxes.YGrid = 'on';
            app.UIAxes.Position = [11 265 810 535];

            % Create XaxisDropDownLabel
            app.XaxisDropDownLabel = uilabel(app.ResultsTab);
            app.XaxisDropDownLabel.HorizontalAlignment = 'right';
            app.XaxisDropDownLabel.Enable = 'off';
            app.XaxisDropDownLabel.Position = [451 194 39 22];
            app.XaxisDropDownLabel.Text = 'X-axis';

            % Create Xaxis
            app.Xaxis = uidropdown(app.ResultsTab);
            app.Xaxis.Items = {'Time', 'Oxidizer Mass', 'Fuel Mass', 'Total Motor Mass', 'Center of Mass', 'Chamber Pressure', 'Total Propellant Mass', 'Tank Pressure', 'Oxidizer Mass Flow Rate', 'Fuel Mass Flow Rate', 'Total Mass Flow Rate', 'OF Ratio', 'Regression Rate', 'Thrust', 'Injector Pressure Drop'};
            app.Xaxis.ValueChangedFcn = createCallbackFcn(app, @x_axis, true);
            app.Xaxis.Enable = 'off';
            app.Xaxis.Position = [636 194 180 22];
            app.Xaxis.Value = 'Time';

            % Create PlotTitleEditFieldLabel
            app.PlotTitleEditFieldLabel = uilabel(app.ResultsTab);
            app.PlotTitleEditFieldLabel.HorizontalAlignment = 'right';
            app.PlotTitleEditFieldLabel.Enable = 'off';
            app.PlotTitleEditFieldLabel.Position = [453 223 52 22];
            app.PlotTitleEditFieldLabel.Text = 'Plot Title';

            % Create PlotTitle
            app.PlotTitle = uieditfield(app.ResultsTab, 'text');
            app.PlotTitle.Editable = 'off';
            app.PlotTitle.Enable = 'off';
            app.PlotTitle.Position = [513 223 305 22];
            app.PlotTitle.Value = 'Thrust vs Time';

            % Create YaxisDropDownLabel
            app.YaxisDropDownLabel = uilabel(app.ResultsTab);
            app.YaxisDropDownLabel.HorizontalAlignment = 'right';
            app.YaxisDropDownLabel.Enable = 'off';
            app.YaxisDropDownLabel.Position = [451 164 38 22];
            app.YaxisDropDownLabel.Text = 'Y-axis';

            % Create Yaxis
            app.Yaxis = uidropdown(app.ResultsTab);
            app.Yaxis.Items = {'Time', 'Oxidizer Mass', 'Fuel Mass', 'Total Propellant Mass', 'Total Motor Mass', 'Center of Mass', 'Chamber Pressure', 'Tank Pressure', 'Oxidizer Mass Flow Rate', 'Fuel Mass Flow Rate', 'Total Mass Flow Rate', 'OF Ratio', 'Regression Rate', 'Thrust', 'Injector Pressure Drop'};
            app.Yaxis.ValueChangedFcn = createCallbackFcn(app, @y_axis, true);
            app.Yaxis.Enable = 'off';
            app.Yaxis.Position = [636 164 180 22];
            app.Yaxis.Value = 'Thrust';

            % Create PerformanceSummaryTextAreaLabel
            app.PerformanceSummaryTextAreaLabel = uilabel(app.ResultsTab);
            app.PerformanceSummaryTextAreaLabel.BackgroundColor = [0.9412 0.9412 0.9412];
            app.PerformanceSummaryTextAreaLabel.HorizontalAlignment = 'right';
            app.PerformanceSummaryTextAreaLabel.Position = [152 250 129 22];
            app.PerformanceSummaryTextAreaLabel.Text = 'Performance Summary';

            % Create PerformanceSummary
            app.PerformanceSummary = uitextarea(app.ResultsTab);
            app.PerformanceSummary.Editable = 'off';
            app.PerformanceSummary.BackgroundColor = [0.8 0.8 0.8];
            app.PerformanceSummary.Position = [12 6 410 246];
            app.PerformanceSummary.Value = {'You must first run a simulation to view results or plot data'};

            % Create AddPlot
            app.AddPlot = uibutton(app.ResultsTab, 'push');
            app.AddPlot.ButtonPushedFcn = createCallbackFcn(app, @plot, true);
            app.AddPlot.Enable = 'off';
            app.AddPlot.Position = [450 35 100 22];
            app.AddPlot.Text = 'Add Plot';

            % Create ClearPlot
            app.ClearPlot = uibutton(app.ResultsTab, 'push');
            app.ClearPlot.ButtonPushedFcn = createCallbackFcn(app, @clear_plot, true);
            app.ClearPlot.Enable = 'off';
            app.ClearPlot.Position = [717 35 100 22];
            app.ClearPlot.Text = 'Clear Plot';

            % Create SavePlot
            app.SavePlot = uibutton(app.ResultsTab, 'push');
            app.SavePlot.ButtonPushedFcn = createCallbackFcn(app, @save_plot, true);
            app.SavePlot.Enable = 'off';
            app.SavePlot.Position = [583 35 100 22];
            app.SavePlot.Text = 'Save Plot';

            % Create SaveResults
            app.SaveResults = uibutton(app.ResultsTab, 'push');
            app.SaveResults.ButtonPushedFcn = createCallbackFcn(app, @save_results, true);
            app.SaveResults.Enable = 'off';
            app.SaveResults.Position = [717 5 100 22];
            app.SaveResults.Text = 'Save Results';

            % Create ExportRSE
            app.ExportRSE = uibutton(app.ResultsTab, 'push');
            app.ExportRSE.ButtonPushedFcn = createCallbackFcn(app, @save_rse, true);
            app.ExportRSE.Enable = 'off';
            app.ExportRSE.Position = [450 5 100 22];
            app.ExportRSE.Text = 'Export .RSE';

            % Create ExportCSV
            app.ExportCSV = uibutton(app.ResultsTab, 'push');
            app.ExportCSV.ButtonPushedFcn = createCallbackFcn(app, @save_csv, true);
            app.ExportCSV.Enable = 'off';
            app.ExportCSV.Position = [583 5 100 22];
            app.ExportCSV.Text = 'Export .CSV';

            % Create XaxisUnitsDropDownLabel
            app.XaxisUnitsDropDownLabel = uilabel(app.ResultsTab);
            app.XaxisUnitsDropDownLabel.HorizontalAlignment = 'right';
            app.XaxisUnitsDropDownLabel.Enable = 'off';
            app.XaxisUnitsDropDownLabel.Position = [451 134 70 22];
            app.XaxisUnitsDropDownLabel.Text = 'X-axis Units';

            % Create XaxisUnits
            app.XaxisUnits = uidropdown(app.ResultsTab);
            app.XaxisUnits.Items = {'s'};
            app.XaxisUnits.Enable = 'off';
            app.XaxisUnits.Position = [636 134 180 22];
            app.XaxisUnits.Value = 's';

            % Create YaxisUnitsDropDownLabel
            app.YaxisUnitsDropDownLabel = uilabel(app.ResultsTab);
            app.YaxisUnitsDropDownLabel.HorizontalAlignment = 'right';
            app.YaxisUnitsDropDownLabel.Enable = 'off';
            app.YaxisUnitsDropDownLabel.Position = [452 104 69 22];
            app.YaxisUnitsDropDownLabel.Text = 'Y-axis Units';

            % Create YaxisUnits
            app.YaxisUnits = uidropdown(app.ResultsTab);
            app.YaxisUnits.Items = {'N', 'lbf', 'kgf'};
            app.YaxisUnits.Enable = 'off';
            app.YaxisUnits.Position = [636 104 180 22];
            app.YaxisUnits.Value = 'N';

            % Create LegendEditFieldLabel
            app.LegendEditFieldLabel = uilabel(app.ResultsTab);
            app.LegendEditFieldLabel.HorizontalAlignment = 'right';
            app.LegendEditFieldLabel.Enable = 'off';
            app.LegendEditFieldLabel.Position = [453 73 46 22];
            app.LegendEditFieldLabel.Text = 'Legend';

            % Create Legend
            app.Legend = uieditfield(app.ResultsTab, 'text');
            app.Legend.Editable = 'off';
            app.Legend.Enable = 'off';
            app.Legend.Position = [513 73 305 22];
            app.Legend.Value = 'Thrust';

            % Create AboutTab
            app.AboutTab = uitab(app.AppTabs);
            app.AboutTab.Title = 'About';

            % Create TextArea
            app.TextArea = uitextarea(app.AboutTab);
            app.TextArea.Editable = 'off';
            app.TextArea.BackgroundColor = [0.8 0.8 0.8];
            app.TextArea.Position = [11 9 810 786];
            app.TextArea.Value = {'The Hybrid Rocket Analysis Program (HRAP) was developed by Robert (Drew) Nickel for use by the University of Tennessee Rocket Engineering Team, part of the Student Space Technology Association at UTK. HRAP is a versatile tool utilizing a thermodynamic equilibrium model for simulation of self-pressurizing hybrid rocket motors, especially those powered with Nitrous Oxide stored as a saturated liquid-vapor mixture. HRAP models all phases of a typical nitrous oxide hybrid rocket burn, with an equilibrium tank model, a transient chamber pressure model, options for both constant OF and shifting OF regression models, an isentropic nozzle model, and a simple but effective mass properties model. This program can be used to model flight motors burning to tank depletion or boilerplate motors with a set burn time, and can generate and export all data and plots as well as compile a .RSE file for use in flight simulations, future iterations will account for subsonic flow and flow separation to better model thrust, non-equilibrium tank models, and two phase injector flow. For an in-depth look at how HRAP works, there is a document linked below. For bug reports or suggestions, send a screenshot, explanation and any additional details to: rnickel1@vols.utk.edu'; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; ''; 'Acknowledgements: I would like to thank Dr Richard Newlands with Aspire Space for his extensive help in the development of HRAP. His hybrid rocket model laid the foundation for HRAP, and I could not have gotten this far without him. I would like to thank James Anderson with Equatorial Space Industries in Singapore for providing test data for model verification. Lastly, I would like to thank Dr. Evans Lyne, Dr. Kuvanc Ekici, Dr. Mark Barker, Dr. Matthew Mench, and Dr. Robert Jacobsen from the University of Tennessee, Thomas Sanders from Contrail Rockets LLC, and James Hudspeth from Imperia Aerospace for their academic and/or financial support of this endeavor.'};

            % Create ClickhereformoreinformationregardingHRAPitsusageandButton
            app.ClickhereformoreinformationregardingHRAPitsusageandButton = uibutton(app.AboutTab, 'push');
            app.ClickhereformoreinformationregardingHRAPitsusageandButton.ButtonPushedFcn = createCallbackFcn(app, @open_hrap_document, true);
            app.ClickhereformoreinformationregardingHRAPitsusageandButton.Position = [221 134 414 22];
            app.ClickhereformoreinformationregardingHRAPitsusageandButton.Text = 'Click here for more information regarding HRAP, its usage, and validation.';

            % Create Image
            app.Image = uiimage(app.AboutTab);
            app.Image.Position = [54 228 452 349];
            app.Image.ImageSource = 'test fire.jpg';

            % Create Label
            app.Label = uilabel(app.AboutTab);
            app.Label.Position = [43 186 770 22];
            app.Label.Text = 'Image: University of Tennessee 3,000 Newton Research Hybrid (left) and 5,000 Newton sounding rocket (right), both developed using HRAP.';

            % Create Image2
            app.Image2 = uiimage(app.AboutTab);
            app.Image2.Position = [540 229 234 349];
            app.Image2.ImageSource = 'Launch.jpg';

            % Create AuthorLabel
            app.AuthorLabel = uilabel(app.HRAP_v2022_06_04);
            app.AuthorLabel.Position = [711 -1 130 22];
            app.AuthorLabel.Text = 'Robert A. Nickel - 2023';

            % Create ReleaseLabel
            app.ReleaseLabel = uilabel(app.HRAP_v2022_06_04);
            app.ReleaseLabel.Position = [12 -1 114 22];
            app.ReleaseLabel.Text = 'Release 2023-07-14';

            % Show the figure after all components are created
            app.HRAP_v2022_06_04.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = HRAP

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.HRAP_v2022_06_04)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.HRAP_v2022_06_04)
        end
    end
end