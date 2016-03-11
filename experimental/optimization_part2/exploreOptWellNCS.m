%% Obtain optimized injection rates by maximizing an objective function.

% NB: export_fig used below. Ensure you have downloaded it from
% <http://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig>
assert(exist('export_fig')~=0, 'Ensure export_fig exists and is on path.')
moduleCheck('mrst-gui')
moduleCheck('ad-core')

% -All inputs are explicitly defined here and passed into
% optimizeFormation_extras.m (i.e., probably no default values are used
% inside optimizeFormation_extras.m)

% -Wells can be either bhp-controlled or rate-controlled.
% -Well placement can be an array covering the whole or part of the
% formation, or placed in best leaf nodes using internal function in
% optimizeFormations_extras.m


% Main Directory name for saving results (each cases results will be put
% into a subdirectory named according to InjYrs and MigYrs)
varDirName = 'testing2/opt_results_one_per_trap_highest_pt_OpenBdrys_stricterTol';

% Figure Directory name
figDirName = [varDirName '/' 'WellPlacementFigs_one_per_trap_highest_pt'];
mkdir(figDirName)

names = [getBarentsSeaNames() getNorwegianSeaNames() getNorthSeaNames()];

% Remove certain formation names:
names = names(~strcmpi(names,'Nordmelafm'));
names = names(~strcmpi(names,'Rorfm'));
names = names(~strcmpi(names,'Notfm'));
names = names(~strcmpi(names,'Knurrfm'));       % @@ can be constructed??
names = names(~strcmpi(names,'Fruholmenfm'));   % @@
names = names(~strcmpi(names,'Cookfm'));
names = names(~strcmpi(names,'Dunlingp'));
names = names(~strcmpi(names,'Paleocene'));

% Remove ones already run:
% names = names(~strcmpi(names,'Arefm'));
%names = names(~strcmpi(names,'Bjarmelandfm'));
% names = names(~strcmpi(names,'Brentgrp'));
% names = names(~strcmpi(names,'Brynefm'));
% names = names(~strcmpi(names,'Garnfm'));
% names = names(~strcmpi(names,'Ilefm'));
%names = names(~strcmpi(names,'Stofm'));
%names = names(~strcmpi(names,'Tiljefm'));
%names = names(~strcmpi(names,'Tubaenfm'));
% names = names(~strcmpi(names,'Fensfjordfm'));
% names = names(~strcmpi(names,'Krossfjordfm'));
% names = names(~strcmpi(names,'Sognefjordfm'));

% Ensure no repeting names
names = unique(names,'stable');

% Load res containing formation names and their coarsening levels.
% Or get res from testing_coarsening_levels()
%load coarsening_levels_dx3000meter.mat;  n = {res{:,1}}; c_level = {res{:,2}};
load coarsening_levels_70percent_of_full_StrapCap.mat;
n       = {names_and_cellsizes{:,1}};
c_level = {names_and_cellsizes{:,3}};

shared_names = intersect(names, n, 'stable');
assert( numel(shared_names) >= numel(names) )
assert( all(strcmpi(sort(shared_names),sort(names)))==1 )


% if running multiple runs, adjust figure visibility
% unless each figure generated in a way to avoid stealing focus
set(0, 'DefaultFigureVisible', 'off');

names = {'Sandnesfm'}

for i=1:numel(names)
    
    fprintf('-------------- FORMATION: %s -----------------\n', names{i})
    fmName      = names{i};
    rhoCref     = 760 * kilogram / meter ^3;

    inx             = find(strcmp(fmName,n));
    coarsening      = c_level{inx};
    [Gt, rock2D]    = getFormationTopGrid( fmName, coarsening );
    if any(isnan(rock2D.perm))
        rock2D.perm = 500*milli*darcy * ones(Gt.cells.num,1);
    end
    if any(isnan(rock2D.poro))
        rock2D.poro = 0.25 * ones(Gt.cells.num,1); 
    end
    
    seainfo = getSeaInfo(fmName, rhoCref);
    gravity on;
    caprock_pressure = (Gt.cells.z * seainfo.water_density * norm(gravity)) ...
                .* (1 + seainfo.press_deviation/100);

    max_rate_fac = 2;
    if any(strcmpi(fmName,{'Sleipnerfm','Huginfmwest','Ulafm','Huginfmeast','Pliocenesand'}))
        max_rate_fac = 4;  % @@ max rates will depend on the starting rates
    %elseif any(strcmpi(fmName,{''}))
    %    max_rate_fac = 4;
    end

    %try
    

        %%% Pass everything in explicitly.
        clear Gt optim init history other
        cp = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
        sch = [];
        max_wvals = []; % will be computed internally using rate_lim_fac
        for r = 1:numel(cp)
            % assuming penalize pressure is on, get optimized rates for a given cp
            [Gt, optim, init, history, other] = optimizeFormation_extras(...
            'dryrun'                         , false                        , ...
            'inspectWellPlacement'           , false                         , ... % if true, will not continue to simulation
            'adjustClosedSystemRates'        , true                         , ... % applied to internally computed rates only, not passed in schedule. Only applied if system closed
            'lineSearchMaxIt'                , 10                           , ...
            'gradTol'                        , 1e-4                         , ...
            'objChangeTol'                   , 1e-4                         , ...
            'modelname'                      , fmName                       , ...
                'coarse_level'               , coarsening                   , ...
            'schedule'                       , sch, ... %'opt_results_Array_in_trap_regions_Pressure_plim90/Sleipnerfm/InjYrs50_MigYrs2_DissOn_0_rateLim10_pre/new_schedule'                           , ...
                'itime'                      , 50 * year                    , ...
                'isteps'                     , 50                           , ...
                'mtime'                      , 2000 * year                   , ...
                'msteps'                     , 200                          , ... 
            'well_placement_type'            , 'one_per_trap'                  , ... % 'use_array', 'one_per_trap', 'one_per_path'
                'max_num_wells'              , 40                           , ... % used in use_array, one_per_trap
                'maximise_boundary_distance' , false                        , ... % used in one_per_path
                'well_buffer_dist'           , 1 * kilo * meter             , ... % dist from edge of internal catchment
                'well_buffer_dist_domain'    , 5 * kilo * meter             , ... % dist from edge of domain    
                'well_buffer_dist_catchment' , 3 * kilo * meter             , ... % dist from edge of external catchment
                'pick_highest_pt'            , true                       , ... % otherwise farthest downslope, used in one_per_trap and one_per_path
                'DX'                         , 1 * kilo*meter               , ... % used in use_array
                'DY'                         , 1 * kilo*meter               , ... % used in use_array
            'well_control_type'              , 'rate'                       , ...
                'rate_lim_fac'               , max_rate_fac                 , ...
                'max_wvals'                  , max_wvals                    , ...
            'btype'                          , 'pressure'                   , ...
            'penalize_type'                  , 'leakage'                    , ... % 'leakage', 'leakage_at_infinity', 'pressure'
                'leak_penalty'               , 10                           , ...
                'pressure_penalty'           , cp(r) , ... %3.0963e-12                        , ... % @@ get appropriate penalty with trial-and-error
                'p_lim_factor'               , 0.9                       , ...
            'surface_pressure'              , 1 * atm                       , ...   
            'refRhoCO2'                     , seainfo.rhoCref               , ...
            'rhoW'                          , seainfo.water_density         , ...
            'muBrine'                       , seainfo.water_mu              , ...
            'muCO2'                         , 0                             , ... % zero value will trigger use of variable viscosity 
            'pvMult'                        , 1e-5/barsa                    , ...
            'refPress'                      , mean(caprock_pressure)        , ... % @@ ? 
            'c_water'                       , 4.3e-5/barsa                  , ... % water compressibility
            'p_range'                       , [0.1, 400] * mega * Pascal    , ... % pressure span of sampled property table
            't_range'                       , [4 250] + 274                 , ...
            'sr'                            , seainfo.res_sat_co2                   , ... % gas
            'sw'                            , seainfo.res_sat_wat                   , ... % brine
            'ref_temp'                      , seainfo.seafloor_temp + 273.15        , ...
            'ref_depth'                     , seainfo.seafloor_depth                , ... 
            'temp_grad'                     , seainfo.temp_gradient                 , ...
            'dissolution'                   , false                                  , ...
                'dis_rate'                  , 0                             , ... % 0 means instantaneous, 0.44 * kilogram / rho / poro / (meter^2) / year = 8.6924e-11;
                'dis_max'                   , 53/seainfo.rhoCref            , ... % 53/760 = 0.07; % 1 kg water holds 0.07 kg of CO2
            'report_basedir'                , './simulateUtsira_results/'   , ... % directory for saving reslts    
            'trapfile_name'                 , []                            , ... % 'utsira_subtrap_function_3.mat'
            'surf_topo'                     , 'smooth' );

            % Save well inspection figure if it was generated,
            % then go to next formation
            if isfield(other,'inspectWellPlacement')
                % Save figure:
                pause(1)
                %saveas(figure(100), [figDirName '/' fmName '_wellPlacement'], 'fig')
                export_fig(figure(100),[figDirName '/' fmName '_wellPlacement'], '-png','-transparent')
                break % exit cp (r) loop, try to save variables, then go to next formation
            end
            
            % keep the 'first' initial schedule for post-processing, and
            % keep the 'first' max well values for further cp iterations
            if r == 1
               init0 = init;
               max_wvals = other.opt.rate_lim_fac * ...
                    max([init0.schedule.control(1).W.val]) * ones(numel(init0.schedule.control(1).W), 1);
            end
            if strcmpi(other.opt.penalize_type,'pressure')
                % Was pressure limit plus a tolerance surpassed? If yes, use
                % next higher cp value. If within plim + tolerance, results
                % obtained with cp value were acceptable. If system is closed
                % and pressure is under plim, rates could be higher, thus
                % increase initial rates slightly and go to next cp. If system
                % is closed and pressure is under plim, okay.
                plim = other.opt.p_lim_factor * other.P_over;
                [perc_of_plim_reach, perc_of_Pover_reach] = ...
                    report_maxPercentage_plim_reached( optim.states, plim, other.P_over );
                if perc_of_Pover_reach/100 - other.opt.p_lim_factor > 0.02
                    % use optimized rates as next iteration's initial rates
                    sch = optim.schedule;
                elseif strcmpi(other.opt.btype,'flux') && perc_of_Pover_reach/100 < other.opt.p_lim_factor
                    % plim was not reached in closed system, thus rates could
                    % be higher. Increase next iteration's initial rates by
                    % some percentage, while within the max rate limit
                    sch = optim.schedule;
                    for wi=1:numel([sch.control(1).W.val])
                        if (sch.control(1).W(wi).val * 1.25) <= max_wvals(wi)
                            sch.control(1).W(wi).val = sch.control(1).W(wi).val * 1.25; % 25 percent increase
                        else
                            sch.control(1).W(wi).val = max_wvals(wi);
                        end
                    end
                else
                    % closed system: if plim < p < (plim + tolerance), thus optimal rates found
                    % open system: if p < (plim + tolerance), optimal rates found
                    break % exit cp (r) loop, try to save variables, then go to next formation
                end
            else
                % penalize leakage or leakage at infinity (without
                % pressure) doesn't require iteratively increasing cp.
                break
            end

            
        
        end
        
        %
        % Save variables
        subVarDirName = [varDirName '/' fmName '/' ...
            'InjYrs',num2str(convertTo(other.opt.itime,year)), ...
            '_MigYrs',num2str(convertTo(other.opt.mtime,year)), ...
            '_DissOn_',num2str(other.dissolution), ...
            '_adjustInitRates',num2str(other.adjustClosedSystemRates)];
        mkdir(subVarDirName);
        save([subVarDirName '/' 'Gt'], 'Gt'); % '-v7.3'); using v7.3 makes size larger!
        save([subVarDirName '/' 'optim'], 'optim');
        save([subVarDirName '/' 'init0'], 'init0');
        save([subVarDirName '/' 'init'], 'init');
        save([subVarDirName '/' 'history'], 'history');
        save([subVarDirName '/' 'other'], 'other'); % does not contain other.fluid
        %
        % Save optimization iterations/details:
        % (Avoid stealing focus if figures already exists)
        set(0, 'CurrentFigure', 10);
        saveas(10,[subVarDirName '/' fmName '_optDetails'], 'fig')
        close(10)
        %
        set(0, 'CurrentFigure', 11);
        saveas(11,[subVarDirName '/' fmName '_optDetails_2'], 'fig')
        close(11)
        %
        set(0, 'CurrentFigure', 50); % only exists if penalizing pressure
        saveas(50,[subVarDirName '/' fmName '_optDetails_3'], 'fig')
        close(50)
        %
        close all
        
    %catch
        % continue the 'for loop' if code under 'try' either finished or
        % failed
    %end



end



            %%% Rate-controlled wells:
        % injected vols could be reduced to avoid neg compressibility values
        % that result from very high injection rates.
        %wellinfo.vols_inj = wellinfo.vols_inj .* 0.2; % 0.1 worked, not 0.2 or higher

    %     % Set initial rates according to initial bhp's
    %     surface_pressure    = 1 * atm;
    %     inSt.pressure  = seainfo.water_density * norm(gravity()) ...
    %         * Gt.cells.z + surface_pressure; %@@ contribution from surf_press is small
    %     s = setSchedule_extras( Gt, rock2D, wellinfo.cinx_inj, 'bhp', ...
    %                                     isteps, itime, msteps, mtime, ...
    %                                     'initState', inSt, ...
    %                                     'initOverP', 10 * mega * Pascal, ...
    %                                     'minval',    sqrt(eps));
    % 
    %     % we want to convert the bhp of the wells into the corresponding flux.
    %     % To do this, we must compute the mobility of co2 in wells, which is
    %     % the relative permeability in the well (assume = 1) divided by the co2
    %     % viscosity (a function of temperature and bhp). Grab the viscosity
    %     % from makeVEFluid:
    %     fluid = makeVEFluid(Gt, rock2D, 'sharp interface'                   , ...
    %                    'fixedT'       ,  caprock_temperature                         , ...
    %                    'wat_rho_pvt'  , [4.3e-5/barsa  , mean(caprock_pressure)] , ...
    %                    'residual'     , [res_sat_wat   , res_sat_co2]       , ...
    %                    'wat_rho_ref'  , water_density                       , ...
    %                    'co2_rho_ref'  , rhoCref                  , ... 
    %                    'wat_rho_pvt'  , [4.3e-5/barsa   , mean(caprock_pressure)] , ...
    %                    'co2_rho_pvt'  , [[0.1, 400] * mega * Pascal   , [4 250] + 274]  , ...
    %                    'co2_mu_pvt'   , [[0.1, 400] * mega * Pascal   , [4 250] + 274]  , ...
    %                    'wat_mu_ref'   , water_mu                    , ...
    %                    'pvMult_fac'   , 1e-5/barsa                     , ...
    %                    'dissolution'  , false                          , ...
    %                    'pvMult_p_ref' , mean(caprock_pressure)                   , ...
    %                    'surf_topo'    , 'smooth'                  , ...
    %                    'top_trap'     , []);
    %     mu_g        = fluid.muG(caprock_pressure);
    %     relperm_g   = fluid.krG(ones(Gt.cells.num,1), caprock_pressure); % will be = 1
    %     mob_g       = relperm_g(wellinfo.cinx_inj) ./ mu_g(wellinfo.cinx_inj);
    %     
    %     wellfluxes  = mob_g .* [s.control(1).W.WI]' .* ...
    %         ([s.control(1).W.val]' - inSt.pressure(wellinfo.cinx_inj));
    %     wellfluxes  = wellfluxes ./ Gt.cells.H(wellinfo.cinx_inj) ./ rock2D.poro(wellinfo.cinx_inj); % @@??
    %     vols_inj    = wellfluxes .* itime;

% 
%         %%% fluxes (qGr) obtained at first time step in initial solution using
%         %%% BHP wells:
%         rates= [2.7264;    3.6041;    2.1753;    3.3926;    2.1596;    2.5669; ...
%                 1.0710;    3.1972;    1.2037;    1.4002;    0.8188;    1.1484; ...
%                 1.4871;    2.0375;    2.3875;    2.3250;    1.5426;    1.7688; ...
%                 2.4895;    0.6788;    1.0161]; % m3/2
%         vols_inj = rates .* itime; % m3
%         clear rates