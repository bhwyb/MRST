% small program to test out calculation of sensitivties and optimization
% 
% years of injection
disp('Starting new injection scenario.')
moduleCheck('co2lab','ad-core','opm_gridprocessing','mex','deckformat', ...
    'coarsegrid','upscaling','incomp','mrst-experimental','optimization');
    clear schedule 
    gravity on
    % Trapping analysis method
    isCellBasedMethod = false; % true to use cell-based method, false to use node-based method
    % FOR PLOTS:
    CO2plumeOutline_SatTol  = (0.01/100); % adjust this value if patch error occurs (which happens when no massCO2 present at specified sat tol)
    press_deviation = 0;  % from hydrostatic (percent) --> used for trapping capacity calculation, not simulation
    N = 1; % coarsening level of grid. (no option yet to re-fine or coarsen Sleipner grid). ==> in progress
    
    
    % For plotting of CO2 plumes
    % bounds of 2008 plume:
    ZoomX1 = 0.4375e6;
    ZoomY1 = 6.47e6;
    ZoomX2 = 0.4395e6;
    ZoomY2 = 6.474e6; 
    % OPTION - Select the grid model to load/use:
  
    
  
    % Physical coordinate of injection well (Singh et al. 2010)
    wellXcoord      = 4.38516e5;
    wellYcoord      = 6.47121e6;
    
    
    % OPTION - Well injection rate:
    useRatesFromSPE134891 = true; extrapolateRates = false; % TODO - implement for true (i.e., extrapolation)
    useUserDefinedRates = false; % TODO - implement
    useSleipnerOriginalInjectionRates = false;
    ratecase = 'original'
    switch ratecase
        case 'SPE'
        % Note: variable annual injection rates are given in Singh et al 2010,
        % however it is likely their paper contains a typo. The injection rates
        % were reported as in units of meter^3/day, however a more realistic value
        % implies the units are meter^3/year.
        inj_year   = [1999; 2000; 2001; 2002; 2003; 2004; 2005; 2006; 2007; 2008; 2009];
        inj_rates  = [2.91e4; 5.92e4; 6.35e4; 8.0e4; 1.09e5; 1.5e5; 2.03e5; 2.69e5; 3.47e5; 4.37e5; 5.4e5] .* meter^3/year;
        % inj_rates is in meter^3/s
        case 'original'
        
        [ CumSurfVol_m3, Mass_kg, ReservoirVol_m3, year ] = getSleipnerOriginalInjectionRates();
        % OLD: load SleipnerOriginalInjectionRates.mat
        
        % We only take the years between 1999 and 2030 as injection years since
        % ReservoirVol_m3 is the cumulative value as of Jan 1st of each year
        % 1999 is taken as first injection year since ReservoirVol_m3 is
        % non-zero starting in 2000 (Jan 1st) which implies there was an
        % injection rate for 1999. The ReservoirVol_m3 amount corresponding to
        % 2031 is used to determine the injection rate in the previous year
        % (2030), thus we assume no injection occurs in 2031 (or a value could
        % be extrapolated).
        inj_year    = year(2:end-1);              clear year
        inj_rates   = zeros(1,numel(inj_year));
        
        ReservoirVol_m3 = ReservoirVol_m3(2:end);
        for i = 1:numel(ReservoirVol_m3)-1
            inj_rates(i) = ( ReservoirVol_m3(i+1) - ReservoirVol_m3(i) ); % annual rate
        end
        inj_rates = inj_rates * meter^3/year; % in meter^3/second
        
        otherwise
        error('The injection rate option was either unvalid or not selected.')
        
    end
    
    
    % Specify and compute time step size for injection period.
    % ***Note: inj_time and inj_steps are applied to each inj_rate given***
    num_years   = 10;
    inj_time    = 1 * year; % DEFAULT. CAN ONLY ADJUST NUMBER OF STEPS.
    inj_steps   = 1;
    dTi         = inj_time / inj_steps; % timestep size in seconds
    
    % Specify fluid properties:
    [rho, mu, sr, sw]   = getValuesSPE134891();
    sr=0;sw=0;%NB
    water_density       = rho(1) * kilogram / meter ^3;
    rhoCref             = rho(2) * kilogram / meter ^3;
    
    seafloor_temp       = 7; % Celsius
    seafloor_depth      = 100; % meters
    temp_gradient       = 35.6; % Celsius / km
    %water_compr_val     = 0; %4.3e-5/barsa; % will convert to compr/Pa
    water_compr_val     = 4.3e-5/barsa; %NB will convert to compr/Pa
    %pvMult              = 0; %1e-5/barsa;
    pvMult              = 1e-5/barsa;%NB
    isDissOn            = false;
    dis_max             = (53 * kilogram / meter^3) / rhoCref; % from CO2store
    
    % kwm? 0.75, 0.54 in Appendix of Singh et al 2010.
    
    
    % OPTION - Select which parameters to modify from original data:
    mod_rock        = true;
    mod_rhoCO2      = true;
    
    % Then, set parameter modifier factors:
    por_mod     = 0.6;
    perm_mod    = 3;
    rhoCO2_mod  = 2/3;
    
    
    
    % ************************ END OF USER OPTIONS ****************************
    refineLevel = 1;
    [ G, Gt, rock, rock2D ] = makeSleipnerModelGrid('modelName','ORIGINALmodel', 'refineLevel',refineLevel, 'plotsOn',false);
    %clear G,rock,
    %%{
    dx= (wellXcoord-Gt.cells.centroids(:,1));
    dy= (wellYcoord-Gt.cells.centroids(:,2));
    ind = dx<1e3 & dx>-1.5e3 & dy<1.3e3 & dy >-4e3;
    clf,plotCellData(G,double(ind))
    G=removeCells(G,find(~ind));
    Gt =topSurfaceGrid(G);
    clear G;
    rock2D.perm=rock2D.perm(ind,:);
    rock2D.poro=rock2D.poro(ind);
    %}
    %return
    %% Modify original parameters (optional) and visualize model grids
    if mod_rock
        disp('Original rock parameters are being modified ...')       
        % modify parameters
        %rock.poro   = rock.poro .* por_mod;
        rock2D.poro = rock2D.poro .* por_mod;
        %rock.perm   = rock.perm .* perm_mod;
        rock2D.perm = rock2D.perm .* perm_mod;
    end

    % Get boundary faces of formation (or grid region)
    bf = boundaryFaces(Gt);   
    %% 2. Basic routine to perform VE simulation, using simulateScheduleAD().
    % _________________________________________________________________________
    % a) set up initial state, OR get literature data:
    
    
    if mod_rhoCO2
        disp('Original CO2 density value is being modified ...')
        rhoCref = rhoCref * rhoCO2_mod;
        %rhoCref = 350;
    end  
    initState.pressure  = Gt.cells.z * norm(gravity) * water_density;   % hydrostatic pressure, in Pa=N/m^2
    initState.s         = repmat([1 0], Gt.cells.num, 1);               % sat of water is 1, sat of CO2 is 0
    initState.sGmax     = initState.s(:,2);                             % max sat of CO2 is initially 0
    initState.rs        = 0 * initState.sGmax;                          % initially 0   
    dv = bsxfun(@minus, Gt.cells.centroids(:,1:2), [wellXcoord, wellYcoord]);
    [v,i] = min(sum(dv.^2, 2));   
    wellCellIndex = i; % or Gt.cells.indexMap(i);
    [i, j] = ind2sub(Gt.cartDims, wellCellIndex);
    % Check coordinate that wellCellIndex corresponds to:
    wellCoord_x = Gt.cells.centroids(wellCellIndex,1);
    wellCoord_y = Gt.cells.centroids(wellCellIndex,2);
    wellCoord_z = 0;
    
    
    clear schedule;
    inj_rates_MtPerYr = inj_rates.*(rhoCref/1e9*365*24*60*60); % Mt/year
    % Put into schedule fields --> [inj period 1; inj period 2; etc...; migration period]
    for i = 1:num_years;%numel(inj_rates) %only use 9 years
        schedule.control(i).W = addWell([], Gt.parent, rock2D, wellCellIndex, ...
            'name', sprintf('W%i', i), 'Type', 'rate', 'Val', inj_rates(i), 'comp_i', [0 1]);
    end
    bdryFaces = find( Gt.faces.neighbors(:,1).*Gt.faces.neighbors(:,2) == 0 );
    
    bdryType = 'pressure';
    bdryVal  = Gt.faces.z(bdryFaces) * water_density * norm(gravity);
    % Then use function bc = addBC(bc, faces, type, value, varargin)
    bc = addBC( [], bdryFaces, bdryType, bdryVal, 'sat', [1 0] );   
    % Put into schedule fields --> [injection period; migration period]
    for i = 1:numel(schedule.control)
        schedule.control(i).bc = bc;
    end
    
    
    % TIME STEP:
    
    % For simulation schedule
    istepvec = repmat( ones(inj_steps, 1) * dTi , [num_years 1] );
    %mstepvec = ones(mig_steps, 1) * dTm;    
    % schedule.step.val and schedule.step.control are same size arrays:
    % schedule.step.val is the timestep (size) used for that control step.
    schedule.step.val       = [istepvec];
    % schedule.step.control is a index (1,2,...) indicating which control
    % (i.e., schedule.control) is to be used for the timestep.
    schedule.step.control = [];
    for i = 1:numel(schedule.control)
            schedule.step.control = [schedule.step.control; ones(inj_steps, 1) * i];        
    end   
    % _________________________________________________________________________
    % c) set up model (grid, rock and fluid properties).
    
    caprock_temperature = 273.15 + seafloor_temp + (Gt.cells.z - seafloor_depth) / 1e3 * temp_gradient; % Kelvin
    ref_p           = mean(initState.pressure); % use mean pressure as ref for linear compressibilities   
    fluid = makeVEFluid(Gt, rock2D, 'sharp interface', ...
        'fixedT'      , caprock_temperature, ...
        'wat_mu_ref'  , mu(1), ...
        'co2_mu_ref'  , mu(2), ...
        'wat_rho_ref' , water_density, ...
        'co2_rho_ref' , rhoCref, ...
        'wat_rho_pvt' , [water_compr_val, ref_p], ...
        'co2_rho_pvt' , [water_compr_val*10, ref_p], ...%NB
        'pvMult_p_ref', ref_p, ...
        'pvMult_fac'  , pvMult, ...
        'residual'    , [sw, sr] , ...
        'dissolution' , isDissOn, 'dis_max', dis_max);
    
    %%
   
   
    %elipticsolver = BackslashSolverAD();
    %elipticsolver = AGMGSolverAD('tolerance',1e-3,'reuseSetup',false);
    %linearsolver = CPRSolverAD('tolerance',1e-5,'ellipticSolver',elipticsolver);
    
    linearsolver = BackslashSolverAD();
    it_target=5;
    rampup_dt = 20*day;
    dtmax=1*year;
    timestepper = ...
       IterationCountTimeStepSelector('targetIterationCount', it_target,...
                                      'minRelativeAdjustment', sqrt(eps),...
                                      'maxRelativeAdjustment', 4, ...
                                      'firstRampupStep',       rampup_dt, ...
                                      'verbose', true,'maxTimestep',dtmax);
    nonlinearsolver = NonLinearSolver('timeStepSelector', timestepper, ...
                                'maxiterations', 2*it_target,'LinearSolver',linearsolver);
    
    nonlinearsolver = NonLinearSolver('LinearSolver',linearsolver);
    model = CO2VEBlackOilTypeModel(Gt, rock2D, fluid);
    tmpschedule=schedule;
    tmpschedule.step.control=tmpschedule.step.control(1);
    tmpschedule.step.val=tmpschedule.step.val(1);
    %    
    %{
    profile off
    profile on
    [wellSols, states, sim_report] = simulateScheduleAD(initState, model, tmpschedule,'NonLinearSolver',nonlinearsolver);
    profile off;
    profile viewer
    figure(),plotCellData(Gt,states{end}.s(:,1))
    %}
    %%
    %return
    %%
    clear CO2VEBlackOilTypeModelSens
    smodel = CO2VEBlackOilTypeModelSens(Gt, rock2D, fluid);
    initState.dz=zeros(Gt.cells.num,1);
    initState.rhofac=1;
    initState.permfac=1;
    initState.porofac=1;
    for i=1:numel(schedule.control)
        schedule.control(i).dz=zeros(Gt.cells.num,1);
        schedule.control(i).rhofac=1;
        schedule.control(i).permfac=1;
        schedule.control(i).porofac=1;
    end
    %
    [wellSols, states, sim_report] = simulateScheduleAD(initState, smodel, schedule);
     %figure(),myplotCellData(Gt,states{end}.s(:,1))
    %return
    %% Save variables in workspace:
    %save(datestr(clock,30));
    figure(1),clf,plotCellData(Gt,states{end}.s(:,1)),colorbar
    plumes = getLayer9CO2plumeOutlines();
    [plumes,topsurface, topfit, hCO2] = makeSurfaceData(plumes,Gt)
    states = addHightData(states,Gt,fluid);
    %%
    sim_year=cumsum(schedule.step.val)/year;
    myplotCellData =@(G,data) plotCellData(G,data,'EdgeColor','none');
    for i=1:numel(plumes)
        % hack
        plume_year = plumes{i}.year-1998;
        tstep=find(sim_year==plume_year);
        if(tstep<=numel(states))
        figure(i),clf
        subplot(1,3,1)           
        myplotCellData(Gt,plumes{i}.h),colorbar;axis tight
        line(plumes{i}.outline(:,1), plumes{i}.outline(:,2),topsurface(plumes{i}.outline)-1, 'LineWidth',3, 'Color','r')
        subplot(1,3,2)
        myplotCellData(Gt,states{tstep}.h),colorbar,axis tight
        subplot(1,3,3)
        myplotCellData(Gt,states{tstep}.h-plumes{i}.h),colorbar,axis tight
        end
    end
    %%
    ny=1;%numel(plumes)
    figure(91),clf,myplotCellData(Gt,double((topfit(Gt.cells.centroids)-topsurface(Gt.cells.centroids))>0))
    [isin,sang]=insidePolygon(plumes{ny}.outline,Gt.cells.centroids);
    figure(92),clf,myplotCellData(Gt,sang),hold on
    plot3(plumes{ny}.outline(:,1), plumes{ny}.outline(:,2),topsurface(plumes{ny}.outline)-100,'*-','LineWidth',3, 'Color','g')
    %%
    newplumes=cell(numel(states),1);
    for i=1:numel(plumes)
        %tstep = plumes{i}.year-1998;
        plume_year = plumes{i}.year-1998;
        tstep=find(sim_year==plume_year);
        if(tstep<=numel(states))
            %newplumes{tstep}=struct('h',states{tstep}.s(:,2).*Gt.cells.H./(1-fluid.res_water));
            %newplumes{tstep}.h=newplumes{tstep}.h+rand(G.cells.num,1).*(newplumes{tstep}.h>0.5)
            %newplumes{tstep}.h=newplumes{tstep}.h+10*(newplumes{tstep}.h>0.5)
            newplumes{tstep}=plumes{i};
        end
        %newplumes{tstep}=plumes{i};      
    end
    a=matchToData(model, wellSols, states, schedule, newplumes)
    %%
    %cc=insidePolygon(line_coord,Gt.cells.centroids);
    %plotCellData(Gt,double(cc))
    %{
    obj_fun = @(wellSols,states,schedule,varargin) matchToData(model, wellSols, states, schedule, newplumes, varargin{:});
    objh = @(tstep) obj_fun(wellSols, states, schedule, 'ComputePartials', true, 'tStep', tstep);
    obj_fun2 = @(wellSols,states) matchToData(model, wellSols, states, schedule, newplumes);
    g    = computeGradientAdjointAD(initState, states, model, schedule, objh);
%}
    %gn    = computeGradientPerturbationAD(initState, model, schedule, obj_fun2);
    %%
    obj_funs = @(wellSols,states,schedule,varargin) matchToDataSens(smodel, wellSols, states, schedule, newplumes, varargin{:});
    objhs = @(tstep) obj_funs(wellSols, states, schedule, 'ComputePartials', true, 'tStep', tstep);
    %obj_fun2 = @(wellSols,states) matchToDatas(smodel, wellSols, states, schedule, newplumes);
    %gs   = computeGradientAdjointAD(initState, states, smodel, schedule, objhs,'ControlVariables', {'well'});
    %gsc   = computeGradientAdjointAD(initState, states, smodel, schedule, objhs,'ControlVariables', {'scell'});
    %%
     gsc   = computeGradientAdjointAD(initState, states, smodel, schedule, objhs,'ControlVariables', {'well','scell','mult'});
    %%
    dobj=cell(size(gsc,1),1);
    dobj{1}=zeros(1,1);
    dobj{2}=zeros(Gt.cells.num,1);
    dobj{3}=zeros(3,1);
    for i=1:size(gsc,2)
       for j=1:size(gsc,1);
        dobj{j}=dobj{j}+gsc{j,i};
       end
    end
    dobj_dz=dobj{2};
    figure(33),clf,myplotCellData(Gt,dobj_dz),colorbar
     ny=numel(plumes);
     line(plumes{ny}.outline(:,1), plumes{ny}.outline(:,2),topsurface(plumes{ny}.outline)-3,'LineWidth',3, 'Color','r')