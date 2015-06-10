function exploreSimulation(varargin)
   
   gravity on;
   moduleCheck('ad-core');

   rhoCref = 760 * kilogram / meter ^3; % an (arbitrary) reference density
   
   opt.grid_coarsening = 4;
   opt.default_formation = 'Utsirafm';
   opt.window_size = [1200 900];
   opt.seafloor_depth = 100 * meter;
   opt.seafloor_temp  =  7; % in Celsius
   opt.temp_gradient  = 35.6; % degrees per kilometer
   opt.water_density  = 1000; % kg per m3
   opt.press_deviation = 0; % pressure devation from hydrostatic (percent)
   opt.res_sat_co2 = 0.21; 
   opt.res_sat_wat = 0.11;
   opt.dis_max = (53 * kilogram / meter^3) / rhoCref; % value from CO2store
   opt.max_num_wells = 10;
   opt.default_rate = 1 * mega * 1e3 / year / rhoCref; % default injection rate
   opt.max_rate = 10 * mega * 1e3 / year / rhoCref; % maximum allowed injection rate
   opt.seafloor_depth = 100 * meter;
   opt.seafloor_temp  =  7; % in Celsius
   opt.temp_gradient  = 35.6; % degrees per kilometer
   opt.water_compr_val = 4.3e-5/barsa;
   opt.water_compr_p_ref = 100 * barsa;
   opt.water_residual = 0.11;
   opt.co2_residual = 0.21;
   opt.inj_time = 50 * year;
   opt.inj_steps = 10;
   opt.mig_time = 3000 * year;
   opt.mig_steps = 30;
   opt.well_radius = 0.3;
   opt.subtrap_file = 'utsira_subtrap_function_3.mat';
   
   opt = merge_options(opt, varargin{:});

   var.Gt                = []; % to be set by the call to 'set_formation'
   var.rock2D            = []; % rock properties of current formation
   var.ta                = []; % trapping analysis of current formation
   var.current_formation = ''; % name of formation currently loaded
   var.data              = []; % current face data to plot
   var.loops             = []; % one or more loops of boundary faces
   var.loops_bc          = []; % bc for all faces (0: closed, 1: semi-open, 2: open)
   var.co2               = CO2props('sharp_phase_boundary', false, 'rhofile', 'rho_demo');
   var.wells             = reset_wells(opt.max_num_wells);
   % Temporary variables used by different functions
   temps = [];
   
   
   
   set_formation(opt.default_formation, false);

   %% Setting up interactive interface
   
   var.h = figure();%'KeyPressFcn', @(obj, e) parse_keypress(e.Key));
   set_size(var.h, opt.window_size(1), opt.window_size(2));
   
   % Graphical window
   var.ax = axes('parent', var.h, 'position', [0.05, 0.12, 0.5, 0.87]);

   % Formation selection
   names = formation_names();
   fsel = uicontrol('parent', var.h, ...
                    'style', 'popup', ...
                    'units', 'normalized', ...
                    'position', [0.05, 0.0 0.3 0.05], ...
                    'string', listify(names), ...
                    'value', formation_number(var.current_formation));
   set (fsel, 'Callback', @(es, ed) set_formation(names{get(es, 'Value')}, ...
                                                  true));
   
   % Group for setting boundaries
   bc_group = setup_bc_group([.56 .8 .16 .15]);

   % Group for setting interaction mode
   imode_group = setup_imode_group([.8 .8 .16 .15]);
   
   % Group for displaying and modifying wells
   [well_group, well_entries] = setup_well_group([.56 .25, .4, .50]);%#ok

   % launch button
   launch_button = uicontrol('parent', var.h, ...
                             'style', 'pushbutton', ...
                             'units', 'normalized', ...
                             'position', [.56 .12 .18 .11], ...
                             'string', 'Launch new simulation!', ...
                             'callback', @(varargin) launch_simulation()); %#ok
                       
   
   % Other options
   [opt_group, opt_choices] = setup_opt_group([.76, .12, .20, .11]);%#ok
   
   %% Launching by calling redraw function
   redraw();
   
   
   % ============================= LOCAL HELPER FUNCTIONS =============================

   function launch_simulation()

      if isempty(var.wells(1).pos)
         % no wells present
         msgbox('Add at least one well before running the simulation', ...
                      'No wells present', 'error', 'modal');
         return;
      end
      
      
      use_dissolution = logical(get(opt_choices.dissolution, 'value'));
      use_trapping    = logical(get(opt_choices.subscale,    'value'));
      
      dh = [];
      topo = 'smooth';
      if use_trapping
         dh = computeToptraps(load(opt.subtrap_file), var.Gt, true);
         topo = 'inf_rough';
      end
      
      % Set up input parameters
      fluid     = makeVEFluid(var.Gt, var.rock2D, 'sharp interface', ...
                              'fixedT'      , caprock_temperature()                          , ...
                              'wat_rho_pvt' , [opt.water_compr_val  , opt.water_compr_p_ref] , ...                              
                              'residual'    , [opt.water_residual   , opt.co2_residual]      , ...
                              'dissolution' , use_dissolution                                , ...
                              'dis_max'     , opt.dis_max                                    , ...
                              'surf_topo'   , topo                                           , ...
                              'top_trap'    , dh);
                              
      model     = CO2VEBlackOilTypeModel(var.Gt, var.rock2D, fluid);
      initState = setup_initstate();
      schedule  = setup_schedule();
  
      % spawn simulation window 
      visualSimulation(initState, model, schedule, 'rhoCref', rhoCref, 'trapstruct', var.ta, 'dh', dh);

   end

   % ----------------------------------------------------------------------------
   
   function schedule = setup_schedule()

      % Create wells 
      W = [];
      for i = 1:opt.max_num_wells

         if ~isempty(var.wells(i).pos)
            wcell_ix = closest_cell(var.Gt, [var.wells(i).pos,0], 1:var.Gt.cells.num);
            W = addWell(W, var.Gt, var.rock2D, wcell_ix, ...
                        'type', 'rate', ...
                        'val', var.wells(i).rate, ...
                        'radius', opt.well_radius, ...
                        'comp_i', [0 1], ...
                        'name', ['I', num2str(i)]);
         end
      end
      W_shut = W;
      for i = 1:numel(W_shut)
         W_shut(i).val = 0;
      end
      
      
      schedule.control(1).W = W;
      schedule.control(2).W = W_shut;
      
      % Define boundary conditionsx
      open_faces = [];
      for i = 1:numel(var.loops)
         faces = var.loops{i};
         bcs   = var.loops_bc{i};
         open_faces = [open_faces; faces(bcs>0)];%#ok
      end
      schedule.control(1).bc = addBC([], open_faces, ...
                                     'pressure', ...
                                     var.Gt.faces.z(open_faces) * opt.water_density * norm(gravity), ...
                                     'sat', [1 0]);
      schedule.control(2).bc = schedule.control(1).bc;
      
      dTi = opt.inj_time / opt.inj_steps;
      dTm = opt.mig_time / opt.mig_steps;
      istepvec = ones(opt.inj_steps, 1) * dTi;
      mstepvec = ones(opt.mig_steps, 1) * dTm;
      
      schedule.step.val = [istepvec; mstepvec];
      schedule.step.control = [ones(opt.inj_steps, 1); ones(opt.mig_steps, 1) * 2];
      
   end

   % ----------------------------------------------------------------------------
   
   function T = caprock_temperature()
      % Return temperature in Kelvin
      T = 273.15 + ...
          opt.seafloor_temp + ...
          (var.Gt.cells.z - opt.seafloor_depth) / 1e3 * opt.temp_gradient;
   end
   
   % ----------------------------------------------------------------------------
   
   function state = setup_initstate()
   
      state.pressure = var.Gt.cells.z * norm(gravity) * opt.water_density;
      state.s = repmat([1 0], var.Gt.cells.num, 1);
      state.sGmax = state.s(:,2);
      
      % If dissolution is activated, we need to add a field for that too
      if logical(get(opt_choices.dissolution, 'value'))
         state.rs = 0 * state.sGmax;
      end
      
   end
      
   % ----------------------------------------------------------------------------
   
   function wells = reset_wells(num)
      wells = repmat(struct('pos', [], 'rate', 0), num, 1);
   end
      
   % ----------------------------------------------------------------------------
   
   function res = get_interaction_type()
      res = get(get(imode_group, 'selectedobject'), 'string');
   end
   
   % ----------------------------------------------------------------------------
   
   function res = get_active_bc_type()
      switch(get(get(bc_group, 'selectedobject'), 'string'))
        case 'Closed'
          res = 0;
        case 'Semi-open'
          res = 1;
        case 'Open'
          res = 2;
        otherwise
          error('missing case');
      end
   end
   
   % ----------------------------------------------------------------------------
   
   function set_uniform_bc_callback(varargin)
      bc_type = get_active_bc_type();
      for l = 1:numel(var.loops_bc)
         var.loops_bc{l} = var.loops_bc{l} * 0 + bc_type;
      end
      redraw();
   end
   
   % ----------------------------------------------------------------------------
   
   function set_rotate_state_callback()
      sel = get(get(imode_group, 'selectedobject'), 'string');
      if strcmpi(sel, 'Rotate model')
         rotate3d(var.ax, 'on'); 
      else
         rotate3d(var.ax, 'off');
      end
   end
   
   % ----------------------------------------------------------------------------
   
   function [group, choices] =  setup_opt_group(pos)
      
      % Create group
      group = uipanel('Visible', 'off',...
                      'units', 'normalized', ...
                      'position', pos);
      
      % create widgets
      choices.dissolution = uicontrol('parent', group, ...
                                      'style', 'checkbox', ...
                                      'units', 'normalized', ...
                                      'position' , [.1, .30, .2, .2]);

      label = uicontrol('parent', group, ...
                        'style', 'text', ...
                        'units', 'normalized', ...
                        'horizontalalignment', 'left', ...
                        'position', [.2, .28, .55, .2], ...
                        'string', 'Include dissolution');%#ok

      choices.subscale = uicontrol('parent', group, ...
                                      'style', 'checkbox', ...
                                      'units', 'normalized', ...
                                      'position' , [.1, .60, .2, .2]);

      label2 = uicontrol('parent', group, ...
                         'style', 'text', ...
                         'units', 'normalized', ...
                         'horizontalalignment', 'left', ...
                         'position', [.2, .58, .80, .2], ...
                         'string', 'Include subscale trapping');%#ok
      
      set(group, 'visible', 'on');      
   end
   
   % ----------------------------------------------------------------------------
   
   function group = setup_imode_group(pos)
      
      % Create group
      group = uibuttongroup('Visible', 'off',...
                             'units', 'normalized', ...
                             'position', pos, ...
                             'selectionchangefcn', @(varargin) set_rotate_state_callback());
      % Create radiobuttons
      b1 = uicontrol(group, 'style', 'radiobutton', ...
                            'string', 'Edit boundaries', ...
                            'units', 'normalized', ...
                            'position', [.1 .1 .9 .3]);%#ok
      b2 = uicontrol(group, 'style', 'radiobutton', ...
                            'string', 'Select wellsites', ...
                            'units', 'normalized', ...
                            'position', [.1 .38 .9 .3]);%#ok
      b3 = uicontrol(group, 'style', 'radiobutton', ...
                            'string', 'Rotate model', ...
                            'units', 'normalized', ...
                            'position', [.1 .66 .9 .3]);%#ok
                            
      set(group, 'visible', 'on');      
   end
   
   % ----------------------------------------------------------------------------

   function [group, entries] = setup_well_group(pos)
      % Create group
      group = uipanel('Visible', 'off',...
                      'units', 'normalized', ...
                      'position', pos);
      
      entries = [];
      
      for i = 1:opt.max_num_wells
         ypos    = ((opt.max_num_wells - (i)) / opt.max_num_wells) * 0.95;
         yheight = (1 / opt.max_num_wells) * 0.95;
         entries = [entries; add_well_entry(group, [0.1, ypos, 0.9, yheight], i)];%#ok
      end
      set(group, 'visible', 'on');
   end
   
   % ----------------------------------------------------------------------------
   
   function we = add_well_entry(group, pos, index)
      we.name = uicontrol('parent', group, ...
                          'style', 'edit',...
                          'string', sprintf('Well %i:', index), ...
                          'horizontalalignment', 'left', ...
                          'units', 'normalized', ...
                          'enable', 'inactive', ...
                          'fontsize', 8, ...
                          'handlevisibility', 'off', ...
                          'position', [pos(1), pos(2), pos(3)*0.1, pos(4)]);

      we.status = uicontrol('parent', group, ...
                            'style', 'edit',... % 'edit' rather than 'text' for vertical alignment
                            'string', '<none>', ...
                            'enable', 'inactive', ...
                            'horizontalalignment', 'center', ...
                            'units', 'normalized', ...
                            'fontsize', 8, ...
                            'handlevisibility', 'off', ...
                            'position', [pos(1) + pos(3)*0.1, pos(2), pos(3)*0.3, pos(4)]);
      
      we.delete = uicontrol('parent', group, ...
                            'style', 'pushbutton', ...
                            'string', 'X', ...
                            'units', 'normalized', ...
                            'position', [pos(1) + pos(3)*0.4, pos(2), pos(3)*0.1, pos(4)], ...
                            'handlevisibility', 'off', ...
                            'callback', @(varargin) clear_well_callback(index));
      we.rate = uicontrol('parent', group, ...
                          'style', 'slider', ...
                          'units','normalized', ...
                          'position', [pos(1) + pos(3)*0.51, pos(2), pos(3) * 0.35, pos(4)], ...
                          'value', opt.default_rate, ...
                          'min', 0, ...
                          'max', opt.max_rate, ...
                          'callback', @(varargin) set_new_rate_callback(index));
      we.rate_view = uicontrol('parent', group, ...
                            'style', 'edit',... % 'edit' rather than 'text' for vertical alignment
                            'string', sprintf('%3.1f Mt', opt.default_rate * year * rhoCref/1e9), ...
                            'enable', 'inactive', ...
                            'horizontalalignment', 'left', ...
                            'units', 'normalized', ...
                            'fontsize', 8, ...
                            'handlevisibility', 'off', ...
                            'position', [pos(1) + pos(3)*0.87, pos(2), pos(3)*0.11, pos(4)]);
   end

   % ----------------------------------------------------------------------------
   
   function set_new_rate_callback(ix)
      if isempty(var.wells(ix).pos)
         % well is inactive, keep rate to default value
         set(well_entries(ix).rate, 'value', opt.default_rate);
      else
         var.wells(ix).rate = get(well_entries(ix).rate, 'value');
      end      
      redraw();
   end
   
   % ----------------------------------------------------------------------------
   
   function clear_well_callback(ix)
      
      % shifting remaining wells up
      for i = ix:(numel(var.wells)-1)
         var.wells(i) = var.wells(i+1);
      end
      var.wells(end) = struct('pos', [], 'rate', 0);

      % redraw with new well information
      redraw();
   end
   
   % ----------------------------------------------------------------------------
   
   function group = setup_bc_group(pos)

      % create radiobutton group
      group = uibuttongroup('Visible', 'off',...
                            'units', 'normalized', ...
                            'position', pos);
      % create radiobuttons
      b1 = uicontrol(group, 'style', 'radiobutton', ...
                            'string', 'Closed', ...
                            'units', 'normalized', ...
                            'position', [.1 .1 .9 .27], ...
                            'HandleVisibility', 'off');%#ok
      b2 = uicontrol(group, 'style', 'radiobutton', ... 
                            'string', 'Semi-open', ...
                            'units', 'normalized', ...
                            'position', [.1 .4 .9 .27], ...
                            'HandleVisibility', 'off');%#ok
      b3 = uicontrol(group, 'style', 'radiobutton', ...
                            'string', 'Open', ...
                            'units', 'normalized', ...
                            'position', [.1 .7 .9 .27], ...
                            'HandleVisibility', 'off');%#ok
      pb1 = uicontrol(group, 'style', 'pushbutton', ...
                             'string', 'Set all', ...
                             'units', 'normalized', ...
                             'position', [.6 .7 .35 .2], ...
                             'handlevisibility', 'off', ...
                             'callback', @set_uniform_bc_callback);%#ok
      
      set(group, 'visible', 'on');
   end
   
   % ----------------------------------------------------------------------------
   
   function click_handler(varargin)

      pt = get(gca,'CurrentPoint'); pt = pt(end,:); 
      fn = [];
      
      switch get_interaction_type()
        case 'Edit boundaries'
          if ~isfield(temps, 'bc_segment_start')
             [temps.bc_segment_start, temps.bc_segment_loop_ix] = ...
                 closest_bface(var.Gt, pt, var.loops);
             fn = @() plot(pt(1), pt(2), '*r');
          else
             bc_segment_end = closest_bface(var.Gt, pt, var.loops, temps.bc_segment_loop_ix);
             
             cur_loop = var.loops{temps.bc_segment_loop_ix};
             ix1 = find(cur_loop == temps.bc_segment_start, 1);
             ix2 = find(cur_loop ==  bc_segment_end, 1);
             if ix1 > ix2 % ensure ix2 > ix1
                tmp = ix1;
                ix1 = ix2;
                ix2 = tmp;
             end
             if ix2-ix1 < numel(cur_loop)/2;
                seq = ix1:ix2;
             else
                seq = [ix2:numel(cur_loop), 1:ix1];
             end

             var.loops_bc{temps.bc_segment_loop_ix}(seq) = get_active_bc_type();
             
             temps = rmfield(temps, 'bc_segment_start');
             temps = rmfield(temps, 'bc_segment_loop_ix');
          end
        case 'Select wellsites'

          % find first empty slot
          for i = 1:numel(var.wells)
             if isempty(var.wells(i).pos)
                break;
             end
          end
          full = (i==numel(var.wells) && ~isempty(var.wells(i).pos));
          if full
             % Discard oldest well, shift others downwards
             var.wells(1:end-1) = var.wells(2:end);
          end
          var.wells(i).pos = pt(1:2);
          var.wells(i).rate = opt.default_rate;
        case 'Rotate model'
          % do nothing - the radio button itself has toggled on rotate when activated
          % rotate3d(var.ax, 'on');
        otherwise
          disp('unimplemented');
          return;
      end
      redraw(fn);
      
      
      % %disp(pts);
      % ix = closest_cell(var.Gt, pts(end,:), 1:var.Gt.cells.num);
      % var.data(ix) = 1;
      % redraw();
   end

   % ----------------------------------------------------------------------------
   
   function redraw(post_fnx)
      axes(var.ax); cla;
      axis auto;
      cla;
      
      % Draw current field
      plotCellData(var.Gt, var.data, 'buttondownfcn', @click_handler);
      
      % Draw boundary conditions
      for i = 1:numel(var.loops)
         loop = var.loops{i};
         loop_bc = var.loops_bc{i};
         plotFaces(var.Gt, loop(loop_bc==0), 'edgecolor', 'r', 'linewidth', 4); % closed
         plotFaces(var.Gt, loop(loop_bc==1), 'edgecolor', 'y', 'linewidth', 4); % closed
         plotFaces(var.Gt, loop(loop_bc==2), 'edgecolor', 'g', 'linewidth', 4); % closed
      end
      
      % Draw wells
      for i = 1:opt.max_num_wells
         w = var.wells(i);
         if ~isempty(w.pos)
            hold on;
            lon = w.pos(1);
            lat = w.pos(2);
            wellcell = closest_cell(var.Gt, [w.pos,0], 1:var.Gt.cells.num);
            plotWell(var.Gt.parent, ...
                     addWell([], var.Gt.parent, var.rock2D, wellcell, 'name', sprintf('W%i', i)), ...
                     'color', 'k', 'fontsize', 20);
            plot3(lon, lat, var.Gt.cells.z(wellcell)*0.98, 'ro', 'markersize', 8, ...
                  'MarkerFaceColor',[0 0 0]);
            set(well_entries(i).status, 'string', sprintf('(%4.2e, %4.2e)', lon, lat));
         else
            set(well_entries(i).status, 'string', '<none>');
         end
         annual_rate = var.wells(i).rate * year * rhoCref/1e9;
         set(well_entries(i).rate_view, 'string', sprintf('%3.1f Mt', annual_rate));
         set(well_entries(i).rate, 'value', var.wells(i).rate);
      end
      
      % Call optional function to complete redraw process
      if nargin>0 && ~isempty(post_fnx)
         hold on;
         post_fnx();
      end
      
      view(0, 90);
   end

   % ----------------------------------------------------------------------------
   
   function set_formation(name, do_redraw)
   
      % Default values, in case values are lacking in model file.
      default_perm = 200 * milli * darcy;
      default_poro = 0.2;
      
      var.current_formation = name;
      
      % Load grid and rock, and assure rock values are valid
      [var.Gt, var.rock2D] = getFormationTopGrid(name, opt.grid_coarsening);
      
      if any(isnan(var.rock2D.poro))
         warning('Replacing missing porosity value with default value.');
         var.rock2D.poro = default_poro * ones(size(var.rock2D.poro));
      end
      if any(isnan(var.rock2D.perm))
         warning('Replacing missing permeability value with default value.');
         var.rock2D.perm = default_perm * ones(size(var.rock2D.perm));
      end
      
      % Run trapping analysis (we need this information to compute
      % inventories)
      var.ta = trapAnalysis(var.Gt,true); %@@ false
      
      var.data = var.Gt.cells.z; 
      var.loops = find_boundary_loops(var.Gt);
      % Setting all boundary conditions to open (2)
      var.loops_bc = cellfun(@(x) 0 * x + 2, var.loops, 'uniformoutput', false);

      var.wells = reset_wells(opt.max_num_wells);
      temps = []; % reset all temporary variables
      
      % Call 'redraw' if requested
      if do_redraw
         redraw();
      end
      
   end
   
end

% ======================= INDEPENDENT HELPER FUNCTIONS =======================

function num = formation_number(name)
   num = find(cellfun(@(x) strcmpi(x, name), formation_names()), 1);
end

% ----------------------------------------------------------------------------

function str = listify(names)

   str = cellfun(@(x) [x,'|'], names, 'uniformoutput', false);
   str = [str{:}];
   str = str(1:end-1);
end

% ----------------------------------------------------------------------------

function names = formation_names()

   names = {'Brentgrp', 'Brynefm', 'Fensfjordfm', 'Gassumfm', 'Huginfmeast', ...
            'Huginfmwest', 'Johansenfm', 'Krossfjordfm', 'Pliocenesand', ...
            'Sandnesfm', 'Skadefm', 'Sleipnerfm', 'Sognefjordfm', 'Statfjordfm', ...
            'Ulafm', 'Utsirafm'};     
end

% ----------------------------------------------------------------------------

function h = set_size(h, res_x, res_y)
% Utility function to resize a graphical window
   
   pos = get(h, 'position');
   set(h, 'position', [pos(1:2), res_x, res_y]);
   
end

% ----------------------------------------------------------------------------

function ix = closest_cell(Gt, pt, candidates)
   
   d = bsxfun(@minus, [Gt.cells.centroids(candidates,:), Gt.cells.z(candidates)], pt);
   d = sum(d.^2, 2);
   [~, ix] = min(d);
   ix = candidates(ix);
end

% ----------------------------------------------------------------------------

function [cix, fix] = boundary_cells_and_faces(Gt)
   
   fix = find(prod(Gt.faces.neighbors, 2) ==0);
   cix = unique(sum(Gt.faces.neighbors(fix, :), 2));
end

% ----------------------------------------------------------------------------

function loops = find_boundary_loops(Gt)

   [~, fix] = boundary_cells_and_faces(Gt); % boundary face indices

   tmp = [fix, Gt.faces.nodes(Gt.faces.nodePos(fix));
          fix, Gt.faces.nodes(Gt.faces.nodePos(fix)+1)];
   tmp = sortrows(tmp, 2);
   tmp = reshape(tmp(:,1), 2, []); % columns now express face neighborships
   
   % defining connectivity matrix
   M = sparse(tmp(1,:), tmp(2,:), 1, Gt.faces.num, Gt.faces.num);
   M = spones(M+M');
   num_loops = 0;
   while nnz(M) > 0
      num_loops = num_loops + 1;
      [loop,~] = find(M, 1);
      next = find(M(loop, :), 1); % find one of the two neighbor faces
      while ~isempty(next)
         M(loop(end), next) = 0; %#ok
         M(next, loop(end)) = 0; %#ok
         loop = [loop; next]; %#ok
         next = find(M(next, :)); 
         assert(numel(next) <= 1);
      end
      assert(loop(1) == loop(end));
      loops{num_loops} = loop(1:end-1); %#ok
   end
end

% ----------------------------------------------------------------------------

function [face_ix, loop_ix] = closest_bface(Gt, pt, loops, imposed_loop)

   if (nargin > 3)
      loops_ixs = imposed_loop;
   else
      loops_ixs = 1:numel(loops);
   end
   
   closest_face = [];
   
   for l = loops_ixs
      % computing closest face for this loop
      loop = loops{l};
      loop_coords = [Gt.faces.centroids(loop,:), Gt.faces.z(loop,:)];
      dist = bsxfun(@minus, loop_coords, pt);
      dist = sum(dist.^2, 2);
      [dmin, num] = min(dist);
      closest_face = [closest_face; [dmin, num]]; %#ok
   end
   
   [~, i] = min(closest_face(:,1));
   loop_ix = loops_ixs(i);
   face_ix = loops{loop_ix}(closest_face(i, 2));
   
end
