function [optim, init, history] = optimizeRates(initState, model, schedule, ...
                                                min_rates, max_rates, varargin)
   
% optim.obj_val_steps
% optim.obj_val_total
% optim.schedule
% optim.wellSols
% optim.states
   
   moduleCheck('ad-core', 'ad-props', 'optimization');

   opt.obj_scaling = [];  % Compute explicitly if not provided from outside
   opt.leak_penalty = 10; % Only used if 'obj_fun' not provided 
   opt.obj_fun = @(G, wellSols, states, schedule) ...
                  leak_penalizer(G, wellSols, states, schedule, opt.leak_penalty);
   
   opt = merge_options(opt, varargin{:});
   
   num_wells = num(schedule.control(1).W);
   assert(numel(max_rates) == num_wells);
   assert(numel(min_rates) == num_wells);
   
   init.schedule = schedule;

   %% Compute initial objective value (if required for scaling)
   if isempty(opt.obj_scaling)
      [init.wellSols, init.states] = ...
          simulateScheduleAD(initState, model, schedule); 
      
      
      init.obj_val_steps = cell2mat(obj.opt_fun(model.G, ...
                                                init.wellSols, ...
                                                init.states, ...
                                                opt.schedule));
      init.obj_val_total = sum(init.obj_val_steps);
      
      % Use value of objective function before optimization as scaling.
      opt.obj_scaling = init.obj_val_total;
   end
   
   
   %% Define limits, scaling and objective function
   
   scaling.box_lims = [min_rates(:), max_rates(:)];
   scaling.obj      = opt.obj_scaling;
   
   obj_evaluator = @(u) eval_objective(u, opt.obj_fun, model, initState, schedule);
   
   %% Call optimization routine
   
   [opt_val, u_opt, history] = unitBoxBFGS(schedule2control(schedule , ...
                                                            scaling) , ...
                                           obj_evaluator);
   
   %% Preparing solution structures
   
   optim.schedule = control2Schedule(u_opt, schedule, scaling);

   [optim.wellSols, optim.states] = simulateScheduleAD(initState, ...
                                                       model, ...
                                                       optim.schedule);
   
   optim.obj_val_steps = obj(optim.wellSols, optim.states);
   optim.obj_val_total = cumsum(cell2mat(optim.obj_val_steps));
   
end

% ----------------------------------------------------------------------------

function [val, der, wellSols, states] = obj_evaluator(u, obj_fun, model, ...
                                                     initState, schedule) 
   
   minu = min(u);
   maxu = max(u);
   if or(minu < -eps , maxu > 1+eps)
      warning('Controls are expected to lie in [0 1]\n')
   end

   boxLims = scaling.boxLims;
   if isfield(scaling, 'obj')
      objScaling = scaling.obj;
   else
      objScaling = 1;
   end
   
   % update schedule:
   schedule = control2schedule(u, schedule, scaling);
   
   % run simulation:
   [wellSols, states] = simulateScheduleAD(initState, model, schedule);
   
   % compute objective:
   vals = obj_fun(model.G, wellSols, states, schedule);
   val  = sum(cell2mat(vals))/objScaling;

   % run adjoint:
   if nargout > 1
      objh = @(tstep)obj_fun(model.G, wellSols, states, schedule, 'ComputePartials', true, 'tStep', tstep);
      g    = computeGradientAdjointAD(initState, states, model, schedule, objh);
      % scale gradient:
      der = scaleGradient(g, schedule, boxLims, objScaling);
      der = vertcat(der{:});
   end
end

% ----------------------------------------------------------------------------

function grd = scaleGradient(grd, schedule, boxLims, objScaling)
dBox   = boxLims(:,2) - boxLims(:,1);
for k = 1:numel(schedule.control)
    grd{k} = (dBox/objScaling).*grd{k};
end
end
    

   
   
end
