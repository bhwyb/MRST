function [problem, state] = equationsWGVEbasicSens(model, state0, state, dt, drivingForces, varargin)

   opt = struct('reverseMode', false, ...
                'resOnly', false, ...
                'iteration', -1);

   opt = merge_options(opt, varargin{:});

   assert(isempty(drivingForces.src)); % unsupported
   W  = drivingForces.W;
   s  = model.operators;
   f  = model.fluid;

   % Extract the current and previous values of all variables to solve for
   [p, sG, sGmax, wellSol,dz,rhofac,permfac,porofac] = model.getProps(state , 'pressure', 'sg', 'sGmax', 'wellsol', 'dz','rhofac','permfac','porofac');
   [p0, sG0]               = model.getProps(state0, 'pressure', 'sg');

   % Stack well-related variables of the same type together
   bhp = vertcat(wellSol.bhp);
   qWs = vertcat(wellSol.qWs);
   qGs = vertcat(wellSol.qGs);

   %% Initialization of independent variables

   if ~opt.resOnly
      % ADI variables needed since we are not only computing residuals
      if ~opt.reverseMode
         [p, sG, bhp, qWs, qGs, dz, rhofac, permfac, porofac] = initVariablesADI(p, sG, bhp, qWs, qGs, dz,rhofac,permfac,porofac);
      else
         zw = zeros(size(bhp)); % dummy
         dzw = zeros(size(dz));
         dew = zeros(size(rhofac));
         [p0, sG0, ~, ~, ~, ~,~,~,~] = initVariablesADI(p0, sG0, zw, zw, zw, dzw,dew,dew,dew);
      end
   end

   sW  = 1 - sG;  % for ease of reading, we define an explicit variable
   sW0 = 1 - sG0; % also for water saturation

   %% Preparing various necessary, intermediate values

   % multiplier for mobilities
   [pvMult, transMult, mobMult, pvMult0] = getMultipliers(f, p, p0);
   
   %
   pvMult = porofac*pvMult;
   pvMult0 = state0.porofac*pvMult0;
   %transMult = permfac*transMult;
   %transMult0 = state0.permfac*transMult0;
   %
   
   % relative permeability
   [krW, krG] = model.evaluteRelPerm({sW, sG}, p, 'sGmax', sGmax);
   krW = krW * mobMult;
   krG = krG * mobMult;

   % Transmissibility
   trans = s.T * transMult;

   % Gravity gradient per face, including the gravity component resulting
   % from caprock geometry
   %gdz = model.getGravityGradient();
   s  = model.operators;
   Gt = model.G;
   g  = model.gravity(3); % @@ requires theta=0
   gdz = g * s.Grad(Gt.cells.z +dz);
   
   % CO2 phase pressure
   pW  = p;
   pW0 = p0;
   pG  = p + f.pcWG(sG, p, 'sGmax', sGmax);

   % Evaluate water and CO2 properties
   [vW, bW, mobW, rhoW, upcw] = ...
       getPhaseFluxAndProps_WGVEsens(model, pW, pG, krW, trans, gdz, 'W', 0, 0,rhofac,permfac,porofac);
   [vG, bG, mobG, rhoG, upcg] = ...
       getPhaseFluxAndProps_WGVEsens(model, pW, pG, krG, trans, gdz, 'G', 0, 0,rhofac,permfac,porofac);
  
   
   bW0 = f.bW(pW0);
   bG0 = f.bG(pW0); % Yes, using water pressure also for gas here

   % Multiply upstream b-factors by interface fluxes to obtain fluxes at
   % standard conditions
   bWvW = s.faceUpstr(upcw, bW) .* vW;
   bGvG = s.faceUpstr(upcg, bG) .* vG;


   %% Setting up brine and CO2 equations

   % Water (Brine)
   eqs{1} = (s.pv / dt) .* (pvMult .* bW .* sW - pvMult0 .* bW0 .* sW0) + permfac*s.Div(bWvW);

   % Gas (CO2)
   eqs{2} = (s.pv / dt) .* (pvMult .* bG .* sG - pvMult0 .* bG0 .* sG0) + permfac*s.Div(bGvG);

   % Include influence of boundary conditions
   eqs = addFluxesFromSourcesAndBC(model, ...
           eqs, {pW, pG}, {rhoW, rhoG}, {mobW, mobG}, {bW, bG}, {sW, sG}, drivingForces);


   %% Setting up well equations
   if ~isempty(W)
      wm = model.wellmodel;
      if ~opt.reverseMode
         wc = vertcat(W.cells);
         [cqs, weqs, ctrleqs, wc, state.wellSol] =                       ...
             wm.computeWellFlux(model, W, wellSol, bhp                 , ...
                                {qWs, qGs}                             , ...
                                p(wc)                                  , ...
                                [model.fluid.rhoWS, model.fluid.rhoGS] , ...
                                {bW(wc), bG(wc)}                       , ...
                                {mobW(wc), mobG(wc)}                   , ...
                                {sW(wc), sG(wc)}                       , ...
                                {}                                     , ...
                                'allowControlSwitching', false         , ...
                                'nonlinearIteration', opt.iteration);
         % Store the separate well equations (relate well bottom hole
         % pressures to influx)
         eqs(3:4) = weqs;

         % Store the control equations (ensuring that each well has values
         % corresponding to the prescribed value)
         eqs{5} = ctrleqs;

         % Add source term to equations.  Negative sign may be surprising if
         % one is used to source terms on the right hand side, but this is
         % the equations on residual form
         eqs{1}(wc) = eqs{1}(wc) - cqs{1};
         eqs{2}(wc) = eqs{2}(wc) - cqs{2};

      else
         [eqs(3:5), names(3:5), types(3:5)] = ...
             wm.createReverseModeWellEquations(model, state0.wellSol, p0);%#ok
      end
   else
      eqs(3:5) = {bhp, bhp, bhp}; % empty ADIs
   end
   if(~opt.reverseMode)
    eqs{6}=dz-drivingForces.dz;
    eqs{7}=rhofac-drivingForces.rhofac;
    eqs{8}=permfac-drivingForces.permfac;
    eqs{9}=porofac-drivingForces.porofac;
   else
     eqs{6} = double2ADI(zeros(numel(dz),1), p0);
     eqs{7} = double2ADI(zeros(numel(rhofac),1), p0);
     eqs{8} = double2ADI(zeros(numel(rhofac),1), p0);
     eqs{9} = double2ADI(zeros(numel(rhofac),1), p0);     
   end
   %% Setting up problem
   primaryVars = {'pressure' , 'sG'   , 'bhp'        , 'qWs'      , 'qGs', 'dz', 'rhofac','permfac','porofac'};
   types = {'cell'           , 'cell' , 'perf'       , 'perf'     , 'well', 'scell','mult','mult','mult'};
   names = {'water'          , 'gas'  , 'waterWells' , 'gasWells' , 'closureWells', 'geometry','mrho','mperm','mporo'};
   if isempty(W)
      % Remove names/types associated with wells, as no well exist
      types = types(1:2);
      names = names(1:2);
   end

   problem = LinearizedProblem(eqs, types, names, primaryVars, state, dt);

end
