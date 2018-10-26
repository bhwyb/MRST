function [problem, state] = transportEquationOilWaterDG(state0, state, model, dt, drivingForces, varargin)

    opt = struct('Verbose'      , mrstVerbose, ...
                 'reverseMode'  , false      ,...
                 'scaling'      , []         ,...
                 'resOnly'      , false      ,...
                 'history'      , []         ,...
                 'solveForWater', true       , ...
                 'solveForOil'  , false      , ...
                 'iteration'    , -1         , ...
                 'stepOptions'  , []         ); % Compatibility only
    opt      = merge_options(opt, varargin{:});
    
    % Frequently used properties
    op       = model.operators;
    fluid    = model.fluid;
    rock     = model.rock;
    G        = model.G;
    disc     = model.disc;
    flux2Vel = disc.velocityInterp.faceFlux2cellVelocity;
    
    % We may solve for both oil and water simultaneously
    solveAllPhases = opt.solveForWater && opt.solveForOil;
    
    % Prepare state for simulation-----------------------------------------
    if opt.iteration == 1 && ~opt.resOnly 
        if model.tryMaxDegree
            % If we are at the first iteration, we try to solve using
            % maximum degree in all cells
            state.degree(~G.cells.ghost) ...
                             = repmat(disc.degree, nnz(~G.cells.ghost), 1);
        end
        % For cells that previously had less than nDof unknowns, we must
        % map old dofs to new
        state = disc.mapDofs(state, state0);
        
    end
    % Update discretizaiton information. This is carried by the state
    % variable, and holds the number of dofs per cell + dof position in
    % state.sdof
    state0 = disc.updateDofPos(state0);
    state  = disc.updateDofPos(state);
    %----------------------------------------------------------------------
    
    % Properties from current and previous timestep------------------------
    [p , sWdof , sW , sOdof , sO , wellSol] = model.getProps(state , ...
                  'pressure', 'swdof', 'water', 'sodof', 'oil', 'wellsol');
    [p0, sWdof0, sW0, sOdof0, sO0,        ] = model.getProps(state0, ...
                  'pressure', 'swdof', 'water', 'sodof', 'oil'           );
    % If timestep has been split relative to pressure, linearly interpolate
    % in pressure.
    pFlow = p;
    if isfield(state, 'timestep')
        dt_frac = dt/state.timestep;
        p       = p.*dt_frac + p0.*(1-dt_frac);
    end
    %----------------------------------------------------------------------
    
    % Initialization of independent variables -----------------------------
    assert(~opt.reverseMode, 'Backwards solver not supported for splitting');
    if solveAllPhases
        if ~opt.resOnly
            [sWdof, sOdof] = model.AutoDiffBackend.initVariablesAD(sWdof, sOdof);
        end
        primaryVars = {'sWdof', 'sOdof'};
        sT = sO + sW;
        [krW, krO] = model.evaluateRelPerm({sW./sT, sO./sT});
    else
        if ~opt.resOnly
            sWdof = model.AutoDiffBackend.initVariablesAD(sWdof);
        end
        primaryVars = {'sWdof'};
        sOdof = 1;
        sO = 1 - sW;
        sT = ones(size(double(sW)));
        [krW, krO] = model.evaluateRelPerm({sW, sO});
    end
    %----------------------------------------------------------------------

%     % We will solve for water saturation dofs
%     primaryVars = {'sWdof'};
    nDof = state.nDof;

    % Get multipliers
    [pvMult, transMult, mobMult, pvMult0] = getMultipliers(model.fluid, p, p0);
    pvMult  = expandSingleValue(pvMult , G);
    pvMult0 = expandSingleValue(pvMult0, G);
    mobMult = expandSingleValue(mobMult, G);
    T       = op.T.*transMult;
    T_all   = model.operators.T_all;
    
    % Phase properties
    gdz = model.getGravityGradient();
    [vW, bW, mobW, rhoW, pW, upcW, dpW, muW] = getPropsWater_DG(model, p, T, gdz, mobMult);
    [vO, bO, mobO, rhoO, pO, upcO, dpO, muO] = getPropsOil_DG(model, p, T, gdz, mobMult);
    
    % Fractional flow function
    fW = @(sW, sO, cW, cO) mobW(sW, cW)./(mobW(sW, cW) + mobO(sO, cO));
    bW0 = fluid.bW(p0);
    
    % Gravity flux
    gp = op.Grad(p);
    [gW, gO] = deal(zeros(G.faces.num, 1));
    gW(op.internalConn) = gp - dpW;
    gO(op.internalConn) = gp - dpO;
    
    % Add gravity flux where we have BCs
    bc = drivingForces.bc;
    if ~isempty(bc)
        
        BCcells = sum(G.faces.neighbors(bc.face,:), 2);
        dz = G.cells.centroids(BCcells, :) - G.faces.centroids(bc.face,:);
        g = model.getGravityVector();
        rhoWBC = rhoW(BCcells);
        rhoOBC = rhoO(BCcells);
        
        gW(bc.face) = rhoWBC.*(dz*g');
        gO(bc.face) = rhoOBC.*(dz*g');
        
    end    
    
    TgW = T_all.*gW;
    TgO = T_all.*gO;
    TgWc = flux2Vel(TgW);
    TgOc = flux2Vel(TgO);
    TgW = TgW./G.faces.areas;
    TgO = TgO./G.faces.areas;
    
    % Viscous flux
    flux  = sum(state.flux,2);   % Total volumetric flux
    vT    = flux./G.faces.areas; % Total flux 
    vTc   = flux2Vel(flux);      % Map face fluxes to cell velocities
    
    % Cell integrand functions---------------------------------------------
        
    % Accumulation term
    acc = @(sW, sW0, c, psi) (pvMult(c).*rock.poro(c).*bW(c).*sW - ...
                              pvMult0(c).*rock.poro(c).*bW0(c).*sW0).*psi/dt;            
    % Convection term
    conv = @(sW,fW,c,grad_psi) bW(c).*fW.*(disc.dot(vTc(c,:),grad_psi) ...
                 + mobO(1-sW,c).*disc.dot(TgWc(c,:) - TgOc(c,:),grad_psi));
    % Cell integrand
    integrand = @(sW, sW0, fW, c, psi, grad_psi) ...
                          acc(sW, sW0, c, psi) - conv(sW, fW, c, grad_psi);
    % Integrate integrand*psi{dofNo} over all cells for dofNo = 1:nDof
    cellIntegral = disc.cellInt(model, integrand, [], state, state0, sWdof, sWdof0, fW);
  
    % Surface integrand functions------------------------------------------
    
    % Flux term
    integrand = @(sWv, sWg, fWv, fWg, c, cv, cg, f, psi) ...
        (bW(cv).*fWv.*vT(f) + bW(cg).*fWg.*mobO(1-sWg,cg).*(TgW(f) - TgO(f))).*psi;%./G.faces.areas(f);
    % % Integrate integrand*psi{dofNo} over all cells surfaces for dofNo = 1:nDof
    faceIntegral = disc.faceFluxInt(model, integrand, [], T, flux, {gW, gO}, {mobW, mobO}, state, sWdof, fW);
  
    % Water equation-------------------------------------------------------
    water = cellIntegral + faceIntegral;

    % Well contributions---------------------------------------------------
    W        = drivingForces.W;
    if ~isempty(W)
        
        perf2well = getPerforationToWellMapping(W);
        wc = vertcat(W.cells);

        wflux = zeros(G.cells.num,1);
        wflux(wc) = sum(vertcat(wellSol.flux), 2);
        isInj = wflux > 0;
        compWell = vertcat(W.compi);
        compPerf = zeros(G.cells.num, 2);
        compPerf(wc,:) = compWell(perf2well,:);
        
        integrand = @(sW, sW0, fW, c, psi, grad_psi) ...
            bW(c).*wflux(c).*(fW.*(~isInj(c)) + compPerf(c,1).*( isInj(c))).*psi;

        prod = disc.cellInt(model, integrand, wc, state, state0, sWdof, sWdof0, fW);
        
        vol = rldecode(G.cells.volumes(wc), nDof(wc), 1);
        ix = disc.getDofIx(state, Inf, wc);
        water(ix) = water(ix) - prod(ix)./vol;

        % Store well fluxes
        [WC, x, cellNo] = disc.getCubature(wc, 'volume');
        x = disc.transformCoords(x, cellNo);
        sWw = disc.evaluateSaturation(x, cellNo, sWdof, state);
        
        wflux_W = WC*(bW(cellNo).*wflux(cellNo) ...
                     .*(fW(sWw, 1-sWw, cellNo, cellNo) .*(~isInj(cellNo)) ...
                     + compPerf(cellNo,1).*( isInj(cellNo))));
        wflux_W = wflux_W./G.cells.volumes(wc);
          
        wflux_O = WC*(bO(cellNo).*wflux(cellNo) ...
                     .*((1-fW(sWw, 1-sWw, cellNo, cellNo)).*(~isInj(cellNo)) ...
                     + compPerf(cellNo,2) .*( isInj(cellNo))));
        wflux_O = wflux_O./G.cells.volumes(wc);

        wflux_O = double(wflux_O);
        wflux_W = double(wflux_W);

        for wNo = 1:numel(W)
            perfind = perf2well == wNo;
            state.wellSol(wNo).qOs = sum(wflux_O(perfind));
            state.wellSol(wNo).qWs = sum(wflux_W(perfind));
        end

    end
    %----------------------------------------------------------------------
    
    state.cfl = dt.*sum(abs(vTc)./G.cells.dx,2);
    
    % Add BCs--------------------------------------------------------------
    if ~isempty(bc)
        fluxBC = @(sW, fW, c, f, psi) ...
        (bW(c).*fW.*vT(f) + bW(c).*fW.*mobO(1-sW,c).*(TgW(f) - TgO(f))).*psi;%./G.faces.areas(f);
        bcInt = disc.faceFluxIntBC(model, fluxBC, bc, state, sWdof, fW);        
        water = water + bcInt;
    end
    
    % Add sources----------------------------------------------------------
    src = drivingForces.src;
    if ~isempty(src)
        cells = src.cell;
        rate = src.rate;
        sat = src.sat;
        integrand = @(c, psi) ...
                        bW(c).*rate.*sat(:,1).*psi;
        source = disc.cellInt(@(sW, sW0, fW, c, psi, grad_psi) ...
            integrand(c, psi), fW, cells, sWdof, sWdof0, state, state0);
        
        vol = rldecode(G.cells.volumes(cells), nDof(cells), 1);
        
        ix = disc.getDofIx(state, Inf, cells);
        water(ix) = water(ix) - source(ix)./vol;
    end
    
    % Make Linearized problem----------------------------------------------
    eqs   = {water  };
    names = {'water'};
    types = {'cell' };

    if ~model.useCNVConvergence
        pv = rldecode(op.pv, nDof, 1);
        eqs{1} = eqs{1}.*(dt./pv);
    end

    problem = LinearizedProblem(eqs, types, names, primaryVars, state, dt);
    
    if model.extraStateOutput
        state = model.storeDensity(state, rhoW, rhoO, []);
        state.G = [double(gW), double(gO)];
    end
    
    if isfield(G, 'parent')
        ix = disc.getDofIx(state, Inf, ~G.cells.ghost);
        
        for eqNo = 1:numel(problem.equations)
            eq = problem.equations{eqNo};
            if isa(eq, 'ADI')
                eq.val = eq.val(ix);
                eq.jac = cellfun(@(j) j(ix,ix), eq.jac, 'unif', false);
            else
                eq = eq(ix);
            end
            problem.equations{eqNo} = eq;
        end
    end

end

function v = expandSingleValue(v,G)

    if numel(double(v)) == 1
        v = v*ones(G.cells.num,1);
    end
    
end