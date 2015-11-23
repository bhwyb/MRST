function [problem, state] = pressureEquationOilWaterPolymer(state0, ...
    state, model, dt, drivingForces, varargin)

opt = struct('Verbose', mrstVerbose, ...
             'reverseMode', false,...
             'resOnly', false,...
             'propsPressure', [], ...
             'staticWells',  false, ...
             'iteration', -1);

opt = merge_options(opt, varargin{:});

W = drivingForces.W;

assert(isempty(drivingForces.bc) && isempty(drivingForces.src))

s = model.operators;
f = model.fluid;

% Polymer shear thinning/thickening
usingShear = isfield(f, 'plyshearMult');

% Properties at current timestep
[p, sW, c, cmax, wellSol] = model.getProps(state, 'pressure', 'water', ...
    'polymer', 'polymermax', 'wellsol');

% Properties at previous timestep
[p0, sW0, c0, cmax0] = model.getProps(state0, 'pressure', 'water', ...
   'polymer', 'polymermax');


pBH    = vertcat(wellSol.bhp);
qWs    = vertcat(wellSol.qWs);
qOs    = vertcat(wellSol.qOs);
qWPoly = vertcat(wellSol.qWPoly);

%Initialization of independent variables ----------------------------------

if ~opt.resOnly,
    % ADI variables needed since we are not only computing residuals.
    if ~opt.reverseMode,
        [p, qWs, qOs, qWPoly, pBH] = ...
            initVariablesADI(p, qWs, qOs, qWPoly, pBH);
    else
        assert(0, 'Backwards solver not supported for splitting');
    end
end
primaryVars = {'pressure', 'qWs', 'qOs', 'qWPoly', 'bhp'};

p_prop = opt.propsPressure;
if isempty(p_prop)
    p_prop = p;
end
% otherPropPressure = ~isempty(p_prop);
% if ~otherPropPressure
%     p_prop = p;
% end

% -------------------------------------------------------------------------
sO  = 1 - sW;
sO0 = 1 - sW0;

[krW, krO] = model.evaluteRelPerm({sW, sO});

% Multipliers for properties
[pvMult, transMult, mobMult, pvMult0] = getMultipliers(model.fluid, ...
    p_prop, p0);

% Modifiy relperm by mobility multiplier (if any)
krW = mobMult.*krW; krO = mobMult.*krO;

% Compute transmissibility
T = s.T.*transMult;

% Gravity contribution
gdz = model.getGravityGradient();

% Evaluate water and polymer properties
ads  = effads(c, cmax, model);
[vW, vP, bW, muWMult, mobW, mobP, rhoW, pW, upcw, dpW, extraOutput] = ...
    getFluxAndPropsWaterPolymer_BO(model, p, p_prop, sW, c, ads, ...
    krW, T, gdz); %, 'shear', ~otherPropPressure);
bW0 = f.bW(p0);

% Evaluate oil properties
[vO, bO, mobO, rhoO, pO, upco, dpO] = getFluxAndPropsOil_BO(model, ...
    p, p_prop, sO, krO, T, gdz);
bO0 = getbO_BO(model, p0);

% if otherPropPressure
%     % We have used a different pressure for property evaluation, undo the
%     % effects of this on the fluxes.
%     dp_diff = s.Grad(p) - s.Grad(p_prop);
%     
%     vW = -s.faceUpstr(upcw, mobW).*s.T.*(dpW + dp_diff);
%     vO = -s.faceUpstr(upco, mobO).*s.T.*(dpO + dp_diff);
%     vP = -s.faceUpstr(upcw, mobP).*s.T.*(dpW + dp_diff);
%     
%     % If other property pressure is used, then the shear thinning is
%     % computed here, and not above
%     if usingShear
%         poroFace  = s.faceAvg(model.rock.poro);
%         faceArea  = model.G.faces.areas(s.internalConn);
%         Vw        = vW./(poroFace .* faceArea); % water velocity
%         muWMultf  = s.faceUpstr(upcw, muWMult);
%         [shearMult, ~, shearReport] = ...
%             getPolymerShearMultiplier(model, Vw, muWMultf);
%         vW        = vW .* shearMult;
%         vP        = vP .* shearMult;
%         extraOutput.shearMult   = shearMult;
%         extraOutput.shearReport = shearReport;
%     end
% end

% These are needed in transport solver, so we output them regardless of
% any flags set in the model.
state = model.storeFluxes(state, vW, vO, vP);
state = model.storeUpstreamIndices(state, upcw, upco, []);
if model.extraStateOutput
    state = model.storebfactors(state, bW, bO, []);
    state = model.storeMobilities(state, mobW, mobO, mobP);
end

if usingShear
    % The shear multipliers are stored for use in the transport solver
    state = model.storeShearMultiplier(state, extraOutput.shearMult);
end

if model.extraPolymerOutput
    state = model.storeEffectiveWaterVisc(state, extraOutput.muWeff);
    state = model.storeEffectivePolymerVisc(state, extraOutput.muPeff);
    state = model.storePolymerAdsorption(state, ads);
    state = model.storeRelpermReductionFactor(state, extraOutput.Rk);
    if usingShear
        state.ShearThinningReport = extraOutput.shearReport;
    end
end


% EQUATIONS ---------------------------------------------------------------
% Upstream weight b factors and multiply by interface fluxes to obtain the
% fluxes at standard conditions.
bOvO = s.faceUpstr(upco, bO).*vO;
bWvW = s.faceUpstr(upcw, bW).*vW;

% Oil
oil = (s.pv/dt).*( pvMult.*bO.*sO - pvMult0.*bO0.*sO0) + s.Div(bOvO);

% Water
wat = (s.pv/dt).*( pvMult.*bW.*sW - pvMult0.*bW0.*sW0 ) + s.Div(bWvW);

[eqs, names, types] = deal({});

% well equations
if ~isempty(W)
    wc    = vertcat(W.cells);
    perf2well = getPerforationToWellMapping(W);
    if opt.staticWells
        q = vertcat(state.wellSol.flux);
        
        qW = q(:, 1);
        qO = q(:, 2);
        
        cqs = {bW(wc).*qW, bO(wc).*qO};
    else
        pw   = p(wc);
        rhos = [f.rhoWS, f.rhoOS];
        bw   = {bW(wc), bO(wc)};
        mw   = {mobW(wc), mobO(wc)};
        s    = {sW(wc), 1 - sW(wc)};
        wm   = model.wellmodel;
        
        % Polymer well equations
        [~, wciPoly, iInxW] = getWellPolymer(W);
        cw        = c(wc);
        cw(iInxW) = wciPoly;
        
        if usingShear
            % Compute shear rate multiplier for wells
            % The water velocity is computed using a the reprensentative 
            % radius rR.
            % rR = sqrt(rW * rA)
            % rW is the well bore radius.
            % rA is the equivalent radius of the grid block in which the 
            %       well is completed.

            assert(isfield(W, 'rR'), ...
                'The representative radius needs to be suppplied.');

            muWMultW = muWMult(wc);
            % Maybe should also apply this for PRODUCTION wells.
            pinx = wciPoly==0;
            if isfield(wellSol, 'poly_prev')
                % Special hack for the sequential solver with shear
                % thinning. If a well control has changed such that the
                % polymer injection has gone down, we can get a peak in the
                % bhp at the next timestep if we set the shear multiplier
                % to zero. Therefore, we do not force the multiplier to
                % zero the first timestep.
                polyRedInx  = ([wellSol.poly_prev]>[W.poly])';
                polyRedWell = polyRedInx(perf2well);
                pinx  = pinx & ~polyRedWell(iInxW);
            end
            muWMultW(iInxW(pinx)) = 1;

            cqs = wm.computeWellFlux(model, W, wellSol, pBH, ...
                {qWs, qOs}, pw, rhos, bw, mw, s, {},...
                'nonlinearIteration', opt.iteration);

            % The following formulations assume that the wells are always
            % in the z direction 
            % IMPROVED HERE LATER
            [~, ~, dz] = cellDims(model.G, wc);
            
            % HACK FOR 2D MODELS
            if all(dz==0)
                dz(:) = 1;
            end
            
            if model.extraPolymerOutput
                cqsW0 = double(cqs{1});
                mobW0 = double(mw{1});
            end

            rR = vertcat(W.rR);
            A  = rR.*dz*2*pi; % representative area of each well cell
            VW0W = double(bW(wc)).*double(cqs{1}) ./ ...
                   (model.rock.poro(wc).*A);
            [shearMultW, VW1W] = getPolymerShearMultiplier(model, VW0W, ...
                muWMultW);

            % Apply shear multiplier to water
            mw{1} = mw{1}.*shearMultW;

        end
        
        [cqs, weqs, ctrleqs, wc, state.wellSol, cqr]  = ...
            wm.computeWellFlux(model, W, wellSol, pBH, {qWs, qOs}, pw, ...
            rhos, bw, mw, s, {}, 'nonlinearIteration', opt.iteration);
        eqs(2:3) = weqs;
        eqs{5} = ctrleqs;

        qW = cqr{1};
        qO = cqr{2};
        
        % Polymer well equations
        % Well polymer rate for each well is water rate in each perforation
        % multiplied with polymer concentration in that perforated cell.
        Rw = sparse(perf2well, (1:numel(perf2well))', 1, ...
            numel(W), numel(perf2well));
        cqsPoly = Rw*(cqs{1}.*cw);
        eqs{4}  = qWPoly - cqsPoly;

        if usingShear
            % Store well shear multipliers as these are needed in the
            % transport solver
            shearMultW = double(shearMultW);
            for wnr = 1:numel(state.wellSol)
                ix = perf2well == wnr;
                state.wellSol(wnr).shearMult = shearMultW(ix);
            end
        end

        % Save extra polymer welldata if requested
        if model.extraPolymerOutput
            cqsPoly    = double(cqsPoly);
            for wnr = 1:numel(state.wellSol)
                ix = perf2well == wnr;
                state.wellSol(wnr).cqsPoly = cqsPoly(wnr);
                if usingShear
                    % Special flag (see above for this hack)
                    state.wellSol(wnr).shearFixFlag = false;
                    if isfield(wellSol, 'poly_prev')
                        state.wellSol(wnr).shearFixFlag = polyRedWell(wnr);
                    end
                end
            end
        end

        names(2:5) = {'oilWells', 'waterWells', 'polymerWells', ...
            'closureWells'};
        types(2:5) = {'perf', 'perf', 'perf', 'well'};

    end
    
    wat(wc) = wat(wc) - cqs{1};
    oil(wc) = oil(wc) - cqs{2};
end

eqs{1} = oil./bO + wat./bW;
names{1} = 'pressure';
types{1} = 'cell';

problem = LinearizedProblem(eqs, types, names, primaryVars, state, dt);

for i = 1:numel(W)
    wp = perf2well == i;
    state.wellSol(i).flux = [double(qW(wp)), double(qO(wp))];
end

end


%--------------------------------------------------------------------------
% Helper functions
%--------------------------------------------------------------------------


% Effective adsorption, depending of desorption or not
function y = effads(c, cmax, model)
   if model.fluid.adsInx == 2
      y = model.fluid.ads(max(c, cmax));
   else
      y = model.fluid.ads(c);
   end
end


function [dx, dy, dz] = cellDims(G, ix)
% cellDims -- Compute physical dimensions of all cells in single well
%
% SYNOPSIS:
%   [dx, dy, dz] = cellDims(G, ix)
%
% PARAMETERS:
%   G  - Grid data structure.
%   ix - Cells for which to compute the physical dimensions
%
% RETURNS:
%   dx, dy, dz -- [dx(k) dy(k)] is bounding box in xy-plane, while dz(k) =
%                 V(k)/dx(k)*dy(k)

    n = numel(ix);
    [dx, dy, dz] = deal(zeros([n, 1]));

    ixc = G.cells.facePos;
    ixf = G.faces.nodePos;

    for k = 1 : n,
       c = ix(k);                                     % Current cell
       f = G.cells.faces(ixc(c) : ixc(c + 1) - 1, 1); % Faces on cell
       e = mcolon(ixf(f), ixf(f + 1) - 1);            % Edges on cell

       nodes  = unique(G.faces.nodes(e, 1));          % Unique nodes...
       coords = G.nodes.coords(nodes,:);            % ... and coordinates

       % Compute bounding box
       m = min(coords);
       M = max(coords);

       % Size of bounding box
       dx(k) = M(1) - m(1);
       if size(G.nodes.coords, 2) > 1,
          dy(k) = M(2) - m(2);
       else
          dy(k) = 1;
       end

       if size(G.nodes.coords, 2) > 2,
          dz(k) = G.cells.volumes(ix(k))/(dx(k)*dy(k));
       else
          dz(k) = 0;
       end
    end
end

