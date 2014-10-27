function [problem, state] = transportEquationOilWater(state0, state, model, dt, drivingForces, varargin)

opt = struct('Verbose', mrstVerbose, ...
             'reverseMode', false,...
             'scaling', [],...
             'resOnly', false,...
             'history', [],...
             'solveForWater', false, ...
             'solveForOil', true, ...
             'iteration', -1, ...
             'stepOptions', []);  % Compatibility only

opt = merge_options(opt, varargin{:});

W = drivingForces.Wells;
assert(isempty(drivingForces.bc) && isempty(drivingForces.src))

s = model.operators;
f = model.fluid;
G = model.G;

assert(~(opt.solveForWater && opt.solveForOil));

[p, sW, wellSol] = model.getProps(state, 'pressure', 'water', 'wellsol');

[p0, sW0] = model.getProps(state0, 'pressure', 'water');

wflux = vertcat(wellSol.flux);

%Initialization of independent variables ----------------------------------

if ~opt.resOnly,
    % ADI variables needed since we are not only computing residuals.
    if ~opt.reverseMode,
        sW = initVariablesADI(sW);
    else
        assert(0, 'Backwards solver not supported for splitting');
    end
end
primaryVars = {'sW'};

clear tmp
g  = norm(gravity);


% -------------------------------------------------------------------------
[krW, krO] = f.relPerm(sW);

clear krW_o krO_o

%dZ = s.grad(G.cells.centroids(:,3));
grav = gravity;
gdz = s.Grad(G.cells.centroids) * grav';

sO = 1 - sW;
% Water
[bW, rhoW, mobW, dpW, Gw] = propsOW_water(sW, krW, gdz, f, p, s);
[bO, rhoO, mobO, dpO, Go] = propsOW_oil(  sO, krO, gdz, f, p, s);


    
if model.extraStateOutput
    state = model.storebfactors(state, bW, bO, []);
    state = model.storeMobilities(state, mobW, mobO, []);
end

if ~isempty(W)
    perf2well = getPerforationToWellMapping(W);
    wc = vertcat(W.cells);
    
    mobWw = mobW(wc);
    mobOw = mobO(wc);
    totMobw = mobWw + mobOw;

    f_w_w = mobWw./totMobw;
    f_o_w = mobOw./totMobw;

    isInj = wflux > 0;
    compWell = vertcat(W.compi);
    compPerf = compWell(perf2well, :);

    f_w_w(isInj) = compPerf(isInj, 1);
    f_o_w(isInj) = compPerf(isInj, 2);

    bWqW = bW(wc).*f_w_w.*wflux;
    bOqO = bO(wc).*f_o_w.*wflux;

    % Store well fluxes
    wflux_O = double(bOqO);
    wflux_W = double(bWqW);
    
    for i = 1:numel(W)
        perfind = perf2well == i;
        state.wellSol(i).qOs = sum(wflux_O(perfind));
        state.wellSol(i).qWs = sum(wflux_W(perfind));
    end

end
%check for p-dependent porv mult:
pvMult = 1; pvMult0 = 1;
if isfield(f, 'pvMultR')
    pvMult =  f.pvMultR(p);
    pvMult0 = f.pvMultR(p0);
end


% Get total flux from state
flux = sum(state.flux, 2);
vT = flux(model.operators.internalConn);

% Stored upstream indices
if model.staticUpwind
    flag = state.upstreamFlag;
else
    flag = multiphaseUpwindIndices({Gw, Go}, vT, s.T, {bW.*mobW, bO.*mobO}, s.faceUpstr);
end

upcw  = flag(:, 1);
upco  = flag(:, 2);

    
mobOf = s.faceUpstr(upco, mobO);
mobWf = s.faceUpstr(upcw, mobW);

totMob = (mobOf + mobWf);
totMob = max(totMob, sqrt(eps));

if opt.solveForWater
    f_w = mobWf./totMob;
    bWvW   = s.faceUpstr(upcw, bW).*f_w.*(vT + s.T.*mobOf.*(Gw - Go));

    wat = (s.pv/dt).*(pvMult.*bW.*sW       - pvMult0.*f.bW(p0).*sW0    ) + s.Div(bWvW);
    wat(wc) = wat(wc) - bWqW;
    
    eqs{1} = wat;
    names = {'water'};
    types = {'cell'};
else
    f_o = mobOf./totMob;
    bOvO   = s.faceUpstr(upco, bO).*f_o.*(vT + s.T.*mobWf.*(Go - Gw));

    oil = (s.pv/dt).*( pvMult.*bO.*(1-sW) - pvMult0.*f.bO(p0).*(1-sW0) ) + s.Div(bOvO);
    oil(wc) = oil(wc) - bOqO;
    
    eqs{1} = oil;
    names = {'oil'};
    types = {'cell'};
end




problem = LinearizedProblem(eqs, types, names, primaryVars, state, dt);
problem.iterationNo = opt.iteration;

% perf2well = getPerforationToWellMapping(W);

end
