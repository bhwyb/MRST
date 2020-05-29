function [state, diagnostics] = computePressureAndDiagnostics(model, varargin)
% Compute pressure equation and corresponding diagnostics for given model 
%
% SYNOPSIS:
% [state, diagnostics] = computePressureAndDiagnostics(model, 'pn1', pv1)
%
% REQUIRED PARAMETERS:
% model -   model containing grid, rock and operators. Occuring fluid-field 
%           of model is NOT used (see optional input fluid)  
%
% 'wells'- option also needs to be non-empty (current version)
%
% OPTIONAL PARAMETERS:
% 'state0' -        Base state (used for mobility-calculations in case of 
%                   non-empty fluid)
%
% 'fluid'  -        Fluid (non-AD-type) for passing to incompTPFA
%
% 'wells'  -        Well-structure passed on to incompTPFA
%
% 'ellipticSolver'- Linear solver passed to incompTPFA
% 
% 'maxTOF', 'computeWellTOFs', 'processCycles', 'firstArrival' all passed
%  to computeTOFandTracer
%
% RETURNS
% state         - contining computed pressure/flux
% diagnostics   - structure containing fields 'D' (tof/tracer fields), 'WP'
%                 (interaction allocation/volumes) and 'wellCommuniaction'

%{
Copyright 2009-2020 SINTEF Digital, Mathematics & Cybernetics.

This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).

MRST is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MRST is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MRST.  If not, see <http://www.gnu.org/licenses/>.
%}

opt = struct('D',                        [], ...
             'state',                    [], ...
             'WP',                       [], ...  
             'wellCommunication',        [], ...
             'state0',                   [], ...
             'wells',                    [], ...
             'ellipticSolver',    @mldivide, ...
             'maxTOF',              [], ...
             'computeWellTOFs',   true, ...
             'processCycles',     true, ...
             'firstArrival',      true);

[opt, rest] = merge_options(opt, varargin{:}); 

if ~isempty(opt.state0)
    state0 = opt.state0;
else
    state0 = initResSol(model.G, 200*barsa);
end

% if ~isempty(opt.fluid)
%     fluid = opt.fluid;
% else
%     fluid = initSingleFluid('mu' ,         1*centi*poise, ...
%                             'rho', 1014*kilogram/meter^3);
% end

W = opt.wells;
if isempty(W)
    error('Empty well-input, can''t compute diagnostics');
end

% compute pressure
isActive = vertcat(W.status);

if ~isempty(opt.state)
    state = opt.state;
else
    model.fluid = convertToIncompressibleFluid(model, 'state', state0);
    state = incompTPFA(state0, model.G, model.operators.T_all, model.fluid, ...
                       'wells', W(isActive), 'LinSolve', opt.ellipticSolver, ...
                       'use_trans', true);
end

ws = repmat(state.wellSol(1), [1, numel(W)]);
ws(isActive) = state.wellSol;
zix = find(~isActive);
for k = 1:numel(zix)
    ws(zix(k)).flux = zeros(numel(W(zix(k)).cells), 1);
end
state.wellSol = ws;

%% diagnostics part    
if ~isempty(opt.D)
    D = opt.D;
else
    D  = computeTOFandTracer(state, model.G, model.rock, 'wells', W, ...
                                 'maxTOF',              [], ...
                                 'computeWellTOFs',   true, ...
                                 'processCycles',     true, ...
                                 'firstArrival',      true);
end

if ~isempty(opt.WP)
    WP = opt.WP;
else
    WP = computeWellPairs(state, model.G, model.rock, W, D);
end

diagnostics = struct('D', D, 'WP', WP);
         
% compute well-communication matrix
if ~isempty(opt.wellCommunication)
    diagnostics.wellCommunication = opt.wellCommunication;
else
    salloc = cellfun(@sum, {WP.inj.alloc}, 'UniformOutput',false);
    diagnostics.wellCommunication = vertcat(salloc{:});
end
    
end
