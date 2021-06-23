classdef ModelParameter
    properties
        name
        type          = 'value';    % 'value'/'multiplier'
        boxLims                     % upper/lower value(s) for parameters (used for scaling)
        subset                      % subset of parameters (or subset of wells)
        scaling       = 'linear'    % 'linear'/'log'
        referenceValue              % parameter reference values (used for type 'multiplier') 
        belongsTo                   % model/well/state0
        location                    % e.g., {'operators', 'T'}
        n                           % number of parameters
        lumping                     % parameter lumping vector (partition vector) 
        setfun                      % possibly custom set-function (default is setfield)
    end
    
    methods
        function p = ModelParameter(SimulatorSetup, varargin)
            [p, extra] = merge_options(p, varargin{:});
            assert(~isempty(p.name), 'Parameter name can''t be defaulted');
            if isempty(p.belongsTo) || isempty(p.location) 
                p = setupByName(p, SimulatorSetup);
            end
            opt = struct('relativeLimits', [.5 2]);
            opt = merge_options(opt, extra{:});
            p   = setupDefaults(p, SimulatorSetup, opt);
            if isempty(p.setfun)
                % use default
                p.setfun = @(obj, loc, v)setfield(obj, loc{:}, v);
            end
        end
        %------------------------------------------------------------------
        function vs = scale(p, pval)
            % map parameter pval to "control"-vector v \in [0,1]
            if strcmp(p.type, 'multiplier')
                pval = pval./p.referenceValue;
            end
            if strcmp(p.scaling, 'linear')
                vs = (pval-p.boxLims(:,1))./diff(p.boxLims, [], 2);
            elseif strcmp(p.scaling, 'log')
                logLims = log(p.boxLims);
                vs = (log(pval)-logLims(:,1))./diff(logLims, [], 2);
            end
        end
        %------------------------------------------------------------------
        function pval = unscale(p, vs)
            % retrieve parameter pval from "control"-vector v \in [0,1]
            if strcmp(p.scaling, 'linear')
                pval = vs.*diff(p.boxLims, [], 2) + p.boxLims(:,1);
            elseif strcmp(p.scaling, 'log')
                logLims = log(p.boxLims);
                pval = exp(vs.*diff(logLims, [], 2) + logLims(:,1));
            end
            if strcmp(p.type, 'multiplier')
                pval = pval.*p.referenceValue;
            end
        end
        %------------------------------------------------------------------
        function gs = scaleGradient(p, g, pval)
            % map gradient wrt param to gradient vs "control"-vector
            % parameter value pval only needed for log-scalings
            %g = collapseLumps(g, p.lumping, 'sum');
            if strcmp(p.scaling, 'linear')
                if strcmp(p.type, 'value')
                    gs = g.*diff(p.boxLims, [], 2);
                elseif strcmp(p.type, 'multiplier')
                    gs = (g.*p.referenceValue).*diff(p.boxLims, [], 2);
                end
            elseif strcmp(p.scaling, 'log')
                gs = (g.*pval).*diff(log(p.boxLims), [], 2);
            end
        end
        %------------------------------------------------------------------
        function v = getParameterValue(p, SimulatorSetup)
            if ~strcmp(p.belongsTo, 'well')
                v = getfield(SimulatorSetup.(p.belongsTo), p.location{:});
                v = collapseLumps(v(p.subset), p.lumping);
            else % well-parameter (assume constant over control steps)
                v = p.getWellParameterValue(SimulatorSetup.schedule.control(1).W);
            end
        end
        %------------------------------------------------------------------       
        function SimulatorSetup = setParameterValue(p, SimulatorSetup, v)
            if ~strcmp(p.belongsTo, 'well')
                v  = expandLumps(v, p.lumping);
                if isnumeric(p.subset)
                    tmp = getfield(SimulatorSetup.(p.belongsTo), p.location{:});
                    v   = setSubset(tmp, v, p.subset);
                end
                SimulatorSetup.(p.belongsTo) = ...
                    p.setfun(SimulatorSetup.(p.belongsTo), p.location, v);
            else % well-parameter (assume constant over control steps)
                for k = 1:numel(SimulatorSetup.schedule.control)
                    SimulatorSetup.schedule.control(k).W = ...
                        p.setWellParameterValue(SimulatorSetup.schedule.control(k).W, v);
                end
            end
        end
        %------------------------------------------------------------------       
        function v = getWellParameterValue(p, W)
            assert(strcmp(p.belongsTo, 'well'))
            v = applyFunction(@(x)getfield(x, p.location{:}), W(p.subset));
            if iscell(p.lumping)
                v = applyFunction(@(vi,lump)collapseLumps(vi, lump), v, p.lumping);
            end
            v = vertcat(v{:});
        end
        %------------------------------------------------------------------       
        function W = setWellParameterValue(p, W, v)
            sub = p.subset;
            if ~isnumeric(sub)
                sub = 1:numel(W);
            end 
            nc = arrayfun(@(w)numel(w.cells), W(sub));
            [i1, i2] = deal(cumsum([1;nc(1:end-1)]), cumsum(nc));
            v  = applyFunction(@(i1,i2)v(i1:i2), i1, i2);
            if iscell(p.lumping)
                v = applyFunction(@(vi,lump)expandLumps(vi, lump), v, p.lumping);
            end
            for k = sub
                W(k) = setfield(W(k), p.location{:}, v{k});
            end
        end
        %------------------------------------------------------------------       
        function m = getMultiplerValue(p, SimulatorSetup, doLump)
            if strcmp(p.type, 'multiplier')
                m = p.getParameterValue(SimulatorSetup)./p.referenceValue;
                if nargin == 3 && doLump
                    m = collapseLumps(p, m, @mean);
                end
            else
                error('Parameter %s is not of type ''multiplier''', p.name);
            end
        end
    end
end

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

function p = setupDefaults(p, SimulatorSetup, opt)
% Make sure setup makes sense and add boxLims if not provided
rlim  = opt.relativeLimits;
range = @(x)[min(min(x)), max(max(x))];
if isempty(p.subset)
    p.subset = ':';
end
if islogical(p.subset)
    p.subset = find(p.subset);
end
% check if well-parameter
if strcmp(p.belongsTo, 'well') && isempty(p.lumping)
    % for non-empty lumping, there should be a list of lumping-vectors for
    % each included well.
    if ~ischar(p.subset)
        nw = numel(p.subset);
    else
        nw = numel(SimulatorSetup.schedule.control(1).W);
    end
    p.lumping = cell(nw, 1);
end
v    = getParameterValue(p, SimulatorSetup);
p.n  = numel(v);

if isempty(p.boxLims)
    if strcmp(p.type, 'value')
        p.boxLims = range(v).*rlim;
    else
        p.boxLims = rlim;
    end
end

assert(any(size(p.boxLims,1) == [1, p.n]), ...
    'Property ''boxLims'' does not match number of parameters');

if strcmp(p.type, 'multiplier')
    p.referenceValue = v;
end
end
%--------------------------------------------------------------------------
function p = setupByName(p, SimulatorSetup)
% setup for typical parameters
setfun = [];
switch lower(p.name)
    case 'transmissibility'
        belongsTo = 'model';
        location = {'operators', 'T'};
    case {'permx', 'permy', 'permz'}
        belongsTo = 'model';
        col = find(strcmpi(p.name(end), {'x', 'y', 'z'}));
        location = {'rock', 'perm', {':', col}};
        setfun   = @setPermeabilityFun;
    case 'porevolume'
        belongsTo = 'model';
        location = {'operators', 'pv'};
    case 'conntrans'
        belongsTo = 'well';
        location = {'WI'};
    case {'sw', 'sg'}
        belongsTo = 'state0';
        col = SimulatorSetup.model.getPhaseIndex(upper(p.name(end)));
        location = {'s', {':', col}};
        oix = SimulatorSetup.model.getPhaseIndex('O');
        assert(~isempty(oix), ...
            'Current assumption is that oil is the dependent phase');
        setfun   = @(obj, loc, v)setSaturationFun(obj, loc, v, oix);
    case 'pressure'
        belongsTo = 'state0';
        location = {'pressure'};
    case {'swl', 'swcr', 'swu', 'sowcr', 'sogcr', 'sgl', 'sgcr', ...
            'sgu', 'krw', 'kro', 'krg'}
        belongsTo = 'model';
        map = getScalerMap();
        ix  = map.kw.(upper(p.name));
        [ph, col] = deal(map.ph{ix(1)}, ix(2));
        location = {'rock', 'krscale', 'drainage', ph, {':', col}};
        setfun   = @setRelPermScalersFun;
    otherwise
        error('No default setup for parameter: %s\n', p.name);     
end
if isempty(p.belongsTo)
    p.belongsTo = belongsTo;
end
if isempty(p.location)
    p.location = location;
end
if isempty(p.setfun) && ~isempty(setfun)
    p.setfun = setfun;
end
end
%--------------------------------------------------------------------------            

function map = getScalerMap()
phOpts = {'w', 'ow', 'g', 'og'};
kw  = struct('SWL',   [1,1], 'SWCR',  [1,2], 'SWU', [1,3], ...
             'SGL',   [3,1], 'SGCR',  [3,2], 'SGU', [3,3], ...
             'SOWCR', [2,2], 'SOGCR', [4,2], ...
             'KRW',   [1,4], 'KRO',   [2,4], 'KRG', [3,4]);
map = struct('ph', {phOpts}, 'kw', kw);
end
%--------------------------------------------------------------------------

function v = collapseLumps(v, lumps)
% take mean of each lump
if ~isempty(lumps) && isnumeric(lumps)
    if numel(lumps) == 1 && lumps==1
        % treat as special case (one lump)
        v = sum(v)/numelValue(v);
    else
        if isa(v, 'double')
            v = accumarray(lumps,v, [], @mean);
        else % special treatment in case of ADI
            M = sparse(lumps, (1:numel(lumps))', 1);
            v = (M*v)./sum(M,2);
        end
    end
end
end
%--------------------------------------------------------------------------

function v = expandLumps(v, lumps)
if ~isempty(lumps) && isnumeric(lumps)
    v = v(lumps);
end
end
%--------------------------------------------------------------------------

function v = setSubset(v, vi, sub)
if isa(vi, 'ADI')
    v = double2ADI(v, vi);
end
v(sub) = vi;
end

%--------------------------------------------------------------------------       
function model = setPermeabilityFun(model, location, v)
% utility for setting permx/y/z possibly as AD and include effect on
% transmissibilities
[nc, nd] = size(model.rock.perm);
perm = model.rock.perm;
if ~iscell(perm)
    perm = mat2cell(perm, nc, ones(1,nd));
end
col = location{end}{end};
assert(col<=nd, 'Can''t get column %d since perm has %d column(s)', col, nd);
perm{col} = v;
% transmissibilities
th = 0;
for k = 1:nd
    th = th + perm2directionalTrans(model, perm{k}, k);
end
cf = model.G.cells.faces(:,1);
nf = model.G.faces.num;
% mapping from from cell-face to face
M = sparse(cf, (1:numel(cf))', 1, nf, numel(cf));
% consider only internal faces
ie = model.operators.internalConn;
model.operators.T = 1./(M(ie,:)*(1./th));
% if all perms are doubles, set to nc x nd array
if all(cellfun(@(p)isa(p, 'double'), perm))
    model.rock.perm = cell2mat(perm);
else
    model.rock.perm = perm;
end
end

%--------------------------------------------------------------------------       
function state = setSaturationFun(state, location, v, oix)
assert(isa(v, 'double'), 'Setting saturation to class %s is not supported', class(v));
pix = location{end}{end};
ds = v-state.s(:, pix);
state.s(:, pix) = v;
state.s(:, oix) =  state.s(:, oix) - ds;
end

%--------------------------------------------------------------------------       
function model = setRelPermScalersFun(model, location, v)
if ~isa(v, 'ADI')
    model = setfield(model, location{:}, v);
else
    % last location is column no, second last is phase
    col = location{end}{end};
    ph  = location{end-1};
    d   = getfield(model, location{1:end-2});  %#ok
    if ~isfield(d, 'tmp') || ~isfield(d.tmp, ph)
        nc = model.G.cells.num;
        d.tmp.(ph) = mat2cell(d.(ph), nc, ones(1,4));
    end
    d.tmp.(ph){col} = v;
    d.(ph) = @(cells, col)d.tmp.(ph){col}(cells);
    model.rock.krscale.drainage = d;
end
end
       

function ti = perm2directionalTrans(model, p, cdir)
% special utility function for calculating transmissibility along coordinate direction cdir
% In particular:
% trans = t1+t2+t2, where ti = perm2dirtrans(model, perm(:, i), i);
assert(size(p,2)==1, ...
       'Input p should be single column representing permeability in direction cdir');
% make perm represent diag perm tensor with zero perm orthogonal to cdir
dp = value(p);
r.perm = zeros(numel(dp), 3);
r.perm(:, cdir) = dp;
ti = computeTrans(model.G, r);
if isa(p, 'ADI')
    % make ti ADI (note ti is linear function of p)
    p = p./dp;
    cellno = rldecode(1 : model.G.cells.num, diff(model.G.cells.facePos), 2).';
    ti = ti.*p(cellno);
end
end


