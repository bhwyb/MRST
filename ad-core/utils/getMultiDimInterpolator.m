function [fn, F, dFdx] = getMultiDimInterpolator(x, Fx, extrap)
% Get a multidimensional interpolator (with support for ADI varibles)
%
% SYNOPSIS:
%       fn = getMultiDimInterpolator(x, Y);
%
% PARAMETERS:
%   x       - Cell array containing the vectors for the points of the
%             function. Must have equal length to the number of dimensions
%             in Y.
%
%   Y       - A matrix of dimension equal to x, representing a structured
%             dataset for multdimensional interpolation
%
% RETURNS:
%   fn      - A function for interpolation with variable arguments equal to
%             the length of x.
%
% SEE ALSO:
%   `interpTable`
    if nargin == 2
        extrap = 'linear';
    end
    
    sz = size(Fx);
    assert(iscell(x));
    if numel(x) > 1
        assert(numel(sz) == numel(x));
        assert(all(cellfun(@numel, x) == sz));
    end
    
    nvar = numel(x);
    dFdx = cell(nvar, 1);
    
    for i = 1:nvar
        if diff(x{i}(1:2)) < 0
            x{i} = x{i}(end:-1:1);
            Fx = flip(Fx, i);
        end
    end
    
    F = griddedInterpolant(x, Fx, 'linear', extrap);
    for i = 1:nvar
        dyi = diff(Fx, 1, i);
        dxi = diff(x{i});
        % delta y / delta x
        if nvar > 1
            % Permute to allow for use of bsxfun
            p = 1:nvar;
            p(i) = 1;
            p(1) = i;
            dyidx = bsxfun(@rdivide, permute(dyi, p), dxi);
            % Permute back
            dyidx = ipermute(dyidx, p);
        else
            dyidx = dyi./dxi;
        end

        % Evaluate using midpoints
        xi = x;
        xi{i} = xi{i}(1:end-1) + dxi/2;
        dFdx{i} = griddedInterpolant(xi, dyidx, 'nearest', 'nearest');
    end
    
    fn = @(varargin) interpTableND(F, dFdx, varargin{:});
end

function Ye = interpTableND(Y, dY, varargin)
    Yq = varargin;
    nvar = numel(dY);
    assert(numel(Yq) == nvar);
    
    isad = cellfun(@(x) isa(x, 'ADI'), Yq);
    isnewad = cellfun(@(x) isa(x, 'GenericAD'), Yq);
    if any(isad)
        % Get a sample variable
        ad = Yq(isad);
        ad = ad{1};
        
        % Create double as input for the function evaluation
        Yqd = cellfun(@value, Yq, 'UniformOutput', false);
        Ye = Y(Yqd{:});
        % Cast to ADI
        if any(isnewad)
            Ye = double2GenericAD(Ye, ad);
        else
            Ye = double2ADI(Ye, ad);
        end
        for i = 1:nvar
            if isad(i)
                % If it is AD, compute derivative using chain rule
                dydx = dY{i}(Yqd{:});
                ix = (1:numel(dydx))';
                d = sparse(ix, ix, dydx);
                for j = 1:numel(Ye.jac)
                    Ye.jac{j} = Ye.jac{j} + d*Yq{i}.jac{j};
                end
            end
        end
    else
        % Just use interpolator directly
        Ye = Y(Yq{:});
    end
end