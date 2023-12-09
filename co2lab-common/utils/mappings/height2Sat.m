function s = height2Sat(h, hmax, Gt, sw, sg, varargin)
% Convert from height to (fine scale) saturation.  By default, a sharp-interface
% approximation is assumed.  If a capillary fringe model should be used, the
% function needs to be provided with the optional arguments 'invPc3D', 'rhoW'
% and 'rhoG'.
% 
% SYNOPSIS:
%   s = height2Sat(sol, Gt, fluid)
%
% PARAMETERS:
%   h - CO2 plume thickness.  One scalar value for each column in the
%       top-surface grid.
%
%       Values less than zero are treated as zero while values below the
%       bottom of a column are treated as the column depth.
%
%   hmax - historically maximum thickness.  One scalar value for each
%          column in the top surface grid
%    
%   Gt - A top-surface grid as defined by function 'topSurfaceGrid'.
%
%   sw - residual water saturation
%   sg - residual gas saturation
% 
%   invPc3D (optional) - If this argument is provided, a capillary fringe
%                        model is assumed.  'invPc3D' should then be the
%                        inverse fine-scale capillary pressure function,
%                        i.e. taking a capillary pressure value as argument
%                        and returning the corresponding saturation.  This
%                        function will be available from the fluid model if
%                        a capillary fringe model is used.
%
%   rhoW, rhoG (optional) - water and CO2 densities, given for each cell in Gt.
%                           These are only required input if a capillary fringe
%                           model is called for, i.e. 'invPc3D' provided.
%
% RETURNS:
%   s - Saturation - one value for each cell in the underlying 3D model.
%   Corresponds to state.s for the 3D problem.
%
% SEE ALSO:
%   `accumulateVertically`, `integrateVertically`

%{
Copyright 2009-2024 SINTEF Digital, Mathematics & Cybernetics.

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

    opt = merge_options(struct('invPc3D', [], 'rhoW', [], 'rhoG', []), varargin{:});
    
    if isempty(opt.invPc3D)
        s = sharp_interface_h2s(h, hmax, Gt, sw, sg);
    else
        s = cap_fringe_h2s(h, hmax, Gt, sw, sg, opt.invPc3D, rhoW, rhoG);
    end
end

% ----------------------------------------------------------------------------
function s = cap_fringe_h2s(h, hmax, Gt, sw, sg, invPc3D, rhoW, rhoG)
    
    % endpoint scaling factor
    C = sg / (1 - sw);
    
    % remap upscaled variables to fine-scale grid
    remap = @(x, ixs) x(ixs);
    to_finescale = @(var) remap(rldecode(var, diff(Gt.cells.columnPos)), Gt.colums.cells);
    
    h_all = to_finescale(h);
    hmax_all = to_finescale(hmax);
    drho_all = to_finescale(rhoW - rhoG);
    
    % compute capillary pressure and take inverse to get effective and max saturations
    s_eff = invPc3D(max(h_all - Gt.parent.cells.centroids(:, 3), 0) .* drho_all * norm(gravity));
    smax =  invPc3D(max(hmax_all - Gt.parent.cells.centroids(:, 3), 0) .* drho_all * norm(gravity));
    
    % combine s_eff and smax to get current fine-scale saturation
    s = (1 - C) * s_eff + C * smax;
    
end

% ----------------------------------------------------------------------------
function s = sharp_interface_h2s(h, hmax, Gt, sw, sg)
    
    s = zeros(numel(Gt.columns.cells),1);
    % n: number of completely filled cells
    % t: fill degree for columns single partially filled cell
    [n, t] = fillDegree(h, Gt); %
    
    % number of cells in each column
    nc = diff(Gt.cells.columnPos);
    
    % compute internal cellNumber in the column for each cell
    cellNoInCol = mcolon(ones(Gt.cells.num,1), diff(Gt.cells.columnPos))';
    
    % f(cells with s == 1)    > 0
    % f(cells with 1 > s > 0) = 0
    % f(cells with s == 0)    < 0
    f = rldecode(n, nc)-cellNoInCol+1;
    
    % completely filled cells
    s(Gt.columns.cells(f>0)) = 1*(1-sw);
    
    %partially filled cells
    s(Gt.columns.cells(f==0)) = t(n<nc)*(1-sw);
    
    if sg > 0 && any(hmax > h)
        % %hysteresis:
        [n_sr, t_sr] = fillDegree(hmax, Gt);
        
        % remove all cells where hmax - h == 0 and leave only the fraction that is
        % residual co2 in cells that have both residual and free co2
        ix = find(n_sr == n);
        t_sr(ix) = max(t_sr(ix) - t(ix),0);
        
        ix2 = n_sr - n >= 1;
        f_sr = rldecode(n_sr, nc) - cellNoInCol + 1;
        
        % cells with residual saturation in the whole cell
        s(Gt.columns.cells(f_sr>0 & f<0)) = sg;
        
        % cells with residual saturation in bottom part of a cell and free co2 on top
        currSat = s(Gt.columns.cells(f_sr>0 &f ==0));
        s(Gt.columns.cells(f_sr>0 & f==0)) = currSat+(1-t(ix2))*sg;
        
        % cells with possible residual saturation in part of the cell and water in the bottom
        currSat = s(Gt.columns.cells(f_sr==0));
        s(Gt.columns.cells(f_sr==0)) = currSat + t_sr(n_sr<nc)*sg;
        
    end
end

