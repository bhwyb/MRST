function fluid = initVEFluidHForm(g_top, varargin)
% Initialize incompressible two-phase fluid model for VE using saturation
% height formulation
%
% SYNOPSIS:
%   fluid = initVEFluidHForm
%   fluid = initVEFluidHForm(g_top, 'pn1', pv1, ...)
%
% PARAMETERS:
%   g_top   - grid structure for top surface
%
%   'pn'/pv - List of 'key'/value pairs defining specific fluid
%             characteristics. The following parameters must be defined:
%               - mu  -- phase viscosities in units of Pa*s,
%               - rho -- phase densities in units of kiilogram/meter^3,
%               - sr  -- residual CO2 saturation,
%               - sw  -- residual water saturation,
%               - kwm -- phase relative permeability at 'sr' and 'sw'.
%
% RETURNS:
%   fluid - Fluid data structure representing the current state of the
%           fluids within the reservoir model. Contains scalar fields and
%           function handles that take a structure sol containing a field h
%           (height) as argument; sol is normally the reservoir
%           solution structure.
%             -- Scalar fields:
%                  - mu  -- phase viscosities in units of Pa*s
%                  - rho -- phase densities in units of kilogram/meter^3
%                  - sw  -- residual water saturation for water
%                  - sr  -- residual phase saturation for CO2
%                  - kwm -- phase relative permeability at 'sr' and 'sw'
%
%             -- Function handles:
%                  - mob -- pseudo mobility
%                  - pc  -- second-order term in the transport equation
%                           ("capillary pressure" function); at the moment,
%                           only a linear function, pc(h)=h, is implemented
%                  - mob_avg -- average mobility used to compute timestep
%                               in transport equation.
%
% EXAMPLE:
%
% fluid = initVEFluidHForm(g_top, 'mu' , [0.1 0.4]*centi*poise, ...
%                                 'rho', [600 1000].* kilogram/meter^3, ...
%                                 'sr', 0.2, 'sw', 0.2, 'kwm', [1 1]);
%
% % Alternative: obtain same as above (except that it does not honour
% % residual trapping when computing mobilites) but done with tables:
%
% H = max(g_top.cells.H);
%
% fluid = initVEFluidHForm(g_top, 'mu', [0.1 0.4]*centi*poise, ...
%                                 'rho', [600 1000].* kilogram/meter^3,...
%                                 'sr', 0 , 's_w' 0,
%                                 'kwm', [   1,   1]);
%
% SEE ALSO:
%   `initFluid`, `initResSol`, `initWellSol`, `solveIncompFlow`.

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

% $Date: 2012-01-30 11:41:03 +0100 (Mon, 30 Jan 2012) $
% $Revision: 9020 $

opt = struct('mu' , [], 'rho', [], 'sr', [], ...
             'sw', [], 'kwm', [1, 1]);

opt = merge_options(opt, varargin{:});

n_mu = numel(opt.mu); n_rho = numel(opt.rho); n_kwm = numel(opt.kwm);
assert ((n_mu == 2) && (n_rho == 2) && (n_kwm == 2));
assert (~isempty(opt.sr) && ~isempty(opt.sw))

mu  = opt.mu;

mob = @(sol) [opt.kwm(1)*sol.h/mu(1), ...
              opt.kwm(2)*(g_top.cells.H-(sol.h + opt.sr*(sol.h_max-sol.h)))/mu(2)];

% used to compute time step in transport eq:
%mob_avg = @(sol, H) mob(struct('h', sol.h, 'h_max', sol.h));
%mob_avg = @(sol, H) [opt.kwm(1)*sol.h/mu(1), opt.kwm(2)*(H-sol.h)/mu(2)];

% "capillary pressure" function, at the moment only h
pc =  @(h) (h);


fluid = struct('pc',         pc,                  ...
               'mob',        mob,                 ...
               'mu',         mu,                  ...
               'rho',        opt.rho,             ...
               'res_gas',    opt.sr,              ...
               'res_water',  opt.sw,              ...
               'fluxInterp', 0,                   ...
               'kwm',        opt.kwm);