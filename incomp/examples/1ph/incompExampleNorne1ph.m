%% Norne: Single-Phase Pressure Solver
% In the example, we will solve the single-phase pressure equation
%
% $$\nabla\cdot v = q, \qquad v=\textbf{--}\frac{K}{\mu}\nabla p,$$
%
% using the corner-point geometry from the Norne field.
%
% The purpose of this example is to demonstrate how the two-point flow solver
% can be applied to compute flow on a real grid model that has degenerate
% cell geometries and non-neighbouring connections arising from a large
% number of faults, eroded layers, pinch-outs, and inactive cells. A more
% thorough examination of the model is given in a
% <matlab:edit('showNorne.m') the Norne overview.>
%
% We will not use real data for rock parameters and well paths. Instead,
% synthetic permeability data are generated by (make-believe) geostatistics
% and vertical wells are placed quite haphazardly throughout the domain.
%
% Experimentation with this model is continued in
% <matlab:edit('incompNorne2ph.m') incompNorne2ph.m>, in which we solve a
% simplified two-phase model.

mrstModule add incomp

%% Read and process the model
% Before running this example, you should run the
% <matlab:edit('showNorne.m') showNorne> tutorial. We start by reading the
% model from a file in the Eclipse formate (GRDECL). As shown in this
% overview, the model has two components, of which we will only use the
% first one.
grdecl = fullfile(getDatasetPath('norne'), 'OPM.GRDECL');
grdecl = readGRDECL(grdecl);
usys   = getUnitSystem('METRIC');
grdecl = convertInputUnits(grdecl, usys);
G = processGRDECL(grdecl); clear grdecl;
G = computeGeometry(G(1));

%% Set rock and fluid data
% The permeability is lognormal and isotropic within nine distinct layers
% and is generated using our simplified 'geostatistics' function and then
% transformed to lay in the interval 200-2000 mD. The single fluid has
% density 1000 kg/m^3 and viscosity 1 cP.
gravity off
K          = logNormLayers(G.cartDims, rand(9,1), 'sigma', 1.5);
K          = 200 + (K-min(K))/(max(K)-min(K))*1800;
perm       = convertFrom(K(G.cells.indexMap), milli*darcy); clear K;
rock       = makeRock(G, perm, 1);
fluid      = initSingleFluid('mu' ,    1*centi*poise     , ...
                             'rho', 1000*kilogram/meter^3);

clf,
K = log10(rock.perm);
plotCellData(G,K,'EdgeColor','k','EdgeAlpha',.1);
axis off, view(15,60), 
cs = [200 400 700 1000 1500 2000];
caxis(log10([min(cs) max(cs)]*milli*darcy));
h=colorbarHist(K,caxis,'South',100); zoom(2.5),
set(h, 'XTick', log10(cs*milli*darcy), 'XTickLabel', num2str(round(cs)'));


%% Introduce wells
% The reservoir is produced using a set production wells controlled by
% bottom-hole pressure and rate-controlled injectors. Wells are described
% using a Peacemann model, giving an extra set of equations that need to be
% assembled. For simplicity, all wells are assumed to be vertical and are
% assigned using the logical (i,j) subindex.

% Plot grid outline
clf
subplot('position',[0.02 0.02 0.96 0.96]);
plotGrid(G,'FaceColor','none','EdgeAlpha',0.1);
axis tight off, zoom(1.1), view(-5,58)

% Set six vertical injectors, completed in each layer.
nz = G.cartDims(3);
I = [ 9, 26,  8, 25, 35,  10];
J = [14, 14, 35, 35, 68,  75];
R = [ 2,  2,  4,  4, 0.5,0.5]*1000*meter^3/day;
radius = .1; W = [];
for i = 1 : numel(I),
   W = verticalWell(W, G, rock, I(i), J(i), 1:nz, 'Type', 'rate', ...
                    'InnerProduct', 'ip_tpf', ...
                    'Val', R(i), 'Radius', radius, 'Comp_i', 1, ...
                    'name', ['I$_{', int2str(i), '}$']);
end
plotGrid(G, vertcat(W.cells), 'FaceColor', 'b');
prod_off = numel(W);

% Set five vertical producers, completed in each layer.
I = [17, 12, 25, 35, 15];
J = [23, 51, 51, 88, 94];
for i = 1 : numel(I),
   W = verticalWell(W, G, rock, I(i), J(i), 1:nz, 'Type', 'bhp', ...
                    'InnerProduct', 'ip_tpf', ...
                    'Val', 300*barsa(), 'Radius', radius, ...
                    'name', ['P$_{', int2str(i), '}$'], 'Comp_i', 0);
end
plotGrid(G, vertcat(W(prod_off + 1 : end).cells), 'FaceColor', 'r');
plotWell(G,W,'height',200);

%% Compute transmissibilities and solve linear system
% First, we compute one-sided transmissibilities for each local face of
% each cell in the grid. Then, we form a two-point discretization by
% harmonic averaging the one-sided transmissibilities and solve the
% resulting linear system to obtain pressures and fluxes.
T    = computeTrans(G, rock);
rSol = initState(G, W, 350*barsa, 1);
rSol = incompTPFA(rSol, G, T, fluid, 'wells', W);

%%
% Plot the fluxes
clf
cellNo  = rldecode(1:G.cells.num, diff(G.cells.facePos), 2) .';
plotCellData(G, sqrt(accumarray(cellNo,  ...
   abs(convertTo(faceFlux2cellFlux(G, rSol.flux), meter^3/day)))), ...
   'EdgeColor', 'k','EdgeAlpha',.1);
plotWell(G,W,'height',200,'color','c');
h=colorbar('horiz'); axis tight off; view(20,80)
zoom(1.9), camdolly(0,.15,0);
set(h,'Position',get(h,'Position')+[.18 -.08 -.1 0]);

%%
% Plot the pressure distribution
clf
plotCellData(G, convertTo(rSol.pressure(1:G.cells.num), barsa), ...
             'EdgeColor','k','EdgeAlpha',.1);
plotWell(G, W, 'height', 200, 'color', 'b');
h=colorbar('horiz'); axis tight off; view(20,80)
zoom(1.9), camdolly(0,.15,0);
set(h,'Position',get(h,'Position')+[.18 -.08 -.1 0]);

%%

% <html>
% <p><font size="-1">
% Copyright 2009-2015 SINTEF ICT, Applied Mathematics.
% </font></p>
% <p><font size="-1">
% This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).
% </font></p>
% <p><font size="-1">
% MRST is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% </font></p>
% <p><font size="-1">
% MRST is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% </font></p>
% <p><font size="-1">
% You should have received a copy of the GNU General Public License
% along with MRST.  If not, see
% <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses</a>.
% </font></p>
% </html>
