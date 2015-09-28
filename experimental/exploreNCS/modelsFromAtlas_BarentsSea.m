%% Grid Models from the CO2 Storage Atlas
% We show how to develop volumetric models of sand bodies based on the
% datasets from the CO2 Storage Atlas. At the end of the example, we
% analyse the different formations and compute the capacity for structural
% trapping in each formation. 
%
% The datasets are used to create interpolants which give both height and
% thickness on a fine grid. Any regions where the thickness/height is zero
% or not defined is removed, giving a GRDECL file defining the intersection
% of these datasets.

try
   require co2lab
catch %#ok<CTCH>
   mrstModule add co2lab
end

%% Example: Bjarmeland formation
% The formations discussed in the atlas cover large areas, and even though
% the spatial resolution in the geographical data is low compared with
% simulation models used, e.g., to simulate petroleum reservoirs, the
% resulting grids can be fairly large. A full grid realization of the
% Utsira will contain around 100k cells in the lateral direction. By
% coarsening the grid by a factor two in each of the lateral directions, we
% end  up with the more managable amount of 25k cells. The parameter 'nz'
% determines the number of fine cells in the logical k-direction and can be
% used to produce layered models for full simulations.

gr = getAtlasGrid('Bjarmelandfm', 'coarsening', 2, 'nz', 1);

% Process the grid
G = processGRDECL(gr{1});

% The coarsening may cause parts of the region to be disconnected. Here,  we
% only consider the first and largest grid produced by processGRDECL and
% add geometry data (cell volumes, face normals, etc) to it.
G = computeGeometry(G(1));

% We plot the full grid, colorized by cell volumes. Light is added to the
% scene to better reveal reliefs in the top surface of the reservoir.
% These folds can be a target for structural trapping of migrating CO2.
clf;
plotCellData(G, G.cells.volumes, 'EdgeColor','none')
title('Bjarmeland formation - full grid with thickness and heightmap')
axis tight off
light('Position',[-1 -1 -1],'Style','infinite');
lighting phong
view(90,45)

%% Full 3D visualization of all grids in North Sea
% Not all formations in the data set supply both a height map of the top
% surface and a map of the formation thickness, which are both needed to
% construct a volumetric sandbody. Next, we visualize the different
% sandbodies that can be constructed based on pairs of depth and thickness
% maps. Because the grids are large, we will use functionality from the
% 'opm_gridprocessing' and 'libgeometry' mex-modules to speed up the
% processing.

% Load module
moduleCheck('libgeometry','opm_gridprocessing'); 

% Count number of sand bodies
grdecls = getAtlasGrid(getNorwegianSeaNames(),'coarsening',10);
ng = numel(grdecls);

% Set view angles
viewMat = [-110 55; -110 55; -110 55; -110 55; -110 55; -110 55];

% Loop through and visualize all the 3D models at full resolution
for i=1:ng;
   grdecl = getAtlasGrid(grdecls{i}.name, 'coarsening', 1);
   G = processgrid(grdecl{1});   % G = processGRDECL(grdecl{1});
   G = mcomputeGeometry(G(1));   % G = computeGeometry(G(1));
   %clf;
   figure;
   plotGrid(G,'FaceColor', [1 .9 .9], 'EdgeAlpha', .05);
   view(viewMat(i,:)); axis tight off
   light('Position',[-1 -1 1],'Style','infinite');lighting phong
   title(grdecls{i}.name)
   drawnow
end

%% Basic capacity estimates for the CO2 atlas data (Barents Sea)
% Finally, we compute the potential volumes available for structural
% trapping. To better report progress, we first load a low resolution
% version to get names of all aquifers. Then, we load and process the
% full-resolution versions to compute the geometrical volume inside all
% structural traps under each top surface
res = cell(ng,1);
fprintf('------------------------------------------------\n');
for i=1:ng
   fprintf('Processing %s ... ', grdecls{i}.name);
   grdecl  = getAtlasGrid(grdecls{i}.name, 'coarsening', 1);
   G       = mprocessGRDECL(grdecl{1});
   G       = mcomputeGeometry(G(1));
   Gt      = topSurfaceGrid(G);
   ta      = trapAnalysis(Gt, false);
   
   res{i}.name      = grdecls{i}.name;
   res{i}.cells     = Gt.cells.num;
   res{i}.zmin      = min(Gt.cells.z);
   res{i}.zmax      = max(Gt.cells.z);
   res{i}.volume    = sum(G.cells.volumes);
   res{i}.trapvols  = volumesOfTraps(Gt,ta);
   res{i}.capacity  = sum(res{i}.trapvols);
   fprintf('done\n');
end

%%
% Show table of volumes
fprintf('\n\nOverview of trapping capacity:\n')
fprintf('\n%-13s| Cells  | Min  | Max  | Volume   | Capacity  | Percent\n', 'Name');
fprintf('-------------|--------|------|------|----------|-----------|-----------\n');
for i=1:ng
   fprintf('%-13s| %6d | %4.0f | %4.0f | %4.2e |  %4.2e | %5.2f \n',...
      res{i}.name, res{i}.cells, res{i}.zmin, res{i}.zmax, res{i}.volume, ...
      res{i}.capacity, res{i}.capacity/res{i}.volume*100);
end
fprintf('-------------|--------|------|------|----------|-----------|-----------\n\n');
