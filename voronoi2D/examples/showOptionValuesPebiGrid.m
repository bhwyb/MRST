%% Example
% In this example we show all optional options for pebiGrid.

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2016 Runar Lie Berge. See COPYRIGHT.TXT for details.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}  

%% wellLines
% This sets the wells in our reservoir. The wells are stored as a cell
% array, each element corresponding to one well
w = {[0.2,0.8;0.5,0.7;0.8,0.8],...
     [0.5,0.2]};
gS = 0.1;
pdims=[1,1];
G = pebiGrid(gS, pdims,'wellLines',w);

plotGrid(G);
hold on
plotGrid(G,G.cells.tag,'facecolor','b')
title('Two wells')
axis equal
%% wellGridFactor
% The wellGridFactor sets the relative size of the well cells. If
% wellGridFactor=0.5 the well cells will have about half the size of the
% reservoir sites:
w = {[0.2,0.3;0.8,0.7]};
gS = 0.1;
pdims=[1,1];
G1 = pebiGrid(gS, pdims,'wellLines',w,'wellGridFactor',1);
G2 = pebiGrid(gS, pdims,'wellLines',w,'wellGridFactor',1/2);

figure()
plotGrid(G1)
plotGrid(G1,G1.cells.tag,'faceColor','b')
title('wellGridFactor=1')
axis equal
figure()
plotGrid(G2)
plotGrid(G2,G2.cells.tag,'faceColor','b')
title('wellGridFactor=1/2')
axis equal




%% wellRefinement
% wellRefinement is a logical parameter which is set to true if we whish
% the grid to be refined towards the wells.
w = {[0.2,0.3;0.8,0.7]};
gS = 0.1;
pdims=[1,1];
G = pebiGrid(gS, pdims,'wellLines',w,'wellGridFactor',1/4,'wellRefinement',true);

figure()
plotGrid(G)
plotGrid(G,G.cells.tag,'faceColor','b')
title('wellRefinement')
axis equal


%% wellEps
% wellEps controlls the refinement towards the wells. The cell sizes are
% increasing exponentially away from the wells: exp(dist(x,well)/wellEps).
% Notice that you have to scale wellEps to your reservoir size, or distMesh
% might take a very long time to converge.
w = {[0.2,0.3;0.8,0.7]};
gS = 0.1;
pdims=[1,1];
eps1 = 1/5;
eps2 = 1/2;
G1 = pebiGrid(gS, pdims,'wellLines',w,'wellGridFactor',1/4, ...
              'wellRefinement',true,'wellEps',eps1);
G2 = pebiGrid(gS, pdims,'wellLines',w,'wellGridFactor',1/4, ...
              'wellRefinement',true,'wellEps',eps2);

figure()
plotGrid(G1)
plotGrid(G1,G1.cells.tag,'faceColor','b')
title('wellEps = 1/5')
axis equal

figure()
plotGrid(G2)
plotGrid(G2,G2.cells.tag,'faceColor','b')
title('wellEps = 1/2')
axis equal


%% ProtLayer
% Adds a protection layer around wells. The protection sites are place
% normal along the well path
x = linspace(0.2,0.8);
y = 0.5+0.1*sin(pi*x);
w = {[x',y']};
gS = 0.1;
pdims=[1,1];
G = pebiGrid(gS, pdims,'wellLines',w,'protLayer',true);

figure()
plotGrid(G);
hold on
plotGrid(G,G.cells.tag,'facecolor','b')
title('Protection Layer on')
axis equal

%% protD
% This parameter sets the distance from the protection sites to the well
% path. This is cellarray of functions, one for each well path.
w = {[0.2,0.8;0.8,0.8],...
     [0.5,0.2;0.1,0.5]};
gS = 0.1;
pdims=[1,1];
protD = {@(p)0.01+0.1*p(:,1), @(p) 0.01*ones(size(p,1),1)};
G = pebiGrid(gS, pdims,'wellLines',w,'protLayer',true,'protD',protD);

figure()
plotGrid(G);
hold on
plotGrid(G,G.cells.tag,'facecolor','b')
title('Specifying Protection Distance')
axis equal



%% faultLines
% This sets the faults in our reservoir. The wells are stored as a cell
% array, each element corresponding to one fault
f = {[0.2,0.8;0.5,0.7;0.8,0.8],...
     [0.5,0.2;0.1,0.5]};
gS = 0.1;
pdims=[1,1];
figure()
G = pebiGrid(gS, pdims,'faultLines',f);
plotGrid(G);
hold on
plotFaces(G,G.faces.tag,'edgeColor','r')
title('Fault Lines')
axis equal


%% faultGridFactor
% The faultGridFactor sets the relative distance between the fault cells 
% along the fault paths. If fautlGridFactor=0.5 the fault cells will be 
% have spaceing about half the size of the reservoir cells:
f = {[0.2,0.3;0.8,0.7]};
gS = 0.1;
pdims=[1,1];
G1 = pebiGrid(gS, pdims,'faultLines',f,'faultGridFactor',1);
G2 = pebiGrid(gS, pdims,'faultLines',f,'faultGridFactor',1/2);

figure()
plotGrid(G1)
title('faultGridFactor=1')
axis equal
figure()
plotGrid(G2)
title('faultGridFactor=1/2')
axis equal

%% circleFactor
% The fault sites are generated by setting placing a set of circles along
% the fault path with a distance gS*faultGridFactor. The radius of the
% circles are gS*faultGridFactor*circleFactor. The fault sites are place
% at the intersection of these circles. Notice that in the second example,
% the fault is not traced by edges in the grid. This is because the
% distance between fault sites becomes so large that distmesh managed to
% force reservoir sites between the fault sites. 
f = {[0.2,0.3;0.8,0.7]};
gS = 0.1;
pdims=[1,1];
G1 = pebiGrid(gS, pdims,'faultLines',f,'circleFactor',0.55);
G2 = pebiGrid(gS, pdims,'faultLines',f,'circleFactor',0.9);

figure()
plotGrid(G1)
title('circleFactor=0.55')
axis equal
figure()
plotGrid(G2)
title('circleFactor=0.9')
axis equal

%% faultRho
% A function that sets the relative size of the fault sites in the domain.
f = {[0.2,0.3;0.8,0.7]};
gS = 0.1;
pdims=[1,1];
faultRho = @(p) 1 - 0.9*p(:,1);
G = pebiGrid(gS, pdims,'faultLines',f,'faultRho',faultRho);

figure()
plotGrid(G)
title('faultRho=@(p) 1 - 0.9*p(:,1)')
axis equal


%% faultRefinement
% faultRefinement is a logical parameter which is set to true if we whish
% the grid to be refined towards the fault. NOTE: faultRefinement is not
% thoroughly together with wellrefinement! 
f = {[0.2,0.3;0.8,0.7]};
gS = 0.1;
pdims=[1,1];
G = pebiGrid(gS, pdims,'faultLines',f,'faultGridFactor',1/4,'faultRefinement',true);

figure()
plotGrid(G)
title('Fault Refinement')
axis equal


%% faultEps
% faultEps controlls the refinement towards the faults. The cell sizes are
% increasing exponentially away from the faults: exp(dist(x,fault)/faultEps).
% Notice that you have to scale faultEps to your reservoir size, or distMesh
% might take a very long time to converge.
f = {[0.2,0.3;0.8,0.7]};
gS = 0.1;
pdims=[1,1];
eps1 = 1/5;
eps2 = 1/2;
G1 = pebiGrid(gS, pdims,'faultLines',f,'faultGridFactor',1/4, ...
              'faultRefinement',true,'faultEps',eps1);
G2 = pebiGrid(gS, pdims,'faultLines',f,'faultGridFactor',1/4, ...
              'faultRefinement',true,'faultEps',eps2);


figure()
plotGrid(G1)
title('faultEps = 1/5')
axis equal

figure()
plotGrid(G2)
title('faultEps = 1/2')
axis equal