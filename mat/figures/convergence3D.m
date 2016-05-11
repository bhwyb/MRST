clc; clear all; close all;

addpath('../../../pebiGridding/voronoi3D/')
addpath('../VEM3D/')
addpath('../')

%%  PROBELM

% % not ok
% f  = @(X) 4*pi^2*X(:,1).*sin(2*pi*X(:,3));
% gD = @(X) X(:,1).*sin(2*pi*X(:,3));
% gN = @(X) 2*pi*X(:,1).*cos(2*pi*X(:,3));

% % ok
% f  =      0;
% gD = @(X) X(:,1).^2- X(:,3).^2;
% gN = @(X) -2*X(:,3);

% % ok
% f  = @(X) -30*X(:,3).^4;
% gD = @(X) X(:,3).^6;
% gN = @(X) 6*X(:,3).^5;

% % semi ok
% f  = @(X) -X(:,1).*exp(X(:,2).*X(:,3)).*(X(:,2).^2+X(:,3).^2);
% gD = @(X) X(:,1).*exp(X(:,2).*X(:,3));
% gN = @(X) X(:,1).*X(:,2).*exp(X(:,2));

% semi ok
f  = @(X) -X(:,2).*exp( X(:,3) ).*( X(:,1).^2 + 2 );
gD = @(X) X(:,1).^2.*X(:,2).*exp( X(:,3) );
gN = @(X) X(:,1).^2.*X(:,2);


%%  GRID DIMENSIONS

% nVec = [400, 800, 1600, 3200].^(1/3);
nVec = [5,6,8,10];
nIt = numel(nVec);
errVec = zeros(nIt, 3);
err2 = zeros(nIt,1);
azel = [150,30];

dest = './conv3D/';

for i = 1:nIt

    %% GENERATE GRID
    
    fprintf('Generating grid ...\n')
    tic;
    n = nVec(i);
    G = voronoiCubeRegular([n,n,n],[1,1,1],.5);
    fprintf('Done in %f seconds.\n\n', toc);
    
    G = computeVEM3DGeometry(G);
    
    %%  SET BC
    
    boundaryEdges = find(any(G.faces.neighbors == 0,2));
    tol = 1e-10;
    isNeu = abs(G.faces.centroids(boundaryEdges,3)-1) < tol;
    bc = VEM3D_addBC([], boundaryEdges(~isNeu), 'pressure', gD);
    bc = VEM3D_addBC(bc, boundaryEdges(isNeu) , 'flux'    , gN);

%     bc = VEM3D_addBC([], boundaryEdges, 'pressure', gD);

    %%  GRID DATA
    
    h = max(G.cells.diameters);
    area = sqrt(sum(G.cells.volumes.^2));
    nK = G.cells.num;

    %%  CALCULATE SOLUTIONS, PLOT GRID AND SOLUTIONS

    Kc = G.cells.centroids;
    cells = 1:G.cells.num;
    r = .8; c = [1,1,0];
    cells = cells(sum(bsxfun(@minus, Kc, c).^2,2) > r^2);
    faceNum = mcolon(G.cells.facePos(cells),G.cells.facePos(cells+1)-1);
    faces = G.cells.faces(faceNum);
    
    remCells = 1:G.cells.num;
    remCells = remCells(~ismember(remCells, cells));
    
    GP = removeCells(G,remCells);
    outerFaces = any(GP.faces.neighbors == 0,2);
    
    outerFaces = find(ismember(G.faces.centroids,GP.faces.centroids(outerFaces,:),'rows'));
    
    clear GP Kc;
    
    %   Grid
    
    gridFig = figure;
    set(gridFig, 'visible','off')
    plotGrid(G, cells, 'facecolor', [238,232,170]/255);
    set(gridFig,'DefaultTextInterpreter', 'LaTex');
    set(gca, 'XTick', [0,1]);
    set(gca, 'YTick', [0,1]);
    xlabel('$x$'); ylabel('$y$');
    view(azel)
    axis equal off;

    fileName = strcat('../../tex/thesis/fig/Grid3D_', num2str(i));
    savePdf(gridFig, fileName);
    clear gridFig;
    
    %   1st order solution
    
    [sVEM1, G] = VEM3D(G,f,bc,1,'cellProjectors', true);
    l2Err1 = l2Error3D(G, sVEM1, gD, 1);
    err2(i) = h^(3/2)*norm(sVEM1.nodeValues-gD(G.nodes.coords),2);
    
    
    if i == 1 || i == nIt
        
        sol1Fig = figure;
        set(sol1Fig, 'visible','off')
        sVEM1 = calculateFaceAverages(G,sVEM1);
        plotFaces(G,outerFaces,sVEM1.faceMoments(outerFaces));
        set(sol1Fig, 'DefaultTextInterpreter', 'LaTex');
        colorbar;
        view(azel);
        axis equal;
        set(gca, 'XTick', [0,.5,1]);
        set(gca, 'YTick', [0,.5,1]);
        set(gca, 'ZTick', [0,.5,1]);
        xlabel('$x$'); ylabel('$y$'); zlabel('$z$');
        colorbar

        fileName = strcat('../../tex/thesis/fig/Sol3D1_', num2str(i));
        savePdf(sol1Fig, fileName);
        clear sol1Fig;

    end
    
    clear sVEM1;

    %   2nd order solution
     
    [sVEM2, G] = VEM3D(G,f,bc,2,'cellProjectors', true); 
    l2Err2 = l2Error3D(G, sVEM2, gD, 2);
    
    if i == 1 || i == nIt

        sol2Fig = figure;
        set(sol2Fig, 'visible','off')
        plotFaces(G,outerFaces,sVEM2.faceMoments(outerFaces));
        set(sol2Fig,'DefaultTextInterpreter', 'LaTex');
        colorbar;
        set(gca, 'XTick', [0,.5,1]);
        set(gca, 'YTick', [0,.5,1]);
        set(gca, 'ZTick', [0,.5,1]);
        xlabel('$x$'); ylabel('$y$'); zlabel('$z$');
        view(azel)
        axis equal
        colorbar;

        fileName = strcat('../../tex/thesis/fig/Sol3D2_', num2str(i));
        savePdf(sol2Fig, fileName);
        clear sol2Fig;

    end
    
    clear sVEM2;
    

    
    errVec(i,:) = [h, sqrt(sum(l2Err1)), sqrt(sum(l2Err2))];

    clear G l2Err1 l2Err2 isNeu cells boundaryEdges faceNum faces outerFaces remCells;
    close all;
    
end

%%  PLOT CONVERGENCE RATES

%   1st order convergence

conv1Fig = figure;
set(conv1Fig,'DefaultTextInterpreter', 'LaTex');
loglog(errVec(:,1), errVec(:,2), '-s');
hold on
loglog(errVec(:,1),(errVec(:,1)./errVec(end,1)).^2*errVec(end,2)*.8)
p1 = polyfit(log(errVec(:,1)), log(errVec(:,2)),1);
lStr = strcat('Slope = ', num2str(p1(1), '%.3f'));
h = legend(lStr, 'Slope =2.0');
set(h,'Interpreter','latex');
xlabel('$\log(h)$'); ylabel('$\log\left(\left\|u-\Pi^\nabla u_h\right\|_{0,\Omega}\right)$');
axis equal;

%%

fileName = strcat('../../tex/thesis/fig/Conv3D1');
% savePdf(conv1Fig, fileName);
cut = 4;
h = conv1Fig;
ps = get(h, 'Position');
ratio = (ps(4)-ps(2)) / (ps(3)-ps(1));
paperWidth = 10;
paperHeight = paperWidth*ratio - cut;
set(h, 'paperunits', 'centimeters');
set(h, 'papersize', [paperWidth paperHeight]);
set(h, 'PaperPosition', [0    0   paperWidth paperHeight]);

print(h, '-dpdf', fileName);

%%

%   2nd order convergence

conv2Fig = figure;
set(conv2Fig,'DefaultTextInterpreter', 'LaTex');
loglog(errVec(:,1), errVec(:,3), '-s');
hold on
loglog(errVec(:,1),(errVec(:,1)./errVec(end,1)).^3*errVec(end,3)*.6)
p2 = polyfit(log(errVec(:,1)), log(errVec(:,3)),1);
lStr = strcat('Slope =', num2str(p2(1), '%.3f'));
h = legend(lStr, 'Slope =3.0');
set(h,'Interpreter','latex');
xlabel('$\log(h)$'); ylabel('$\log\left(\left\|u-\Pi^\nabla u_h\right\|_{0,\Omega}\right)$');
axis equal

%%

fileName = strcat('../../tex/thesis/fig/Conv3D2');
% savePdf(conv2Fig, fileName, 'cut', 0);
h = conv2Fig;
cut = 4;
ps = get(h, 'Position');
ratio = (ps(4)-ps(2)) / (ps(3)-ps(1));
paperWidth = 10;
paperHeight = paperWidth*ratio - cut;
set(h, 'paperunits', 'centimeters');
set(h, 'papersize', [paperWidth paperHeight]);
set(h, 'PaperPosition', [0    0   paperWidth paperHeight]);

print(h, '-dpdf', fileName);