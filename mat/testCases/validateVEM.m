clc; clear; close all;

%   1) Validation of consistency for first order.
%   2) Validation of consistency for second order.
%   3) Point source problem, both orders.

i = 3;
switch i
    case 1

        tol= 1e-6;
        G = unitSquare([10,10],[1,1]);
        G = sortEdges(G);
        G = computeVEM2DGeometry(G);

        state = initState(G, [], 0);

        k = 1;
        
        K = rand(1,3)*1e-12;
        rock.perm = repmat(K, G.cells.num,1);
        mu = 10; rho = 1;
        fluid = initSingleFluid('mu', mu, 'rho', rho);

        tol = 1e-6;
        f = boundaryFaces(G);
        isNeu = abs(G.faces.centroids(f,1))<tol;
        
        gD = @(X) sum(X,2);
        bc = addBCFunc([], f(isNeu), 'flux', -(K(1) + K(2))/mu);
        bc = addBCFunc(bc, f(~isNeu), 'pressure', gD);
        
        tic;
        S = computeVirtualIP(G, rock, k);
        state = incompVEM(state, G, S, fluid, 'bc', bc);
        toc
        
        fprintf('\nError: %.2d\n\n', norm(state.nodePressure -  gD(G.nodes.coords)));
        
        plotVEM2D(G, state, k);

    case 2
    
        tol= 1e-6;
        G = unitSquare([10,10],[1,1]);
        G = sortEdges(G);
        G = computeVEM2DGeometry(G);

        state = initState(G, [], 0);
        
        k = 2;
        
        K = rand(1,3)*1e-12;
        
        mu = 100; rho = 1;
        gD = @(X) -(K(2) + K(3))/K(1)*X(:,1).^2 + X(:,1).*X(:,2) + X(:,2).^2;
        gN = @(X) -((2*X(:,2)-X(:,1))*K(2) - 2*K(3)*X(:,1) + K(1)*X(:,2))/mu;
        
        
        rock.perm = repmat(K, G.cells.num,1);
        fluid = initSingleFluid('mu', mu, 'rho', 1);

        tol = 1e-6;
        f = boundaryFaces(G);
        isNeu = abs(G.faces.centroids(f,1))<tol;
        
        bc = addBCFunc([], f(isNeu), 'flux', gN);
        bc = addBCFunc(bc, f(~isNeu), 'pressure', gD);
        
        tic;
        S = computeVirtualIP(G, rock, k);
        state = incompVEM(state, G, S, fluid, 'bc', bc);
        toc
        
        fprintf('\nError: %.2d\n\n', norm(state.nodePressure -  gD(G.nodes.coords)));
        
        plotVEM2D(G, state, k);
        
    case 3
        
        tol= 1e-6;
%         G = voronoiCube(10,[1,1,1])
        G = cartGrid([2,2,2], [1,1,1]);
        G = computeVEM3DGeometry(G);
        
        k = 1;
        
%         K = [1,0,0,0,0,1];
        
        mu = 100; rho = 1;
%         gD = @(X) -(K(2) + K(3))/K(1)*X(:,1).^2 + X(:,1).*X(:,2) + X(:,2).^2;
%         gN = @(X) -((2*X(:,2)-X(:,1))*K(2) - 2*K(3)*X(:,1) + K(1)*X(:,2))/mu;
        
        rock.perm = repmat(1, G.cells.num,1);
        fluid = initSingleFluid('mu', mu, 'rho', 1);

        tol = 1e-6;
        f = boundaryFaces(G);
        isNeu = abs(G.faces.centroids(f,1))<tol;
%         
%         bc = addBCFunc([], f(isNeu), 'flux', gN);
%         bc = addBCFunc(bc, f(~isNeu), 'pressure', gD);
        bc = addBC([], f, 'pressure', 0);
        tic;
        S = computeVirtualIP(G, rock, k);
        state = incompVEM(state, G, S, fluid, 'bc', bc);
        toc
        
        fprintf('\nError: %.2d\n\n', norm(state.nodePressure -  gD(G.nodes.coords)));
        
        plotVEM2D(G, state, k);
        
end

