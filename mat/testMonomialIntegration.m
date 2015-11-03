clc; close all;

nx = 3; ny = 3;

dx = 1/(nx-1); dy = 1/(ny-1);
[x, y] = meshgrid(0:dx:1, 0:dy:1);

P = [x(:), y(:)];
t = delaunayn(P);

G = triangleGrid(P,t);
G = computeGeometry(G);
G = mrstGridWithFullMappings(G);

plotGrid(G);
K = 2;
[~, X] = nodeData(G,K);
[~, Xmid, normals] = faceData(G,K);

[~, Xb] = baric(X);
xK = Xb(1); yK = Xb(2); hK = cellDiameter(X);

m_int = @(X)  [(X(:,1)-xK).^2./(2*hK) , ...
               (X(:,1)-xK).*(X(:,2)-yK)./hK , ...
               (X(:,1)-xK).^3./(3*hK^2) , ...
               (X(:,1)-xK).^2.*(X(:,2)-yK)./(2*hK^2) , ...
               (X(:,1)-xK).*(X(:,2)-yK).^2./(hK^2)];

I = evaluateMonomialIntegralV2(normals, X, Xmid, m_int);


% hK = sqrt(2);
% X = [2,2;3,2;3,3;2,3];
% [~, XB] = baric(X);
% vol = 1;
% m =      @(X) 1/hK.*[ X(:,1) - XB(:,1),                ...                             %   (1,0)
%                       X(:,2) - XB(:,2),                ...                             %   (0,1)
%                      (X(:,1) - XB(:,1)).^2./hK,        ...                   %   (2,0)
%                      (X(:,1) - XB(:,1)).*(X(:,2)- XB(:,2))./hK, ...         %   (1,1)
%                      (X(:,2) - XB(:,2)).^2./hK];                       %   (0,2)
% 
% I = polygonInt(X,m)/vol;
% Iex = 1/24.*[0, 0, 1, 0, 1];
% I - Iex
% 
% hK = 5;
% X = [0,0;3,0;3,2;3/2,4;0,4];
% [~, XB] = baric(X);
% vol = 21/2;
% m =      @(X) 1/hK.*[ X(:,1) - XB(:,1),                ...                             %   (1,0)
%                       X(:,2) - XB(:,2),                ...                             %   (0,1)
%                      (X(:,1) - XB(:,1)).^2./hK,        ...                   %   (2,0)
%                      (X(:,1) - XB(:,1)).*(X(:,2)- XB(:,2))./hK, ...         %   (1,1)
%                      (X(:,2) - XB(:,2)).^2./hK];                       %   (0,2)
% 
% I = polygonInt(X,m)/vol;
% Iex = 1/176400.*[0, 0 4770, -1452, 8480];
% I - Iex
