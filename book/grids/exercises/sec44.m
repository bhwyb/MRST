%% Extend triangleGrid to triangulated surfaces
load trimesh2d;
G = computeGeometry(myTriangleGrid([xfe 0*xfe yfe], trife));
g = computeGeometry(triangleGrid([xfe yfe],trife));

clf
subplot(2,2,1), 
plot(G.cells.volumes, g.cells.volumes,'o');
v = [G.cells.volumes; g.cells.volumes];
hold on
plot([min(v) max(v)], [min(v) max(v)],'r-');
hold off
axis tight

subplot(2,2,3), plot(G.faces.areas, g.faces.areas, 'o');
a = [G.faces.areas; g.faces.areas];
hold on
plot([min(a) max(a)], [min(a) max(a)],'r-');
hold off, axis tight

subplot(2,2,[2 4]),
plotGrid(G); view(3);
hold on,
c = G.cells.centroids;
plot3(c(:,1),c(:,2),c(:,3),'.r');
f = G.faces.centroids;
plot3(f(:,1),f(:,2),f(:,3),'.b');
hold off
view(0,0); axis([0 100 0 1 0 100]);