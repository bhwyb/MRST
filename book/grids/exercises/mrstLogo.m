load mrst-logo.mat;
p = .05 + .3*(K(G.cells.indexMap)-30)/600;
rock.perm = p.^3.*(1e-5)^2./(0.81*72*(1-p).^2);
rock.poro = p; clear p K;
figure('Position',[440 290 860 500]);
plotCellData(G,rock.poro,'EdgeAlpha',.1); view(74,74);

%%
mrstModule add libgeometry;
rad = @(x,y) sum(bsxfun(@minus,x(:,1:2),y).^2,2);
G = mcomputeGeometry(G);
pv = sum(poreVolume(G,rock));
W = addWell([],G,rock, find(rad(G.cells.centroids,[3.5,180.5])<0.6), ...
            'Type', 'rate','Val', -pv/(20*year));
W = addWell(W,G,rock, find(rad(G.cells.centroids,[3.5,134.5])<0.6), ...
           'Type', 'rate','Val', -pv/(20*year));
W = addWell(W,G,rock, find(rad(G.cells.centroids,[3.5,90.5])<0.6), ...
           'Type', 'rate','Val', -pv/(20*year));
W = addWell(W,G,rock, find(rad(G.cells.centroids,[3.5,22.5])<0.6), ...
           'Type', 'rate','Val', -pv/(20*year));
W = addWell(W,G,rock, find(rad(G.cells.centroids,[42.5,45.5])<0.6), ...
            'Type', 'rate','Val', pv/(15*year));
W = addWell(W,G,rock, find(rad(G.cells.centroids,[42.5,90.5])<0.6), ...
            'Type', 'rate','Val', pv/(15*year));
W = addWell(W,G,rock, find(rad(G.cells.centroids,[42.5,130.5])<0.6), ...
            'Type', 'rate','Val', pv/(15*year));
T = computeTrans(G, rock);

fluid   = initSingleFluid('mu' , 1*centi*poise, ...
                          'rho', 1000*kilogram/meter^3);
state = initState(G,W,100*barsa);
state = incompTPFA(state, G, T, fluid, 'wells', W);

%%
mrstModule add diagnostics
D = computeTOFandTracer(state,G,rock,'wells',W);
figure('Position',[440 290 860 500]);
plotTracerBlend(G, D.ipart, max(D.itracer, [], 2)), 
plotWell(G,W,'Color','k','FontSize',10); view(96,76)
plotGrid(G,'EdgeAlpha',.1,'FaceColor','none'); axis off

%%
mrstModule add coarsegrid streamlines
cla
plotCellData(G,D.ipart,p>1,'EdgeAlpha',.1,'EdgeColor','k','FaceAlpha',.7);
ip = find( (abs(G.cells.centroids(:,1)-15.5)<.6) & (p==1) & ...
   ((G.cells.centroids(:,2)<35) | (G.cells.centroids(:,2)>65)));
h=streamline(pollock(G,state,ip(1:1:end),'reverse',true)); 
set(h,'Color',.95*[1 1 1],'LineWidth',.5);
h=streamline(pollock(G,state,ip(1:1:end)));
set(h,'Color',.95*[1 1 1],'LineWidth',.5);
plotWell(G,W,'Color','k','FontSize',0,'height',12,'radius',2);
