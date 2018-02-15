mrstModule add dg vem vemmech ad-props ad-core ad-blackoil blackoil-sequential

%%

n = 100;
G = computeGeometry(cartGrid([n,1], [100,10]));
G.nodes.coords = G.nodes.coords;
G = computeVEMGeometry(G);

rock = makeRock(G, 100*milli*darcy, 0.4);
fluid = initSimpleADIFluid('phases', 'WO', ...
                           'rho', [1000, 800]*kilogram/meter^3, ...
                           'mu', [0.3, 1]*centi*poise);
                       
modelfi = TwoPhaseOilWaterModel(G, rock, fluid);
modelFV = getSequentialModelFromFI(modelfi);
modelDG = modelFV;
modelDG.transportModel = TransportOilWaterModelDG(G, rock, fluid, 'degree', 0);
                       
%%

time = 2*year;
rate = 1*sum(poreVolume(G, rock))/time;
W = [];
W = addWell(W, G, rock, 1, 'type', 'rate', 'val', rate, 'comp_i', [1,0]);
W = addWell(W, G, rock, G.cells.num, 'type', 'bhp', 'val', 50*barsa, 'comp_i', [1,0]);


dt = 30*day;
dtvec = rampupTimesteps(time, dt, 0);

schedule = simpleSchedule(dtvec, 'W', W);

%%

state0      = initResSol(G, 100*barsa, [0,1]);
nDof        = modelDG.transportModel.basis.nDof;
state0.sdof = zeros(G.cells.num*nDof, 2);
state0.sdof(1:nDof:G.cells.num*nDof,2) = 1;

[ws, state, rep] = simulateScheduleAD(state0, modelDG, schedule);

%%

[ws2, state2, rep2] = simulateScheduleAD(state0, modelFV, schedule);

%%

figure
x = linspace(0,100,n);

steps = [1:4:25];
clr = lines(numel(steps));
h = [];
for sNo = 1:numel(steps)
    hold on
    h(sNo) = plot(x, state {steps(sNo)}.s(:,1), '--', 'linew', 4, 'color', clr(sNo, :));
             plot(x, state2{steps(sNo)}.s(:,1), '-', 'color', clr(sNo,:));
end

lgnd = cellfun(@(ts) ['Timestep', num2str(ts)], mat2cell(steps, 1, ones(1,numel(steps))), 'unif', false);
legend(h, lgnd)

%%

figure;
plotToolbar(G, state, 'plot1d', true);

figure
plotToolbar(G, state2, 'plot1d', true);