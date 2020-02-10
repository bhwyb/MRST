mrstModule add ad-core
%% Conceptual example
group = StateFunctionGrouping('mystorage'); % Store in state.mystorage
% Set the numbers
group = group.setStateFunction('A', TutorialNumber('a'));
group = group.setStateFunction('B', TutorialNumber('b'));
group = group.setStateFunction('X', TutorialNumber('x'));
group = group.setStateFunction('Y', TutorialNumber('y'));
% Create products
group = group.setStateFunction('AB', TutorialProduct('A', 'B'));
group = group.setStateFunction('XY', TutorialProduct('X', 'Y'));
% Create the final sum
group = group.setStateFunction('G', TutorialAdd('AB', 'XY'));
disp(group)
%% Plotting
clf;
[h, g] = plotStateFunctionGroupings(group, 'TextArg', {'FontSize', 16, 'horizontalalignment', 'center', 'verticalalignment', 'middle'});
colormap(brighten(lines, 0.5))
printStateFunctionGroupingTikz(g)
%% Show evaluation
clc
state0 = struct('a', 3, 'b', 5, 'x', 7, 'y', 2); % Initial state
state = group.initStateFunctionContainer(state0); % Enable caching
disp(state.mystorage)

disp('First call:')
G = group.get([], state, 'G'); % First call, triggers execution
disp(state.mystorage)
disp('Second call:')
G = group.get([], state, 'G') % Second call, cached
%% Partial caching
clc
state = group.initStateFunctionContainer(state0);
xy = group.get([], state, 'XY'); % First call, triggers execution
G = group.get([], state, 'G'); % Second call, cached
%% Dependency checking
mrstVerbose on; % Print missing dependencies
group.checkDependencies();
%% Add an unmet dependency
group = group.setStateFunction('F', TutorialProduct('X', 'Z'));
ok = group.checkDependencies();
%% Set up a compositional test model
mrstModule add ad-props compositional
G = cartGrid([200, 200, 1]);
G = computeGeometry(G);
cmodel = GenericOverallCompositionModel(G, makeRock(G, 1, 1), initSimpleADIFluid(), getCompositionalFluidCase('spe5'));
cmodel.AutoDiffBackend = DiagonalAutoDiffBackend('useMex', true);
cmodel = cmodel.validateModel();
%% Repeated calls will be faster, as values are acached
statec = initCompositionalState(G, 10*barsa,  273.15 + 30, [0.3, 0.4, 0.3], rand(1, 6), cmodel.EOSModel);
stateAD = cmodel.validateState(statec);
stateAD = cmodel.getStateAD(stateAD);

fprintf('First evaluation: '); tic(); rho = cmodel.getProp(stateAD, 'Density'); toc();
fprintf('Second evaluation: '); tic(); rho = cmodel.getProp(stateAD, 'Density'); toc();
%% Set up a black-oil model
pth = getDatasetPath('spe1');
fn  = fullfile(pth, 'BENCH_SPE1.DATA');
% deck = readEclipseDeck(fn);
[state0, bomodel, schedule, nonlinear] = initEclipseProblemAD(fn);
bomodel = bomodel.validateModel();

%% Plot density relations for a black-oil model
figure;
[h, gbo] = plotStateFunctionGroupings(bomodel, 'Stop', 'Density', 'includeState', false);
clc
printStateFunctionGroupingTikz(gbo);

%% Plot density relations for a compositional model
figure;
[~, gcomp] = plotStateFunctionGroupings(cmodel,'Stop', 'Density', 'includeState', false);
clc
printStateFunctionGroupingTikz(gcomp);

%% Plot all functions
figure;
[~, gbos] = plotStateFunctionGroupings(bomodel, 'includeState', false);
clc
printStateFunctionGroupingTikz(gbos);
%% Plot all functions (as symbols)
figure;
[h2, gbos2, g] = plotStateFunctionGroupings(bomodel, 'includeState', false, 'label', 'label');
clc
printStateFunctionGroupingTikz(gbos2, h2);