%% 
% This 1D 
%
% data from deck
%

try
    require ad-core ad-blackoil ad-eor ad-props deckformat mrst-gui
catch
    mrstModule add ad-core ad-blackoil ad-eor ad-props deckformat mrst-gui
end

current_dir = fileparts(mfilename('fullpath'));

fn = fullfile(current_dir, 'SURFACTANT1D.DATA');
gravity off

deck = readEclipseDeck(fn);
deck = convertDeckUnits(deck);

fluid = initDeckADIFluid(deck);

G = initEclipseGrid(deck);
G = computeGeometry(G);

rock  = initEclipseRock(deck);
rock  = compressRock(rock, G.cells.indexMap);



%% Set up simulation parameters
% We want a layer of oil on top of the reservoir and water on the bottom.
% To do this, we alter the initial state based on the logical height of
% each cell. The resulting oil concentration is then plotted.


nc = G.cells.num;
state0 = initResSol(G, 300*barsa, [ .2, .8]);

% Add zero surfactant concentration to the state.
state0.c    = zeros(G.cells.num, 1);
state0.cmax = state0.c;


%% Set up systems.

modelSurfactant = FullyImplicitOilWaterSurfactantModel(G, rock, fluid, ...
                                                  'inputdata', deck, ...
                                                  'extraStateOutput', true);

% Convert the deck schedule into a MRST schedule by parsing the wells
schedule = convertDeckScheduleToMRST(modelSurfactant, deck);


%% Run the schedule
% Once a system has been created it is trivial to run the schedule. Any
% options such as maximum non-linear iterations and tolerance can be set in
% the system struct.

state0.ads = computeEffAds(state0.c, 0, modelSurfactant.fluid);
state0.adsmax = state0.ads;

resulthandler = ResultHandler('dataDirectory', pwd, 'dataFolder', 'cache', 'cleardir', true);
[wellSolsSurfactant, statesSurfactant] = simulateScheduleAD(state0, modelSurfactant, ...
                                                  schedule, 'OutputHandler', ...
                                                  resulthandler);

figure()
plotToolbar(G, statesSurfactant, 'startplayback', true, 'plot1d', true)
