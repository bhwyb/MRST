%% Ensemble of GPSNET models 
% GPSNET models constitutes of a network model representing the reservoir
% and well connections. The model consists of a set of 1D reservoir models
% between the wells. In the graph terminology, the wells are represented as
% edges and the 1D reservoirs connecting the wells are edges.
% 
% In this example we set up an ensemble of such models, and we run ensemble
% simulations with uncertain operator properties with each connection
% having homogeneous operator properties throughout the connection.

mrstModule add ad-core ad-blackoil mrst-gui ad-props ...
    example-suite incomp ensemble dd-models diagnostics

mrstVerbose off

%% Set the name of the base problem and ensemble size
% The base problem contains all properties that are common throughout the
% ensemble

baseProblemName = 'ensemble_base_problem_simple_gpsnet_model';
baseProblemOptions = {};

ensembleSize = 70;
numConnections = 4; % Not a parameter, but hard-coded in the baseProblem

%% Define where to store truth and ensemble simulations
directoryTruth = fullfile(mrstOutputDirectory(), ...
                          'historyMatching', 'truth','network_operator', ...
                          baseProblemName);
                      
topDirectory = fullfile(mrstOutputDirectory(), ...
                        'historyMatching', 'tutorial', 'network_operator', baseProblemName);

%% Create example
baseExample = MRSTExample(baseProblemName, ...
                          'deleteOldResults', false, ...
                          'plotNetwork', false);

                      
%% Create the full model so that we can generate observations
trueExample = MRSTExample(baseExample.options.fullExampleName, ...
                          'deleteOldResults', false);
trueProblem = trueExample.getPackedSimulationProblem('Directory', directoryTruth);

plotExample = false;
rerunTrueProblemFromScratch = false;
overwriteObservation = true;

if rerunTrueProblemFromScratch
    clearPackedSimulatorOutput(trueProblem);
end
simulatePackedProblem(trueProblem);
if plotExample
    [wellSols, states, reports] = getPackedSimulatorOutput(trueProblem);
    trueExample.plot(states);
end

% Read and store the observations in a QoI object
trueQoI = WellQoIHM('wellNames', {'P1', 'P2'}, ...
                    'fldname', {'qOs', 'qWs'});
trueQoI = trueQoI.validateQoI(trueProblem);

% Create a separate ResultHandler for the observations 
observationResultHandler = trueQoI.ResultHandler;
observationResultHandler.dataPrefix = 'observedQoI';

% Create a separate ResultHandler for the observations.
% Need to build a new ResultHandler from scratch, so that we do not
% overwrite the dataPrefix property of observationResultHandler
truthResultHandler = ResultHandler('dataPrefix', 'trueQoI', ...
                                   'writeToDisk', observationResultHandler.writeToDisk,...
                                   'dataDirectory', observationResultHandler.dataDirectory, ...
                                   'dataFolder', observationResultHandler.dataFolder, ...
                                   'cleardir', false);

% Define the observation uncertainty and perturb the observations
% accordingly
obsStdDev = 0.00004;
trueObservations  = trueQoI.getQoI(trueProblem); 
if numel(observationResultHandler.getValidIds) < 1 || overwriteObservation
    for w = 1:numel(trueQoI.wellNames)
        for f = 1:numel(trueQoI.fldname)
            perturbedObservations{w}{f} = trueObservations{w}{f} + randn(size(trueObservations{w}{f}))*obsStdDev;
        end
    end
    observationResultHandler{1} = {perturbedObservations};
end
if numel(truthResultHandler.getValidIds) < 1 || overwriteObservation
    truthResultHandler{1} = {trueObservations};
end


                      
%% Define samples that give different transmissibilise for each connection

% Initializing as a log-Gaussian distribution around the value obtained
% from flow diagnostics 
% (assuming FD finds the same T for all connections)
logMean = log(baseExample.model.operators.T(baseExample.options.connectionIndices.faces{1}(1)));
transData = cell(ensembleSize, 1);
for i = 1:ensembleSize
    transData{i}.T = exp(logMean + 1*randn(1, numConnections));
end


%% Create sample object
transSamples = NetworkOperatorSamplesHM('data', transData, ...
                                        'connectionIndices', baseExample.options.connectionIndices)

%% Create QoI, with the observations
qoi = WellQoIHM(...
    'wellNames', {'P1', 'P2'}, ...
    'fldname', {'qOs', 'qWs'}, ...
    'observationResultHandler', observationResultHandler, ...
    'truthResultHandler', truthResultHandler, ...
    'observationCov', obsStdDev^2);


%% Create the ensemble
transEnsemble = MRSTHistoryMatchingEnsemble(baseExample, transSamples, qoi, ...
    'directory', fullfile(topDirectory, 'trans'), ...
    'simulationStrategy', 'parallel', ...
    'maxWorkers', 8, ...
    'reset', true, ...
    'verbose', true);

%% Run ensemble
transEnsemble.simulateEnsembleMembers();


%% Get simulated observations
disp('simulated observations: ')
size(transEnsemble.getEnsembleQoI())

%% Get the matrix of ensemble samples 
disp('Matrix of ensemble samples (parameters):')
size(transEnsemble.getEnsembleSamples())


%% Do history matching
transEnsemble.doHistoryMatching()

%% Run new ensemble
transEnsemble.simulateEnsembleMembers();

%% Plot original and updated ensemble results
transEnsemble.plotQoI('subplots', true, 'clearFigure', false, ...
    'cmapName', 'lines', ...
    'plotTruth', true, ...
    'legend', {'observations', 'truth', 'posterior mean', 'prior mean'});

                








%% History match with porevolumes
FDmean = baseExample.model.operators.pv(baseExample.options.connectionIndices.cells{1}(1));
pvData = cell(ensembleSize, 1);
for i = 1:ensembleSize
    pvData{i}.pv = FDmean + 2000*randn(1, numConnections);
end

pvSamples = NetworkOperatorSamplesHM('data', pvData, ...
                                     'connectionIndices', baseExample.options.connectionIndices, ...
                                     'pvScale', baseExample.model.G.cells.volumes(1)/100)



%% Create combo ensemble
pvEnsemble = MRSTHistoryMatchingEnsemble(baseExample, pvSamples, qoi, ...
    'directory', fullfile(topDirectory, 'pv'), ...
    'simulationStrategy', 'parallel', ...
    'maxWorkers', 8, ...
    'reset', true, ...
    'verbose', true)

%% Run ensemble
pvEnsemble.simulateEnsembleMembers();


%% Get simulated observations
disp('simulated observations: ')
size(pvEnsemble.getEnsembleQoI())

%% Get the matrix of ensemble samples 
disp('Matrix of ensemble samples (parameters):')
size(pvEnsemble.getEnsembleSamples())


%% Do history matching
pvEnsemble.doHistoryMatching()

%% Run new ensemble
pvEnsemble.simulateEnsembleMembers();


%% Plot original and updated ensemble results
pvEnsemble.plotQoI('subplots', true, 'clearFigure', false, ...
    'cmapName', 'lines', ...
    'plotTruth', true, ...
    'legend', {'observations', 'truth', 'posterior mean', 'prior mean'});










%% History match with both porevolumes and transmissibilities
operatorData = cell(ensembleSize, 1);
for i = 1:ensembleSize
    operatorData{i}.pv = pvData{i}.pv;
    operatorData{i}.T = transData{i}.T;
end

operatorSamples = NetworkOperatorSamplesHM('data', operatorData, ...
                                     'connectionIndices', baseExample.options.connectionIndices, ...
                                     'pvScale', baseExample.model.G.cells.volumes(1)/100)


                                 
                                
%% Create combo ensemble
operatorEnsemble = MRSTHistoryMatchingEnsemble(baseExample, operatorSamples, qoi, ...
    'directory', fullfile(topDirectory, 'operator'), ...
    'simulationStrategy', 'parallel', ...
    'maxWorkers', 8, ...
    'reset', true, ...
    'verbose', true)

%% Run ensemble
operatorEnsemble.simulateEnsembleMembers();


%% Get simulated observations
disp('simulated observations: ')
size(operatorEnsemble.getEnsembleQoI())

%% Get the matrix of ensemble samples 
disp('Matrix of ensemble samples (parameters):')
size(operatorEnsemble.getEnsembleSamples())


%% Do history matching
operatorEnsemble.doHistoryMatching()

%% Run new ensemble
operatorEnsemble.simulateEnsembleMembers();

%% Plot original and updated ensemble results
operatorEnsemble.plotQoI('subplots', true, 'clearFigure', false, ...
    'cmapName', 'lines', ...
    'plotTruth', true, ...
    'legend', {'observations', 'truth', 'posterior mean', 'prior mean'});
