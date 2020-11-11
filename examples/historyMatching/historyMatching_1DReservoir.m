%% History mathing of 1D reservoir simulation
% In this example, we use well production data to estimate 
mrstModule add ad-core ad-blackoil mrst-gui ad-props ...
    example-suite incomp ensemble

mrstVerbose off


%% Set up and simulate the true solution
% We will here use an identical twin experiment, where we use the same
% problem for both generating the truth and as a base for our ensemble.


trueProblemName = 'ensemble_base_problem_1d_reservoir';
numCells = 10;
trueProblemOptions = {'ncells', numCells, ...
                      'rngseed', 1};
                  
directoryTruth = fullfile(mrstOutputDirectory(), ...
                          'historyMatching', 'truth', ...
                          trueProblemName);
                  
trueExample = MRSTExample(trueProblemName, trueProblemOptions{:});
trueProblem = trueExample.getPackedSimulationProblem('Directory', directoryTruth);

plotExample = false;
rerunTrueProblemFromScratch = false;


if rerunTrueProblemFromScratch
    clearPackedSimulatorOutput(trueProblem);
end
simulatePackedProblem(trueProblem);
if plotExample
    [wellSols, states, reports] = getPackedSimulatorOutput(trueProblem);
    trueExample.plot(states);
end



%% Generate observations 
% Define a QOI for storing the relevant qoi for our problem

trueQoI = WellQoI('wellNames', {'P1'}, 'fldname', {'qOs'});
trueQoI = trueQoI.validateQoI(trueProblem);
trueObservations = trueQoI.getQoI(trueProblem);


% Create a separate ResultHandler for the observations 
observationResultHandler = trueQoI.ResultHandler;
observationResultHandler.dataPrefix = 'observedQoI';

% Define observation uncertainty 
obsStdDev = 0.0004;

% Add some observation noise and store output
if numel(observationResultHandler.getValidIds) < 1
    for w = 1:numel(trueQoI.wellNames)
        for f = 1:numel(trueQoI.fldname)
            perturbedObservations{w}{f} = trueObservations{w}{f} + randn(size(trueObservations{w}{f}))*obsStdDev;
        end
    end
    observationResultHandler{1} = {perturbedObservations};
end

%% Select and populate samples for the stochastic components in the ensemble

ensembleSize = 23;

configData = cell(ensembleSize, 1);
for i = 1:ensembleSize
    configData{i}.poro = gaussianField(trueExample.model.G.cartDims, [0.2 0.4]); 
    configData{i}.perm = configData{i}.poro.^3.*(1e-5)^2./(0.81*72*(1-configData{i}.poro).^2);
end

samples = RockSamples('data', configData)

%% Select quantity of interest class matching the what we have as observations
% We validate the QoI with the trueProblem, since this will be our ensemble
% base problem as well.

qoi = WellQoI('wellNames', {'P1'}, 'fldname', {'qOs'}, ...
                  'observationResultHandler', observationResultHandler, ...
                  'observationCov', obsStdDev^2)

%% Create the ensemble

ensemble = MRSTHistoryMatchingEnsemble(trueExample, samples, qoi, ... 
    ... %'directory', uniqueDirectory, ...
    'simulationStrategy', 'parallel', ...
    'maxWorkers', 8, ...
    'verbose', true, ...
    'reset', true...
    );


%% Displaying the observations and observation error cov through the ensemble
disp('observation and scaling vector')
[obsVector, obsScaling] = ensemble.qoi.getObservationAndScaling();
disp('observation error covariance matrix')
ensemble.qoi.getObservationErrorCov()


%% Run ensemble
ensemble.simulateEnsembleMembers();

%% Get simulated observations
disp('simulated observations')
size(ensemble.getEnsembleQoI())

%% Get the matrix of ensemble samples 
size(ensemble.getEnsembleSamples())

%% Do history matching
disp('updated sample object:')
updatedSamples = ensemble.doHistoryMatching()


%% Create a new ensemble with updated samples
updatedEnsemble = MRSTHistoryMatchingEnsemble(trueExample, updatedSamples, qoi, ... 
    ... %'directory', uniqueDirectory, ...
    'simulationStrategy', 'parallel', ...
    'maxWorkers', 8, ...
    'verbose', true, ...
    'reset', false, ...
    'historyMatchingIteration', 2 ...
    );

%% Run new ensemble
updatedEnsemble.simulateEnsembleMembers();

%% Plot first ensemble
h = ensemble.plotQoI('color', [0.1 0.6 0.4]);

%% Plot updated ensemble
updatedEnsemble.plotQoI('h', h, 'color', [0.8 0.2 0], 'clearFigure', false);

%% Plot first ensemble
ensemble.plotQoI('h', h, 'clearFigure', false, 'color', [0.1 0.6 0.4]);


%% Plot diff between the qois
figure
hold on
for i = 1:ensemble.num
    plot(updatedEnsemble.qoi.ResultHandler{i}{1}{1} - ensemble.qoi.ResultHandler{i}{1}{1})
end







