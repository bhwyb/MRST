classdef physicalModel
    properties
        % Unique identifier for the model
        name
        % The fluid model
        fluid
        % Operators used for construction of systems
        operators
        % Inf norm tolerance for nonlinear iterations
        nonlinearTolerance
        % Grid
        G
        
        % Maximum pressure change
        dpMax
        % Maximum saturation change
        dsMax
    end
    
    methods
        function model = physicalModel()
            model.dpMax = inf;
            model.dsMax = .2;
            model.nonlinearTolerance = 1e-8;
        end
        
        function model = setupOperators(model, G, rock, varargin)
            % Set up divergence/gradient/transmissibility operators
            model.operators = setupSimComp(G, rock, varargin{:});
        end
        
        function [problem, state] = getEquations(model, state0, state, drivingForces, dt, varargin) %#ok
            % Get the equations governing the system
            error('Base class not meant for direct use')
        end
        
        function state = updateState(model, state, dx, drivingForces) %#ok
            % Update state based on non-linear increment
            error('Base class not meant for direct use')
        end
        
        function [convergence, values] = checkConvergence(model, problem, n)
            % Check convergence based on residual tolerances
            if nargin == 2
                n = inf;
            end
            
            values = norm(problem, n);
            convergence = all(values < model.nonlinearTolerance);
            
            if mrstVerbose()
                for i = 1:numel(values)
                    fprintf('%s (%s): %2.2e\t', problem.equationNames{i}, problem.types{i}, values(i));
                end
                fprintf('\n')
            end
        end
        
        function [state, convergence] = stepFunction(model, state, state0, dt, drivingForces, solver, varargin)
            % Make a single linearized timestep
            [problem, state] = model.getEquations(state0, state, dt, drivingForces, varargin{:});
            convergence = model.checkConvergence(problem);
            if ~convergence
                dx = solver.solveLinearProblem(problem);
                state = model.updateState(state, problem, dx, drivingForces); %%
            end
        end
    end
end