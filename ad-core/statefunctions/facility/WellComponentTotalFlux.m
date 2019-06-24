classdef WellComponentTotalFlux < StateFunction
    % Component total flux for wells (with treatment for cross-flow)
    properties

    end
    
    methods
        function gp = WellComponentTotalFlux(varargin)
            gp@StateFunction(varargin{:});
            gp = gp.dependsOn({'FacilityWellMapping'});
            gp = gp.dependsOn('ComponentPhaseFlux');
        end
        
        function v = evaluateOnDomain(prop, model, state)
            ncomp = model.getNumberOfComponents();
            nph = model.getNumberOfPhases();
            v = cell(ncomp, 1);
            phase_flux = prop.getEvaluatedDependencies(state, 'ComponentPhaseFlux');
            
            for c = 1:ncomp
                % Loop over phases where the component may be present
                for ph = 1:nph
                    % Check if present
                    m = phase_flux{c, ph};
                    if ~isempty(m)
                        if isempty(v{c})
                            v{c} = m;
                        else
                            v{c} = v{c} + m;
                        end
                    end
                end
            end
            % If we have cross-flow and/or we are injecting more than one
            % component, we need to ensure that injecting perforation
            % composition mixtures equal the mix entering the wellbore
            map = prop.getEvaluatedDependencies(state, 'FacilityWellMapping');
            W   = map.W;
            vd  = value(v);
            vd  = horzcat(vd{:});
            vdt = sum(vd,2);
            
            isInjector = map.isInjector(map.perf2well);
            injection  = vdt > 0;
            production = ~injection & vdt ~= 0;
            crossflow = (injection & ~isInjector) | ...
                        (production & isInjector);
            if any(injection)
                ws = state.wellSol(map.active);
                targets = arrayfun(@(x) x.type, W, 'UniformOutput', false);
                val = vertcat(ws.val);
                isZero = ~strcmpi(targets, 'bhp') & val == 0;
                nw = numel(W);
                surfaceComposition = cell(ncomp, nph);
                for c = 1:ncomp
                    % Store well injector composition
                    surfaceComposition(c, :) = model.ReservoirModel.Components{c}.getPhaseComponentFractionWell(model.ReservoirModel, state, W);
                end
                rem = cellfun(@isempty, surfaceComposition);
                [surfaceComposition{rem}] = deal(zeros(nw, 1));
                compi = zeros(nw, ncomp);
                Wcomp = vertcat(W.compi);
                for ph = 1:nph
                    compi = compi + Wcomp(:, ph).*[surfaceComposition{:, ph}];
                end
                if any(crossflow)
                    % allready been disp'ed in WellPhaseFlux
                    compi = crossFlowMixture(vd, compi, map, true, isZero);
                end
                compi_perf = compi(map.perf2well, :);
                vt = zeros(sum(injection), 1);
                for i = 1:nph
                    vt = vt + v{i}(injection);
                end    
                for i = 1:ncomp
                    v{i}(injection) = vt.*compi_perf(injection, i);
                end
            end
        end
    end
end