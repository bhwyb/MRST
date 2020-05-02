classdef WellPhaseFlux < StateFunction
    % Get phase-flux between well-bore and reservoir
    properties
    allowCrossFlow = true;
    end

    methods
        function gp = WellPhaseFlux(varargin)
            gp@StateFunction(varargin{:});
            gp = gp.dependsOn({'FacilityWellMapping', 'PerforationPressureGradient', 'WellIndex'});
            gp = gp.dependsOn({'Mobility'}, 'FlowPropertyFunctions');
            gp.label = 'q_\alpha';
        end

        function q_ph = evaluateOnDomain(prop, model, state)
            map = prop.getEvaluatedDependencies(state, 'FacilityWellMapping');
            [dp, wi] = prop.getEvaluatedDependencies(state, 'PerforationPressureGradient', 'WellIndex');
            mob = model.ReservoirModel.getProps(state, 'Mobility');

            mobw = cellfun(@(x) x(map.cells), mob, 'UniformOutput', false);

            Tdp = -wi.*dp;
            
            % TODO: there can be different way to refactor the following.
            % It can be we apply all the effects to obtain the mobility.
            % Then we calculate the well rates based on the obtained
            % mobility
            mobw = updateMobility(mobw, Tdp, model, state, map); 

            q_ph = calculatePhaseRate(Tdp, mobw, map);

            q_ph = applyOtherEffects(q_ph, Tdp, mobw, model, prop, state, map);

        end
    end
end

function q_ph = calculatePhaseRate(Tdp, mobw, map)
    isInjector = map.isInjector(map.perf2well);
    W = map.W;
    vTdp = value(Tdp);
    injection = vTdp > 0;
    production = ~injection & vTdp ~= 0;
    crossflow = (injection & ~isInjector) | ...
                (production & isInjector);
    crossflow = crossflow & prop.allowCrossFlow;
    
    nph = numel(mobw);
    if any(injection)
        compi = vertcat(W.compi);
        if any(crossflow)
            dispif(model.verbose > 1, 'Crossflow occuring in %d perforations\n', sum(crossflow));
            % Compute cross flow for this phase. The approach here
            % is to calculate (as doubles) the volumetric inflow of
            % all phases into the well-bore. If a well has
            % cross-flow, the phase distribution of the
            % cross-flowing volume is assumed to reflect the inflow
            % conditions, neglecting density change throughout the
            % wellbore.
            q_wb = bsxfun(@times, value(mobw), vTdp);
            compi = crossFlowMixture(q_wb, compi, map);
        end
        compi_perf = compi(map.perf2well, :);
        mobt = zeros(sum(injection), 1);
        for i = 1:nph
            mobt = mobt + mobw{i}(injection);
        end
        for i = 1:nph
            mobw{i}(injection) = mobt.*compi_perf(injection, i);
        end
    end
    q_ph = cell(1, nph);
    for i = 1:nph
        q_ph{i} = mobw{i}.*Tdp;
    end
end

function mobw = updateMobility(mobw, Tdp, model, state, map)
% Currently, this function only handles the effects related to polymer
    vTdp = value(Tdp);
    injection = vTdp > 0;

    % for polymer injecting, we need to modify the injecting
    % mobility
    check = @(prop) isprop(model.ReservoirModel, prop) && model.ReservoirModel.(prop);
    haspolymer = check('polymer');
    if (haspolymer)
        if model.ReservoirModel.polymer
            cp = model.ReservoirModel.getProps(state, 'polymer');
            % TODO: we might need to only do this to injection well
            % instead of all the injection perforations
            % TODO: how to handle the crossflow of polymer remains to be
            % fixed
            % TODO: and also, we can trying to have the fully
            % mixing multiplier (or some other thing related) as
            % state function, then a few places related to
            % different viscosities can be built upon these state
            % functions
            wc_inj = (map.cells(injection));
            cp_inj = cp(wc_inj);
            wIx = 1;
            mobw_injw = mobw{wIx}(injection);

            viscpmult = model.ReservoirModel.getProps(state, 'PolymerEffViscMult');
            viscpmult_inj = viscpmult(wc_inj);
            viscpmultfull = model.ReservoirModel.fluid.muWMult(cp_inj);

            mobw{wIx}(injection) = mobw_injw ./ viscpmultfull .* viscpmult_inj;
        end
    end
end

function q_ph = applyOtherEffects(q_ph, Tdp, mobw, model, prop, state, map)
    check = @(prop) isprop(model.ReservoirModel, prop) && model.ReservoirModel.(prop);
    haspolymer = check('polymer');
    if haspolymer
        hasShear= check('usingShear') || check('usingShearLog') || check('usingShearLogshrate');
        if hasShear
            mobw = applyShearEffectsWell(mobw, q_ph, prop, model.ReservoirModel, state);

            q_ph = calculatePhaseRate(Tdp, mobw, map);

        end
    end
end



%{
Copyright 2009-2019 SINTEF Digital, Mathematics & Cybernetics.

This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).

MRST is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MRST is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MRST.  If not, see <http://www.gnu.org/licenses/>.
%}
