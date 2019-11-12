function [F, Q] = getFractionalFlowMagnitude(model, state)
    nph = model.water + model.oil + model.gas;
    nc = model.G.cells.num;
    
    if nph == 1
        F = ones(nc, 1);
    else
        [F, Q] = getFractionalFlowMagnitudeInternal(model, state);
    end
    
    F = findLargestEigenvalue(F);
    Q = findLargestEigenvalue(Q);
end

function [F, Q] = getFractionalFlowMagnitudeInternal(model, state)
    if isempty(model.FlowPropertyFunctions)
        model = model.validateModel();
    end
    state = model.reduceState(state, true);
    s = model.getProp(state, 'saturation');
    nph = size(s, 2);
    s = arrayfun(@(x) s(:, x), 1:nph, 'UniformOutput', false);
    
    [s{1:end-1}] = initVariablesAD_diagonal(s{1:end-1});
    s{end} = 1;
    for i = 1:nph-1
        s{end} = s{end} - s{i};
    end
    state = model.setProp(state, 'saturation', s);

    mob = model.getProps(state, 'Mobility');

    mobT = 0;
    for i = 1:nph
        mobT = mobT + mob{i};
    end
    nc = model.G.cells.num;
    
    F = zeros(nc, (nph-1)*(nph-1));
    for i = 1:nph-1
        frac = mob{i}./mobT;
        
        offset = (nph-1)*(i-1)+1;
        F(:, offset:offset+(nph-2)) = frac.jac{1}.diagonal;
    end
    pc = model.getProp(state, 'CapillaryPressure');
    Q = [];
    % Note to self: Fix capillary pressure here
    
%     if hasOW || hasOG
%         assert(false, 'Not (re)-implemented yet!');
%     	Q = zeros(model.G.cells.num, 4);
%         mobOW = double(mobO.*mobW)./double(mobT);
%         mobOG = double(mobO.*mobG)./double(mobT);
%         mobWG = double(mobW.*mobG)./double(mobT);
%         if hasOW
%             pcow = f.pcOW(sW);
%             dow = J(pcow, 1);
%             Q(:, 1) =  dow.*mobWG;
%             Q(:, 2) = -dow.*(mobWG + mobOW);
%         end
%         if hasOG
%             pcog = f.pcOG(sG);
%             dog = J(pcog, 2);
%             Q(:, 3) =  dog.*(mobOG + mobWG);
%             Q(:, 4) = -dog.*mobWG;
%         end
%     else
%         Q = [];
%     end
end


function F = findLargestEigenvalue(F)
    if size(F, 2) > 1
        nc = size(F, 1);
        nph = sqrt(size(F, 2));
        tmp = zeros(nc, 1);
        for i = 1:nc
            M = reshape(F(i, :), nph, nph);
            tmp(i) = max(abs(eig(M)));
        end
        F = tmp;
    else
        F = abs(F);
    end
end