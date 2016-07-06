classdef StateChangeTimeStepSelector < IterationCountTimeStepSelector
    properties
        targetProps = {};
        targetChangeRel = [];
        targetChangeAbs = [];
        relaxFactor = 1;
    end
    
    methods
        function selector = StateChangeTimeStepSelector(varargin)
            selector = selector@IterationCountTimeStepSelector();

            selector = merge_options(selector, varargin{:});
            
            nprops = numel(selector.targetProps);
            if numel(selector.targetChangeRel) == 0
                selector.targetChangeRel = inf(1, nprops);
            end
            
            if numel(selector.targetChangeAbs) == 0
                selector.targetChangeAbs = inf(1, nprops);
            end
            
            assert(numel(selector.targetProps) == numel(selector.targetChangeRel) && ...
                   numel(selector.targetProps) == numel(selector.targetChangeAbs));
        end
        
        function dt = computeTimestep(selector, dt, dt_prev, model, solver, state_prev, state_curr)
            dt0 = dt;
            dt = computeTimestep@IterationCountTimeStepSelector(selector, dt, dt_prev, model, solver, state_prev, state_curr);
            
            if isempty(state_prev)
                return
            end
            dt_next = inf;
            for i = 1:numel(selector.targetProps)
                fn = selector.targetProps{i};
                curr = model.getProp(state_curr, fn);
                if ~iscell(curr)
                    curr = {curr};
                end
                prev = model.getProp(state_prev, fn);
                if ~iscell(prev)
                    prev = {prev};
                end
                
                nvals = numel(curr);
                assert(nvals == numel(prev), ['Mismatch in property size: ', fn]);
                for jj = 1:nvals
                    c = curr{jj};
                    p = prev{jj};
                    if iscell(c)
                        % Assume equally spaced values
                        c = [c{:}];
                    end
                    if iscell(p)
                        p = [p{:}];
                    end
                    
                    if nvals == 1
                        sname = '';
                    else
                        sname = [' (', num2str(jj), ')'];
                    end
                    changeAbs = abs(c - p);
                    changeRel = changeAbs./abs(p);
                    changeRel(isnan(changeRel)) = 0;
                    
                    maxAbs = max(max(changeAbs, [], 2));
                    maxRel = max(max(changeRel, [], 2));
                    
                    w = selector.relaxFactor;
                    
                    targetAbs = selector.targetChangeAbs(i);
                    targetRel = selector.targetChangeRel(i);
                    
                    f_abs = ((1 + w).*targetAbs)/(maxAbs + w*targetAbs);
                    f_rel = ((1 + w).*targetRel)/(maxRel + w*targetRel);
                    f_rel(isnan(f_rel)) = inf;
                    
                    dt_next = min([dt_next, dt_prev*f_rel, dt_prev*f_abs]);
                    dispif(selector.verbose, 'Property ''%s%s'': Rel change %1.2f [%1.2f], abs change: %1.2f [%1.2f]-> adjustment of %1.2f\n', ...
                        fn, sname, maxRel, targetRel, maxAbs, targetAbs, min(f_rel, f_abs));
                end
                dt = min(dt, dt_next);
            end
            
            
        end
    end
end

%{
Copyright 2009-2016 SINTEF ICT, Applied Mathematics.

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
