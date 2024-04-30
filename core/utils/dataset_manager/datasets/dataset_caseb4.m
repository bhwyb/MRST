function [info, present] = dataset_caseb4()
% Info function for CaseB4 dataset. Use getDatasetInfo or getAvailableDatasets for practical purposes.

%{
Copyright 2009-2024 SINTEF Digital, Mathematics & Cybernetics.

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
    [info, present] = datasetInfoStruct(...
        'name', 'CaseB4', ...
        'fileurl', 'https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/caseB4.zip', ...
        'hasGrid', true, ...
        'hasRock', false, ...
        'description', [...
           'The dataset contains four realizations of two intersecting ' ...
           'faults modelled with a stair-step and a pillar grid with ' ...
           '36x48 or 72x96 cells in the lateral resolution'], ...
        'hasFluid', false, ...
        'filesize',    0.5, ...
        'examples', {'showCaseB4', ...
                     'book:showCaseB'}, ...
        'modelType', 'Corner-point, stair-step and pillar' ...
         );
end
