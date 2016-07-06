function [info, present] = dataset_bedmodel2()
% Info function for bedModel2 dataset. Use getDatasetInfo or getAvailableDatasets for practical purposes.

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
    [info, present] = datasetInfoStruct(...
        'name', 'BedModel2', ...
        'fileurl', 'http://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/bedModel2.zip', ...
        'hasGrid', true, ...
        'hasRock', true, ...
        'description', [...
           'The model represents a 30^3 cm model of a sedimentary bed ' ...
           'that contains six different rock types. The model is a good ',...
           'example of a corner-point grid having a large number of ', ...
           'inactive cells and cells with degenerate geometry.'], ...
        'hasFluid', false, ...
        'filesize',    3.4, ...
        'cells',  91831, ...
        'source', 'Statoil', ...
        'modelType', 'Corner-point' ...
         );
end
