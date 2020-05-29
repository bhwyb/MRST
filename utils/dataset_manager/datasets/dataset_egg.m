function [info, present] = dataset_egg()
% Info function for Egg dataset. Use getDatasetInfo or getAvailableDatasets for practical purposes.

%{
Copyright 2009-2020 SINTEF Digital, Mathematics & Cybernetics.

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

    helper = ['Visit the 4TU website to download (free registration required).'...
              ' Extract the contents of the folder "Egg_Model_Data_Files_v2" from the zip archive'...
              ' into a folder named "Egg" under the current data directory: "',...
              mrstDataDirectory(), '"'];

    [info, present] = datasetInfoStruct(...
        'name', 'Egg', ...
        'website', 'https://data.4tu.nl/repository/uuid:916c86cd-3558-4672-829a-105c62985ab2#DATA', ...
        'fileurl', '', ...
        'hasGrid', true, ...
        'hasRock', true, ...
        'hasFluid', true, ...
        'cells', 18553, ...
        'instructions', helper , ...
        'examples', { ...
                     }, ...
        'description', [], ...
        'modelType', 'Two-phase oil-water, corner-point' ...
         );
end
