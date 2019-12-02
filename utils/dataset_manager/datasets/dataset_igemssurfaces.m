function [info, present] = dataset_igemssurfaces()
%Dataset Function for IGEMS Surfaces

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

   [info, present] = datasetInfoStruct(...
       'name', 'IGEMS surfaces', ...
       'website', 'https://www.nr.no/IGEMS/', ...
       'fileurl', 'http://files.nr.no/igems/surfaces.zip', ...
       'hasGrid', true, ...
       'hasRock', false, ...
       'cells', 180000, ...
       'examples', {'co2lab:showIGEMS', ...
                    'co2lab:trapsIGEMS', ...
                    'co2lab:fillTreeIGEMS', ...
                    'co2lab:showTrapsInteractively'}, ...
       'filesize', 600, ...
       'description', [...
           'Surfaces generated in the IGEMS project to study the impact of ' ...
           'caprock shape on CO2 migration using geologically realistic ' ...
           'models.  NB: This is a large dataset (600 Mb) containing 15 ' ...
           'different models with 100 realizations each.'], ... 
       'modelType', 'Surface');
end
