function [info, present] = dataset_simplecomp()
% Info function for simple compositional dataset. 
% Use getDatasetInfo or getAvailableDatasets for practical purposes.

%{
Copyright 2009-2015 SINTEF ICT, Applied Mathematics.

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
        'name', 'Simple_compositional_problem', ...
        'website', 'www.sintef.no/mrst', ...
        'fileurl', 'http://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/simple_comp.zip', ...
        'hasGrid', true, ...
        'hasRock', true, ...
        'hasFluid', true, ...
        'cells',   1000, ...
        'examples', {}, ...
        'description', 'A simple 1D compositional problem with CO2, Methane and Decane.',...
        'filesize',    38.2, ...
        'modelType', 'Two-phase compositional model. Cartesian 1D grid' ...
         );
end
