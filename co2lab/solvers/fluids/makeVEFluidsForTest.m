function fluid = makeVEFluidsForTest(fluid,fluid_case,varargin)
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
   opt = struct('res_water', 0, 'res_gas', 0, 'Gt', [], 'rock', []);
   opt = merge_options(opt, varargin{:});

   fluid_names = {'simple'               ,...
                  'sharp_interface'      ,...
                  'cap_linear'           ,...
                  'cap_1D_table_P'       ,...
                  'cap_1D_table_kscaled' ,...
                  'cap_1D_table_SH'      ,...
                  'integrated'};

   if(nargin == 0)
      fluid = fluid_names;
      return;
   else
      if(~any(strcmp(fluid_case, fluid_names)))
         disp(['Error wrong fluid name', fluid_case])
         disp('Valid names are')
         for i = 1:numel(fluid_names)
            disp(fluid_names{i});
         end
         error('Wrong fluid name')
      end
   end

   switch fluid_case
     case 'simple'
       fluid.krG       = @(sg, varargin) sg;
       fluid.krW       = @(so, varargin) so;
       fluid.pcWG      = @(sg, p, varargin) norm(gravity) * (fluid.rhoWS .* fluid.bW(p) - ...
                                            fluid.rhoGS .* fluid.bG(p)) .* (sg) .* opt.Gt.cells.H;
       fluid.res_gas   = 0;
       fluid.res_water = 0;
       fluid.invPc3D   = @(p) 1 - (sign(p + eps) + 1) / 2;
       fluid.kr3D      = @(s) s;
     case 'integrated'
       fluid = addVERelpermIntegratedFluid(fluid         , ...
                                           'res_water'   , opt.res_water ,...
                                           'res_gas'     , opt.res_gas   ,...
                                           'Gt'          , opt.Gt        ,...
                                           'kr_pressure' , true          ,...
                                           'Gt'          , opt.Gt        ,...
                                           'int_poro'    , false         ,...
                                           'rock'        , opt.rock);
     case 'sharp_interface'
       fluid = addVERelperm(fluid, opt.Gt,...
                            'res_water', opt.res_water,...
                            'res_gas', opt.res_gas);

    case 'cap_linear'
        fluid = addVERelpermCapLinear(fluid,0.3*max(opt.Gt.cells.H)*10*(fluid.rhoWS-fluid.rhoGS),...
                                      'res_gas'   ,opt.res_gas    , ...
                                      'res_water' ,opt.res_water  , ...
                                      'beta'      ,2              , ...
                                      'H'         ,opt.Gt.cells.H , ...
                                      'kr_pressure' ,true);
    case 'VE_1D_table_test'

      S_tab = linspace(0, 1, 10)';
      kr_tab = S_tab;
      h_tab = S_tab * max(opt.Gt.cells.H);
      table_co2_1d = struct('SH'         , S_tab .* opt.Gt.cells.H  , ...
                            'krH'        , kr_tab .* opt.Gt.cells.H , ...
                            'h'          , h_tab                    , ...
                            'is_kscaled' , false);
      table_co2_1d.invPc3D = @(p) 1 - (sign(p + eps) + 1) / 2;
      table_co2_1d.kr3D = @(s) s;
      S_tab_w = S_tab;
      kr_tab_w = S_tab_w;
      table_water_1d = struct('S', 1 - S_tab_w, 'kr', kr_tab_w, 'h', []);
      fluid = addVERelperm1DTables(fluid,...
                                   'height', opt.Gt.cells.H,...
                                   'table_co2', table_co2_1d,...
                                   'table_water',table_water_1d);
     case 'cap_1D_table_SH'
       drho = 400;
       C = max(opt.Gt.cells.H) * 0.4 * drho * norm(gravity);
       alpha = 0.5;
       beta = 3;
       samples = 100;
       table_co2_1d = makeVEtables('invPc3D'    , @(p) max((C ./ (p + C)).^(1 / alpha), opt.res_water) ,...
                                   'is_kscaled' , false        ,...
                                   'kr3D'       , @(s) s.^beta ,...
                                   'drho'       , drho         ,...
                                   'Gt'         , opt.Gt       ,...
                                   'samples'    , samples);
       S_tab = linspace(0, 1, 10)';
       S_tab_w = S_tab;
       kr_tab_w = S_tab_w;
       table_water_1d = struct('S', 1 - S_tab_w, 'kr', kr_tab_w, 'h', []);
       fluid = addVERelperm1DTables(fluid,...
                                    'res_water'   , opt.res_water  ,...
                                    'res_gas'     , opt.res_gas    ,...
                                    'height'      , opt.Gt.cells.H ,...
                                    'table_co2'   , table_co2_1d   ,...
                                    'table_water' , table_water_1d);
     case 'cap_1D_table_P'
       drho = 400;
       C = max(opt.Gt.cells.H) * 0.4 * drho * norm(gravity);
       alpha = 0.5;
       beta = 3;
       samples = 100;
       table_co2_1d = makeVEtables('invPc3D', @(p) max((C ./ (p + C)).^(1 / alpha), opt.res_water),...
                                   'is_kscaled' , false        , ...
                                   'kr3D'       , @(s) s.^beta , ...
                                   'drho'       , drho         , ...
                                   'Gt'         , opt.Gt       , ...
                                   'samples'    , samples);
       S_tab = linspace(0, 1, 10)';
       S_tab_w = S_tab;
       kr_tab_w = S_tab_w;
       table_water_1d = struct('S', 1 - S_tab_w, 'kr', kr_tab_w, 'h', []);
       fluid = addVERelperm1DTablesPressure(fluid         , ...
                                            'res_water'   , opt.res_water  ,...
                                            'res_gas'     , opt.res_gas    ,...
                                            'height'      , opt.Gt.cells.H ,...
                                            'table_co2'   , table_co2_1d   ,...
                                            'table_water' , table_water_1d ,...
                                            'kr_pressure' , true);
      case 'cap_1D_table_kscaled'
        kscale = sqrt(0.1 / (100 * milli * darcy)) * fluid.surface_tension;
        drho = 400;
        C = 1;
        alpha = 0.5;
        beta = 3;
        samples = 100;
        table_co2_1d = makeVEtables('invPc3D', @(p) max((C ./ (p + C)).^(1 / alpha), opt.res_water),...
                                    'is_kscaled' , true         ,...
                                    'kr3D'       , @(s) s.^beta ,...
                                    'drho'       , drho         ,...
                                    'Gt'         , opt.Gt       ,...
                                    'samples'    , samples      ,...
                                    'kscale'     , kscale);
        S_tab = linspace(0, 1, 10)';
        S_tab_w = S_tab;
        kr_tab_w = S_tab_w;
        table_water_1d = struct('S', 1 - S_tab_w, 'kr', kr_tab_w, 'h', []);
        fluid = addVERelperm1DTablesPressure(fluid,...
                                             'res_water'   , opt.res_water  ,...
                                             'res_gas'     , opt.res_gas    ,...
                                             'height'      , opt.Gt.cells.H ,...
                                             'table_co2'   , table_co2_1d   ,...
                                             'table_water' , table_water_1d ,...
                                             'rock'        , opt.rock       ,...
                                             'kr_pressure' , true);
     otherwise
       error('No such fluid case')
   end
   fluid.name = fluid_case;
end
