function plotWellAllocationComparison(D1, WP1, D2, WP2)
%Plot a panel comparing well-allocation from models with different resolution
%
% SYNOPSIS
%   plotWellWallocationComparision(D1, WP1, D2, WP2)
%
% PARAMETERS:
%   D1, D2   - data structure with basic data for flow diagnostics computed
%              by a call to 'computeTOFandTracer' for model 1 and model 2
%
%   WP1, WP2 - data structure containing information about well pairs,
%              computed by a call to 'computeWellPairs'
%
% DESCRIPTION:
%   The routine makes a bar plot for each well segment that is represented
%   in the input data D/WP. (There is no check that the well segments in
%   D1 and D2 are the same). For injection wells, the bars represent
%   the cumulative outfluxes, from the bottom to the top of the segment,
%   that have been attributed to the different producers. The bars are
%   shown in color for model 1, with a unique color representing each of
%   the segments of the producers,  and in solid black lines for model 2.
%   For the production wells, the bars represent influxes that can be
%   attributed to different injector segments. If the two models predict
%   the same flux allocation, the color bars and the solid lines should be
%   matching.
%
% SEE ALSO:
%   computeTOFandTracer, computeWellParis, expandWellCompletions

%{
Copyright 2009-2014 SINTEF ICT, Applied Mathematics.

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

% Extract and format the well-allocation factors for model 1
nit = numel(D1.inj);
npt = numel(D1.prod);
for i=1:numel(WP1.inj)
   nc   = numel(WP1.inj(i).z);
   atmp = zeros(nc, nit+npt);
   atmp(:,D1.prod) = WP1.inj(i).alloc;
   WP1.inj(i).alloc = atmp;
end
for i=1:numel(WP1.prod)
   nc = numel(WP1.prod(i).z);
   atmp = zeros(nc, nit+npt);
   atmp(:,D1.inj) = WP1.prod(i).alloc;
   WP1.prod(i).alloc = atmp;
end
wp1(D1.inj)  = WP1.inj(:);
wp1(D1.prod) = WP1.prod(:);

% Extract and format the well-allocation factors for model 2
nit = numel(D2.inj);
npt = numel(D2.prod);
for i=1:numel(WP2.inj)
   nc   = numel(WP2.inj(i).z);
   atmp = zeros(nc,nit+npt);
   atmp(:,D2.prod) = WP2.inj(i).alloc;
   WP2.inj(i).alloc  = atmp;
end
for i=1:numel(WP2.prod)
   nc   = numel(WP2.prod(i).z);
   atmp = zeros(nc,nit+npt);
   atmp(:,D2.inj) = WP2.prod(i).alloc;
   WP2.prod(i).alloc  = atmp;
end
wp2(D2.inj)  = WP2.inj(:);
wp2(D2.prod) = WP2.prod(:);

% Plot the well-allocation factors for fine/coarse scale models
nsp = floor( (numel(wp1)+1)/2);
for i=1:numel(wp1)
   subplot(2,nsp,i);
   
   [~,ix]   = sort(wp1(i).z);
   wp1(i).z = wp1(i).z(ix);
   wp1(i).alloc = wp1(i).alloc(ix,:);
   if numel(wp1(i).z) == 1  % Need to trick Matlab
      z = [wp1(i).z; wp1(i).z+1];
      a = [cumsum(wp1(i).alloc,1); zeros(1,numel(wp1(i).alloc))];
      bwidth = 2*numel(wp2(i).z)+3;
   else
      z = flipud(wp1(i).z);
      a = cumsum(flipud(wp1(i).alloc),1);
      bwidth=.8;
   end
   h = barh(z, a,'stacked','BarWidth',bwidth);
   hold on
   if numel(wp2(i).z) == 1  % Need to trick Matlab
      z = [wp2(i).z; wp2(i)+1];
      a = [cumsum(wp2(i).alloc,1); zeros(1,numel(wp2(i).alloc))];
   else
      z = flipud(wp2(i).z);
      a = cumsum(flipud(wp2(i).alloc),1);
   end
   %barh(z,a,'stacked','FaceColor','none','EdgeColor','k','LineWidth',2);
   barh(z,a,'stacked','FaceColor','none','LineWidth',2);
   hold off
   zm = min(wp2(i).z); zM = max(wp2(i).z);
   set(gca,'YDir','reverse', 'YLim', [zm zM] + [-.2 .2]*(zM-zm));

   if max(wp1(i).alloc(:))>0
      hl=legend(h(i),wp1(i).name,4); set(hl,'FontSize',8); legend boxoff
   else
      hl=legend(h(i),wp1(i).name,3); set(hl,'FontSize',8); legend boxoff
   end
end
cmap = jet(nit+npt);
c = 0.9*cmap + .1*ones(size(cmap)); colormap(c);
drawnow;

end