function [bcfun, num_bc_ctrls] = set_bcfun(G, binfo, bcond)

   assert(size(bcond, 1) == 3 || size(bcond, 1) == 5);
   if size(bcond, 1) == 3
      bcond = [bcond; zeros(2, 2)]; % add placeholder for zero gradient
   end
   
   bottom_nodes = binfo.side_nodes{6};
   side_nodes = unique(vertcat(binfo.side_nodes{1:4}));
   
   % identify bottom edge as a separate set of nodes, and take it out of
   % the sets 'side_nodes' and 'bottom_nodes'
   bottom_edge_nodes = intersect(bottom_nodes, side_nodes);
   side_nodes = setdiff(side_nodes, bottom_edge_nodes);
   bottom_nodes = setdiff(bottom_nodes, bottom_edge_nodes); 
   
   % compute the ranges within which we optimise
   % bc_ranges = [bcond(:, 1) - md_mult * bcond(:, 2), ...
   %              bcond(:, 1) + md_mult * bcond(:, 2)];
   
   % identify active optimization variables and setup mapping to full vector
   % of optimization variables
   active_u = find(bcond(:,2) > 0);
   num_bc_ctrls = numel(active_u);
   
   full_u = @(u) accumarray([active_u(:), (1:numel(active_u))'], ...
                            1, [5, numel(active_u)]) * ...
                 reshape(u(1:numel(active_u)), [], 1);
   
   % setup boundary condition functions
   
   roller_bottom = ...
       SparseMultiArray(0, bottom_nodes, 'n') * SparseMultiArray([NaN, NaN, 1], 'd');
   
   disp_bc_fun = theta_e1_e2_grad(G, bcond(:,1), bcond(:, 2));
   disp_bc_fun_zlock = theta_e1_e2_grad(G, bcond(:,1), bcond(:, 2), 'zdisp', 0);
      
   % collect components in common return structure
   bcfun = ...
       @(u) struct('disp_bc', roller_bottom + ...
                   disp_bc_fun(side_nodes, full_u(u)) + ...
                   disp_bc_fun_zlock(bottom_edge_nodes, full_u(u)), ...
                   'force_bc', []);
end

% ----------------------------------------------------------------------------
function side_bc = theta_e1_e2_grad(G, prior, sigma, varargin)
   
   opt.zdisp = nan; % set to numeric value to impose displacement in z-direction
   opt = merge_options(opt, varargin{:});
   
   ipol = @(t, vmin, vmax) vmin * (1-t) + vmax * t;
   varval = @(u, ix) prior(ix) + ipol(u(ix), 0, sigma(ix));
   
   theta = @(u) varval(u, 1);
   e1    = @(u) varval(u, 2);
   e2    = @(u) varval(u, 3);
   de1dz = @(u) varval(u, 4);
   de2dz = @(u) varval(u, 5);
   
   % define rotation tensor
   R = @(u, ix1, ix2) ...
       SparseMultiArray([cos(theta(u)); 
                     -sin(theta(u)); 
                     sin(theta(u));  
                     cos(theta(u)); 
                     1], ...
                    [1, 1; 1, 2; 2, 1; 2, 2; 3, 3], {ix1, ix2});
   
   % un-rotated strain tensor (constant in depth)
   E0 = @(u, ix1, ix2) SparseMultiArray([e1(u); e2(u); 1], ...
                                    [1, 1; 2, 2; 3, 3], {ix1, ix2});
   
   % un-rotated, depth-dependent strain tensor
   E1 = @(u, ix1, ix2) SparseMultiArray([de1dz(u); de2dz(u); 1], ...
                                    [1, 1; 2, 2; 3, 3], {ix1, ix2});
           
   % rotated strain tensor (constant component)
   M0 = @(u) R(u, 'd', 'tmp1') * E0(u, 'tmp1', 'tmp2') * R(u, 'd2', 'tmp2');
   
   % rotated strain tensor (depth-dependent component)
   M1 = @(u) R(u, 'd', 'tmp1') * E1(u, 'tmp1', 'tmp2') * R(u, 'd2', 'tmp2');
   
   % constructing boundary condition structure
   ncoords = G.nodes.coords;
   ncoords(:,3) = 1 * opt.zdisp; %  (zdisp = NaN for roller conditions in z-dir)
   
   side_bc = ...
       @(n_ix, u) ...
       (M0(u) * SparseMultiArray(ncoords(n_ix,:), {'n_loc', 'd2'}) * ...
        SparseMultiArray([], [(1:numel(n_ix))', n_ix], {'n_loc', 'n'})) + ...
       (M1(u) * SparseMultiArray(ncoords(n_ix,:), {'n_loc', 'd2'}) * ...
        SparseMultiArray([], [(1:numel(n_ix))', n_ix], {'n_loc', 'z'}) * ...
        SparseMultiArray(G.nodes.coords(n_ix, 3), [n_ix(:), n_ix(:)], {'z', 'n'}));
end

