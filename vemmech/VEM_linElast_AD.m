function [uu, extra] = VEM_linElast_AD(G, E, nu, el_bc, load, varargin)
% Assemble and solve the linear elasticity equations using VEM
%
% SYNOPSIS:
%   function [uu, extra] = VEM_linElast(G, E, nu, el_bc, load, varargin)
%
% DESCRIPTION: Assemble and solve the linear elastisity equations using the
% Virtual Element method.
% PARAMETERS:
%   G        - Grid structure as described by extended_grid_structure
%   E        - Young's modulus, for all cells in the grid (may be AD)
%   nu       - Poisson's ratio, for all cells in the grid (may be AD)
%   el_bc    - Elastic boundary condition structure. It contains the fields
%             'disp_bc'  : displacement boundary condition. It contains the
%                          fields
%                  'nodes'    : nodes where the displacement condition is applied
%                  'uu'       : value for the displacement
%                  'mask'     : if false then displacement values that are
%                               imposed in given Cartesian directions are in
%                               fact ignored.
%             'force_bc'  : force boundary condition applied on faces. It contains the
%                           fields
%                  'faces' : faces where the force is applied
%                  'force' : value of the force that is applied
%
%   load     - loading term
%
% OPTIONAL PARAMETERS:
%  'linsolve'             - Linear solver
%  'blocksize'            - block size used in the assembly
%  'add_operators'        - Add operator in output
%  'force_method'         - Method for computing the loading term, see function calculateVolumeTerm below for
%                           a list of the possible alternatives.
%  'alpha_scaling'        - Coefficient of the stabilisation term (default 1)
%  'S'                    - Stabilization matrix to use (used only in very special cases, experimental feature)
%  'experimental_scaling' - Scaling proposed in [Andersen et al: http://arxiv.org/abs/1606.09508v1]
%  'pressure'             - Pressure field, used at loading term
%  'extra'                - If system was already previously discretized, the
%                           resulting 'extra' struct can be passed along to
%                           this call, to avoid having to call VEM_assemble
%                           again.
%  'background_forces'    - Additional forces to apply to nodes (regardless
%                           of boundary conditions)
%
% RETURNS:
%   uu    - Displacement field
%   extra - Extra outputs
%
% EXAMPLE:
%
% SEE ALSO:
%

%{
Copyright 2009-2018 SINTEF Digital, Mathematics & Cybernetics.

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

    opt = struct('linsolve'     , @mldivide                       , ...
                 'blocksize'    , 30000 - (G.griddim - 2) * 20000 , ...
                 'add_operators', true                            , ...
                 'force_method' , 'dual_grad_type'                , ...
                 'alpha_scaling', 1                               , ...
                 'S'            , []                              , ...
                 'experimental_scaling', false                    , ...
                 'pressure'     , [],...
                 'extra'        , [], ...
                 'background_forces', [], ...
                 'no_solve',false);
    opt = merge_options(opt, varargin{:});
    opt.add_operators = opt.add_operators && nargout>1;
    tic; fprintf('Assembling system.\n');
    
    C = Enu2C(value(E), value(nu), G);
    
    if ~isempty(opt.extra)
       % re-use previously computed discretization
       S = opt.extra.S;
       extra = opt.extra;
    elseif(opt.add_operators)
       [S, extra] = VEM_assemble(G, C, ...
                                 'blocksize'            , opt.blocksize, ...
                                 'alpha_scaling'        , opt.alpha_scaling, ...
                                 'S'                    , opt.S, ...
                                 'experimental_scaling' , opt.experimental_scaling);
    else
        S = VEM_assemble(G, C, ...
                         'blocksize'           , opt.blocksize, ...
                         'alpha_scaling'       , opt.alpha_scaling, ...
                         'S'                   , opt.S, ...
                         'experimental_scaling', opt.experimental_scaling);
    end
    toc
    %% Recalculate "weights" (all are calculated in the assembly, they could
    %% in fact only be calculated once)
    tic; fprintf('Recalculating weights.\n');
    if (G.griddim == 2)
        qf_all = G.faces.areas / 2;
        [qc_all, qcvol] = calculateQF(G);
    else
        [qc_all, qf_all, qcvol] = calculateQC(G);
    end
    toc
    %% Apply Diriclet boundary conditions
    tic; fprintf('Computing boundary conditions.\n');
    [u_bc, dirdofs] = el_bc.disp_bc.asVector({'d', 'n'}, [G.griddim, G.nodes.num]);

    ndof = G.griddim * G.nodes.num;
    isdirdofs = false(ndof, 1);
    isdirdofs(dirdofs) = true;

    ignore = isnan(value(u_bc)); % NaN used to indicate suppressed entries 
    u_bc(ignore) = 0;
    isdirdofs(ignore) = false;
    
    rhso = - S * u_bc;

    %% Calculate and apply load terms
    % There are several alternatives, which may lead to different errors in particular for thin
    % long cells, see paper [Andersen et al: http://arxiv.org/abs/1606.09508v1]
    f = calculateVolumeTerm(G, load, qc_all, qcvol, opt);
    if(~isempty(opt.pressure))
        div = VEM_div(G);
        f_pressure = div'*opt.pressure;
    else
        f_pressure = zeros(size(f));
    end
    % Add load to right hand side
    rhs = rhso + f + f_pressure;
    
    if ~isempty(opt.background_forces)
       rhs = rhs + opt.background_forces;
    end

    %% Add boundary forces
    if(isfield(el_bc, 'force_bc') && ~isempty(el_bc.force_bc))
       
       % defining weight tensor in face-node-dim ('f', 'n')
       if G.griddim == 2
          % qf_all has only been computed for faces, so we expand it to
          % face-node space
          QF = SparseTensor(qf_all, 'f') ^ ...
               face_node_tensor(G, 'f', 'n', 'boundary_only', true);
       else
          % qf_all is already in face-node space, but we need to convert
          % it to a tensor
          QF = face_node_tensor(G, 'f', 'n', ...
                                'values', qf_all, ...
                                'boundary_only', true);
       end
       % extend weight tensor to face-node-dim space ('f', 'n', 'd')
       QF = QF * SparseTensor(ones(G.griddim, 1), 'd');
       
       % expand face forces tensor to face-node-dim space (only boundary)
       F = el_bc.force_bc ^ face_node_tensor(G, 'f', 'n', 'boundary_only', true);

       % Contract in faces to get weighted forces in node-dim space
       FQF = QF .* F;
       FQF = FQF.contract('f');
              
       % get components @@ (expand to support ADI)
       fbc = FQF.asVector({'d', 'n'}, [G.griddim, G.nodes.num]);
            
       rhs   = rhs + fbc;
    end
    toc
    tic; fprintf('Reducing degrees of freedom.\n');
    % Reduce the degrees of freedom
    rhs = rhs(~isdirdofs);
    A   = S(~isdirdofs, ~isdirdofs);
    
    %% make matrix perfectly symmetric (numericaly)
    A   = (A + A') / 2; % The matrix is theoretically symmetric, make sure that it
                        % is also symmetric numerically
    toc
    %% Solve the equation system (unless 'no_solve' option)
    tic; fprintf('Solving system.\n');
    if(opt.no_solve)
        x=nan(sum(~isdirdofs),1);
    else
       if isa(rhs, 'ADI')
          b = value(rhs);
       else 
          b = rhs;
       end
        x   = opt.linsolve(A, b);
    end
    
    toc
    
    u   = nan(ndof, 1);

    u(isdirdofs)  = value(u_bc(isdirdofs));
    u(~isdirdofs) = x;
    uu = reshape(u, G.griddim, [])';

    if(nargout == 2)
        extra.A    = A;
        extra.S    = S;
        extra.rhs  = rhs;
        vdiv       = VEM_div(G);
        extra.disc = struct('A'         , A                        , ...
                            'isdirdofs' , isdirdofs                , ...
                            'rhs'       , rhs                      , ...
                            'V_dir'     , u_bc                     , ...
                            'ovol_div'  , vdiv                     , ...
                            'gradP'     , vdiv(:    , ~isdirdofs)' , ...
                            'div'       , vdiv(:    , ~isdirdofs)  , ...
                            'divrhs'    , vdiv * u_bc);
        if isa(E, 'ADI') || isa(nu, 'ADI')
           % we must add derivatives of system matrix with respect to
           % material parameters
           assert(~opt.experimental_scaling) % @@ derivative for experimental
                                             % scaling not yet implemented
           extra.Ax_derivs = compute_Ax_derivs(G, u, extra, E, nu, opt.alpha_scaling, ...
                                               isdirdofs);
        end
    end
end

function ax_derivs = compute_Ax_derivs(G, u, extra, E, nu, gamma, dirdofs)

   assert(G.griddim==3);
   N = G.cells.num;
   cpos = reshape(repmat(1:N, 6, 1), [], 1);
   rep6 = sparse((1:6*N)', cpos, 1, 6 * N, N); % repeat each vector entry 6 times

   nlc = diff(G.cells.nodePos);
   dim = 3;
   
   % repeat each entry '3n' times, where 'n' is the number of nodes for a given cell
   repX = sparse((1:dim*sum(nlc))', rldecode((1:N)', nlc * dim), 1); 
                                                                     
   D_dE = bsxfun(@rdivide, extra.D, rep6 * value(E));  %  d/dE (D)
   
   % we have D = Ds * Dm (where ds is scalar and dm is a matrix)
   Ds = E ./ ((1+nu) .* (1-2*nu)); 
   Dm = bsxfun(@rdivide, extra.D, rep6 * value(Ds));
   
   % computing d/dnu (D), which makes use of the definitions of Ds and Dm above
   Ds_dnu = E .* (1 + 4 * nu) ./ ( (1+nu) .* (1 - 2 * nu) ).^2;
   
   Dm_dnu = spones(extra.D);
   Dm_diag = repmat([-1,-1,-1, -4,-4,-4], 1, G.cells.num);
   Dm_dnu(logical(eye(numel(Dm_diag)))) = Dm_diag;
   
   D_dnu = bsxfun(@times, Dm, rep6 * value(Ds_dnu)) + ...
           bsxfun(@times, Dm_dnu, rep6 * value(Ds));
   
   DNC = diag(extra.NC' * extra.NC);
   trDNC = sum(reshape(DNC, 6, []), 1)';
   c = gamma ./ trDNC .* G.cells.volumes; 
   trD = Ds .* 3 .* (3-5*value(nu)); 
   alpha_dE = c .* trD ./ value(E);
   alpha_dnu = - c .* value(E) .* 6 .* (value(nu)-1) .* (5 * value(nu) -1 ) ./ ...
                                    ( (value(nu) + 1) .* (1 - 2 * value(nu)) ).^2;
   S_dE = spdiags(repX * value(alpha_dE), 0, size(repX, 1), size(repX, 1)); 
   S_dnu = spdiags(repX * value(alpha_dnu), 0, size(repX, 1), size(repX, 1)); 
   
   % compute derivatives based on control variables
   ejac = []; nujac = [];
   if isa(E, 'ADI')
      ejac = [E.jac{:}];
   end
   if isa(nu, 'ADI')
      nujac = [nu.jac{:}];
   end
   if isempty(ejac)
      ejac = 0 * nujac;
   elseif isempty(nujac)
      nujac = 0 * ejac;
   end
   num_ders = size(ejac, 2);
      
   ImPP = extra.I - extra.PP;
   ax_derivs = sparse(numel(u), num_ders);
   for i = 1:num_ders
      
      % compute total derivative for each cell
      Kdu = ...
          extra.WC * ( ...
              bsxfun(@times, D_dE, rep6 * (ejac(:,i) .* G.cells.volumes)) + ...
              bsxfun(@times, D_dnu, rep6 * (nujac(:,i) .* G.cells.volumes))) * ...
          extra.WC' + ...          
          ImPP' * (...
              bsxfun(@times, S_dE, repX * ejac(:,i)) + ...
              bsxfun(@times, S_dnu, repX * nujac(:,i))) * ...
          ImPP;
      
      % Kdu = extra.WC * bsxfun(@times, D_dE, rep6 * ejac(:,i)) * extra.WC' + ...
      %       ImPP' * bsxfun(@times, S_dE, repX * ejac(:,i)) * ImPP + ...
      %       extra.WC * bsxfun(@times, D_dnu, rep6 * nujac(:,i)) * extra.WC' + ...
      %       ImPP' * bsxfun(@times, S_dnu, repX * nujac(:,i)) * ImPP;
      
      % assemble
      Kdu = extra.assemb * Kdu * extra.assemb';

      ax_derivs(:, i) = Kdu * u; % @@ how can we speed up here?
   end
   ax_derivs = ax_derivs(~dirdofs, :);
end


function f = calculateVolumeTerm(G, load, qc_all, qcvol, opt)

    cells  = [1:G.cells.num]';
    inodes = mcolon(G.cells.nodePos(cells), G.cells.nodePos(cells + 1) - 1');
    nodes  = G.cells.nodes(inodes);

    switch opt.force_method

      case 'node_force'
        % Evaluate forces at nodes. The result is weighted, using adjacent
        % cell volume contributions, see paper [Gain et al: doi:10.1016/j.cma.2014.05.005]

        X = G.nodes.coords(nodes, :);
        w = qcvol;
        ll = bsxfun(@times, load(X), w)';

      case  'cell_force_baric'
        %
        % Evaluate the forces at nodes. Then, from the nodal values, compute the (exact)
        % L^2 projection on each cell and, using this cell values, assemble
        % the load term in term of the degrees of freedom.
        %
        % The VEM theory tells us that there exist a virtual basis such that the
        % two steps above can be done exactly. See Ahmad et al (doi:10.1016/j.camwa.2013.05.015)
        %

        nlc = G.cells.nodePos(cells + 1) - G.cells.nodePos(cells);
        X = rldecode(G.cells.centroids(cells, :), nlc);
        lcellnum = rldecode(cells, nlc);
        BB = nan(numel(cells), 1);
        for i = 1:G.griddim
            BB(:, i) = accumarray(lcellnum, G.nodes.coords(nodes, i), [numel(cells), 1]);
        end
        fac  = accumarray(lcellnum, 1, [numel(cells), 1]);
        BB   = bsxfun(@rdivide, BB, fac);
        XB   = X - rldecode(BB, nlc);
        vols = rldecode(G.cells.volumes(cells, :), nlc);

        % Weights to calculate the volume force term from the nodal values.
        if(G.griddim == 3)
            w = (vols ./ rldecode(nlc, nlc) + sum(qc_all(inodes, :) .* (XB), 2));
        else
            assert(G.griddim == 2)
            w = (vols ./ rldecode(nlc, nlc) + sum(qc_all(inodes, :) .* (XB), 2));
        end
        ll   = bsxfun(@times, load(X), w)';

      case 'cell_force'
        % Evaluate the force at the cell centroids. Then, for each node, sum up each
        % adjacent cell contributions after weighting them with volume
        % contribution.

        nlc = G.cells.nodePos(cells + 1) - G.cells.nodePos(cells);
        X   = rldecode(G.cells.centroids(cells, :), nlc);
        w   = qcvol;
        ll  = bsxfun(@times, load(X), w)';

      case 'dual_grad_type'
        % For the virtual basis, there exists a natural divergence operator (from node
        % values to cell values) that can be computed exactly. This operator
        % does not depend on the stabilisation term which is chosen for the
        % stiffness matrix
        %
        % By duality we can define a gradient operator (from cell-values to
        % node values). When the force can be expressed as a gradient, this
        % gives us a way to compute the load term which is implemented here.
        %
        % Such computation has appeared to be more stable, see [Andersen et al: http://arxiv.org/abs/1606.09508v1].
        %
        %
        nlc     = G.cells.nodePos(cells + 1) - G.cells.nodePos(cells);
        X       = rldecode(G.cells.centroids(cells, :), nlc);
        rel_vec = -(X-G.nodes.coords(nodes, :));
        ll      = bsxfun(@times, load(X), qc_all.*rel_vec)';

      otherwise
        error('No such force  calculation')
    end

    ndof = G.griddim * G.nodes.num;
    dofs = mcolon(G.griddim * (nodes - 1) + 1, G.griddim * (nodes - 1) + G.griddim)';
    f    = accumarray(dofs(:), ll(:), [ndof, 1]);

end