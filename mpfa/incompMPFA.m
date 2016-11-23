function state = incompMPFA(state, g, T, fluid, varargin)
%Solve incompressible flow problem (fluxes/pressures) using MPFA-O method.
%
% SYNOPSIS:
%   state = incompMPFA(state, G, T, fluid)
%   state = incompMPFA(state, G, T, fluid, 'pn1', pv1, ...)
%
% DESCRIPTION:
%   This function assembles and solves a (block) system of linear equations
%   defining interface fluxes and cell pressures at the next time step in a
%   sequential splitting scheme for the reservoir simulation problem
%   defined by Darcy's law and a given set of external influences (wells,
%   sources, and boundary conditions).
%
%   This function uses a multi-point flux approximation (MPFA) method with
%   minimal memory consumption within the constraints of operating on a
%   fully unstructured polyhedral grid structure.
%
% REQUIRED PARAMETERS:
%   state  - Reservoir and well solution structure either properly
%            initialized from functions 'initResSol' and 'initWellSol'
%            respectively, or the results from a previous call to function
%            'incompMPFA' and, possibly, a transport solver such as
%            function 'implicitTransport'.
%
%   G, T   - Grid and half-transmissibilities as computed by the function
%            'computeMultiPointTrans'.
%
%   fluid  - Fluid object as defined by function 'initSimpleFluid'.
%
% OPTIONAL PARAMETERS (supplied in 'key'/value pairs ('pn'/pv ...)):
%   wells  - Well structure as defined by functions 'addWell' and
%            'assembleWellSystem'.  May be empty (i.e., W = struct([]))
%            which is interpreted as a model without any wells.
%
%   bc     - Boundary condition structure as defined by function 'addBC'.
%            This structure accounts for all external boundary conditions to
%            the reservoir flow.  May be empty (i.e., bc = struct([])) which
%            is interpreted as all external no-flow (homogeneous Neumann)
%            conditions.
%
%   src    - Explicit source contributions as defined by function
%            'addSource'.  May be empty (i.e., src = struct([])) which is
%            interpreted as a reservoir model without explicit sources.
%
%   LinSolve     - Handle to linear system solver software to which the
%                  fully assembled system of linear equations will be
%                  passed.  Assumed to support the syntax
%
%                        x = LinSolve(A, b)
%
%                  in order to solve a system Ax=b of linear equations.
%                  Default value: LinSolve = @mldivide (backslash).
%
%   MatrixOutput - Whether or not to return the final system matrix 'A' to
%                  the caller of function 'incompMPFA'.
%                  Logical.  Default value: MatrixOutput = FALSE.
%
%   Verbose      - Whether or not to time portions of and emit informational
%                  messages throughout the computational process.
%                  Logical.  Default value dependent on global verbose
%                  setting in function 'mrstVerbose'.
%
% RETURNS:
%   xr - Reservoir solution structure with new values for the fields:
%          - pressure     -- Pressure values for all cells in the
%                            discretised reservoir model, 'G'.
%          - boundaryPressure --
%                            Pressure values for all boundary interfaces in
%                            the discretised reservoir model, 'G'.
%          - flux         -- Flux across global interfaces corresponding to
%                            the rows of 'G.faces.neighbors'.
%          - A            -- System matrix.  Only returned if specifically
%                            requested by setting option 'MatrixOutput'.
%
%   xw - Well solution structure array, one element for each well in the
%        model, with new values for the fields:
%           - flux     -- Perforation fluxes through all perforations for
%                         corresponding well.  The fluxes are interpreted
%                         as injection fluxes, meaning positive values
%                         correspond to injection into reservoir while
%                         negative values mean production/extraction out of
%                         reservoir.
%           - pressure -- Well pressure.
%
% NOTE:
%   If there are no external influences, i.e., if all of the structures
%   'W', 'bc', and 'src' are empty and there are no effects of gravity,
%   then the input values 'xr' and 'xw' are returned unchanged and a
%   warning is printed in the command window. This warning is printed with
%   message ID
%
%           'incompMPFA:DrivingForce:Missing'
%
% EXAMPLE:
%    G   = computeGeometry(cartGrid([3,3,5]));
%    f   = initSingleFluid('mu' ,    1*centi*poise     , ...
%                          'rho', 1014*kilogram/meter^3);
%    rock.perm = rand(G.cells.num, 1)*darcy()/100;
%    bc  = pside([], G, 'LEFT', 2);
%    src = addSource([], 1, 1);
%    W   = verticalWell([], G, rock, 1, G.cartDims(2), ...
%                       (1:G.cartDims(3)), 'Type', 'rate', 'Val', 1/day(), ...
%                       'InnerProduct', 'ip_tpf');
%    W   = verticalWell(W, G, rock, G.cartDims(1),   G.cartDims(2), ...
%                       (1:G.cartDims(3)), 'Type', 'bhp', ...
%                       'Val',  1*barsa(), 'InnerProduct', 'ip_tpf');
%    T   = computeMultiPointTrans(G, rock);
%
%    state = initState(G, W, 100*barsa);
%    state = incompMPFA(state, G, T, f, 'bc', bc, 'src', src, ...
%                       'wells', W, 'MatrixOutput',true);
%
%    plotCellData(G, xr.pressure);
%
% SEE ALSO:
%   computeMultiPointTrans, addBC, addSource, addWell, initSingleFluid,
%   initResSol, initWellSol, mrstVerbose.

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


% Written by Jostein R. Natvig, SINTEF ICT, 2009.

   opt = struct('bc', [], 'src', [], 'wells', [], ...
                'LinSolve',     @mldivide,        ...
                'MatrixOutput', false,            ...
                'Verbose',mrstVerbose);
   opt = merge_options(opt, varargin{:});

   g_vec   = gravity();
   no_grav = ~(norm(g_vec) > 0); %(1 : size(g.nodes.coords,2))) > 0);
   if all([isempty(opt.bc)   , ...
           isempty(opt.src)  , ...
           isempty(opt.wells), no_grav]),
      warning(id('DrivingForce:Missing'),                       ...
              ['No external driving forces present in model--', ...
               'state remains unchanged.\n']);
   end

   cellNo = rldecode(1:g.cells.num, diff(g.cells.facePos), 2) .';
   cf     = g.cells.faces(:,1);
   nf     = g.faces.num;
   nc     = g.cells.num;
   nw     = length(opt.wells);
   n      = nc + nw;

   [totmob, omega, rho] = dynamic_quantities(state, fluid);
   
   % Needed after introduction of gravity
   TT=T;
   Tg = T.Tg;
   T  = T.T;

   C      = sparse(1:size(g.cells.faces, 1), cellNo, 1);
   ind    = [(1:g.cells.num)'; max(g.faces.neighbors, [], 2)];

   %totmob = fluid.Lt(state);
   totmob_mat = spdiags(totmob(ind), 0, numel(ind),numel(ind));
   T      = TT.T * totmob_mat;

   totmob_mat = spdiags(rldecode(totmob, diff(g.cells.facePos)*2), 0, ...
                        size(g.cells.faces,1)*2, size(g.cells.faces,1)*2);
   totFace_mob=1./accumarray(g.cells.faces(:,1),1./totmob(rldecode([1:g.cells.num]', diff(g.cells.facePos))));
   b  = any(g.faces.neighbors==0, 2);
   totFace_mob(~b)=totFace_mob(~b)/2;
   tothface_mob_mat=diag(TT.d1*totFace_mob);
   %Tg     = Tg * totmob_mat;

   % identify internal faces
   i  = all(g.faces.neighbors ~= 0, 2);

   % Boundary conditions and source terms.
   % Note: Function 'computeRHS' is a modified version of function
   % 'computePressureRHS' which was originally written to support the
   % hybrid mimetic method.
   %[ff, gg, hh, grav, dF, dC] = computePressureRHS(g, omega, ...
   %                                                opt.bc, opt.src);
   %[~, gg, hh, grav, dF, ~] = computePressureRHS(g, omega, ...
   %                                                opt.bc, opt.src);
   [~, gg, hh, ~, dF, ~] = computePressureRHS(g, omega, ...
                                                   opt.bc, opt.src);
   % add gravity contribution for each mpfa half face
   grav     = -omega(TT.cno) .* (TT.R * reshape(g_vec(1:g.griddim), [], 1));  
   
   b  = any(g.faces.neighbors==0, 2);
   bf  = any(g.faces.neighbors==0, 2);
   I1 = [(1:g.cells.num)'; g.cells.num + find(b)];
   %D  = sparse(1:size(g.cells.faces,1), double(g.cells.faces(:,1)), 1);
   %D=TT.D;
   sb = full(sum(TT.D, 1)) == 1;
   %d1=TT.d1;
   %c1=TT.c1;
   %T=c1'*Do*iDoBDo*Do';
    %end
   %T = Tg*[C, -D(:,b)*d1(b,:)];
   %C=TT.C;
   %T =  TT.hfhf*[C, -D(:,b)];
   % defin \grad operators to all mpfa sidnes (That is halv of a normal
   % face)
   cf_mtrans=TT.Do'*TT.hfhf*[TT.C, -TT.D(:,sb)];
   % define div operaor form mfpa sides to celle values in addtion to the
   % fluxes out of boundary.
   e_div =  [TT.C, -TT.D(:,sb)]'*TT.Do;
   % multiply fluxes with harmonic mean of mobility
   % this to avid for re asssembly
   % to be equivalent coupled reservoir simulation the value of
   % sum of upwind mobility should be used.
   A=e_div*tothface_mob_mat*cf_mtrans;
   %dghf=TT.hfhf * grav;
   %rhs_g= [TT.C, -TT.D(:,sb)]'*dghf;
   dghf= TT.Do'*TT.hfhf * grav;
   rhs_g= [TT.C, -TT.D(:,sb)]'*TT.Do*tothface_mob_mat*dghf;
   %A  = [TT.C, -TT.D(:,sb)]' *totmob_mat* TT.hfhf*[TT.C, -TT.D(:,sb)];
   %A  = [C, -D(:,b)]' * Tg(:,I1);
   % Gravity contribution for each face
   cf  = g.cells.faces(:,1);
   j   = i(cf) | dF(cf);
   s   = 2*(g.faces.neighbors(cf, 1) == cellNo) - 1;
   %fg  = [C, -D(:,b)]' * (Tg * grav);
   %fg  = accumarray(cf(j), grav(j).*s(j), [g.faces.num, 1]);

   hh_tmp = TT.d1*hh;
   rhs = [gg; -hh_tmp(sb)];
   rhs=rhs+rhs_g;
   %% Eliminate all but the cellpressure
   %BB=A(nc+1:end,nc+1:end);
   %AA=A(1:nc,1:nc);
   %DD=A(nc+1:end,1:nc);
   %DU=A(1:nc,nc+1:end);

   %A=AA-DU*inv(BB)*DD;
   %rhs=rhs(1:nc)+DU*inv(BB)*rhs(nc+1:end);

   %B=A(nc+1:end,
   %A=inv(A(nc+1:end)A(1:nc,1:nc)

   %% Dirichlet condition
   % If there are Dirichlet conditions, move contribution to rhs.  Replace
   % equations for the unknowns by speye(*)*x(dF) = dC.
   %% add gravity

   factor = A(1,1);
   assert (factor > 0)
   %subp=nan(size(A,1),1);
   if any(dF),
      dF_tmp=TT.d1(sb,:)*dF
      ind        = [false(g.cells.num, 1) ; dF_tmp>0];
      is_press = strcmpi('pressure', opt.bc.type);
      face     = opt.bc.face (is_press);
      bcval    = opt.bc.value (is_press);
      dC_tmp=TT.d1(sb,face)*bcval;
      rhs        = rhs - A(:,g.cells.num+1:end)*dC_tmp;
      rhs(ind)   = factor*dC_tmp(dF_tmp>0);
      A(ind,:)   = 0;
      A(:,ind)   = 0;
      A(ind,ind) = factor * speye(sum(ind));
   end
   

   %remove
   %A=A(1:nc,1:nc);
   nnp=length(rhs);
   rhs=[rhs;zeros(nw, 1)];

   %%%%%%%%%%%%%%%%%%%
   % add well equations
   C    = cell (nnp, 1);
   D    = zeros(nnp, 1);
   W    = opt.wells;
   d  = zeros(g.cells.num, 1);
   for k = 1 : nw,
      wc       = W(k).cells;
      nwc      = numel(wc);
      w        = k + nnp;

      wi       = W(k).WI .* totmob(wc);

      dp       = norm(gravity()) * W(k).dZ*sum(rho .* W(k).compi, 2);
      d   (wc) = d   (wc) + wi;

      if     strcmpi(W(k).type, 'bhp'),
         ww=max(wi);
         %ww=1.0;
         rhs (w)  = rhs (w)  + ww*W(k).val;
         rhs (wc) = rhs (wc) + wi.*(W(k).val + dp);
         C{k}     = -sparse(1, nnp);
         D(k)     = ww;

      elseif strcmpi(W(k).type, 'rate'),
         rhs (w)  = rhs (w)  + W(k).val;
         rhs (wc) = rhs (wc) + wi.*dp;

         C{k}     =-sparse(ones(nwc, 1), wc, wi, 1, nnp);
         D(k)     = sum(wi);

         rhs (w)  = rhs (w) - wi.'*dp;

      else
         error('Unsupported well type.');
      end
   end

   C = vertcat(C{:});
   D = spdiags(D, 0, nw, nw);
   A = [A, C'; C D];
   A = A+sparse(1:nc,1:nc,d,size(A,1),size(A,2));

   %if norm(gravity()) > 0,

   %        rhs = rhs + T(:,I1)'*fg(cf);
   %end
   if ~any(dF) && (isempty(W) || ~any(strcmpi({W.type }, 'bhp'))),
      A(1) = A(1)*2;
   end
   ticif(opt.Verbose);
   p = opt.LinSolve(A, rhs);

   tocif(opt.Verbose);

   %% ---------------------------------------------------------------------
   dispif(opt.Verbose, 'Computing fluxes, face pressures etc...\t\t');
   ticif (opt.Verbose);

   % Reconstruct face pressures and fluxes.
   %fpress     =  ...
%          accumarray(g.cells.faces(:,1), (p(cellNo)+grav).*T, [g.faces.num,1])./ ...
%          accumarray(g.cells.faces(:,1), T, [G.faces.num,1]);


   % Neumann faces
   b         = any(g.faces.neighbors==0, 2);
   %fpress(b) = fpress(b) - hh(b)./ft(b);


   % Contribution from gravity
   %fg         = accumarray(cf, grav.*sgn, [nf, 1]);
   %fpress(~i) = fpress(~i) + fg(~i);

   % Dirichlet faces
   %fpress(dF) = dC;


   % Sign for boundary faces
   %sgn  = 2*(G.faces.neighbors(~i,2)==0)-1;
   %ni   = G.faces.neighbors(i,:);
   %flux = -accumarray(find(i),  ft(i) .*(p(ni(:,2))-p(ni(:,1))-fg(i)), [nf, 1]);
   %c    = sum(G.faces.neighbors(~i,:),2) ;
   %fg  = accumarray(cf, grav, [nf, 1]);
   %flux(~i) = -sgn.*ft(~i).*( fpress(~i) - p(c) - fg(~i) );
   %flux = -sgn.*ft((fpress(~i)-p(c)-grav));

   state.pressure(1 : nc) = p(1 : nc);
   %state.flux(:)          = cellFlux2faceFlux(g, T(:, I1) * p(1 : nnp));
   %C=TT.C;
   %T =  TT.hfhf*[C, -D(:,b)];
   %A  = [C, -D(:,b)]' *  TT.hfhf*[C, -D(:,b)];
   omega(TT.cno)
   
   %state.flux = TT.d1'*TT.Do'*totmob_mat*TT.hfhf*[TT.C, -TT.D(:,sb)]*(p);%?????-dg);
   state.flux = TT.d1'*(tothface_mob_mat*cf_mtrans*p(1:nnp) - tothface_mob_mat*dghf);%?????-dg);
   state.flux(~b)=state.flux(~b)/2;
   state.boundaryPressure = p(nc + 1 : nnp);

   for k = 1 : nw,
      wc       = W(k).cells;
      dp       = norm(gravity()) * W(k).dZ*sum(rho.*W(k).compi, 2);
      state.wellSol(k).flux = W(k).WI.*totmob(wc).*(p(nnp+k) + dp - p(wc));
      state.wellSol(k).pressure = p(nnp + k);
   end

   if opt.MatrixOutput,
      state.A   = A;
      state.rhs = rhs;
   end

   tocif(opt.Verbose);
end

%--------------------------------------------------------------------------
% Helpers follow.
%--------------------------------------------------------------------------

function s = id(s)
   s = ['incompMPFA:', s];
end

%--------------------------------------------------------------------------

function [totmob, omega, rho] = dynamic_quantities(state, fluid)
   [mu, rho] = fluid.properties(state);
   s         = fluid.saturation(state);
   kr        = fluid.relperm(s, state);

   mob    = bsxfun(@rdivide, kr, mu);
   totmob = sum(mob, 2);
   omega  = sum(bsxfun(@times, mob, rho), 2) ./ totmob;
end
