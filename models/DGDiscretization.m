classdef DGDiscretization < WENODiscretization
    
    properties

        degree              % Degree of discretization, dG(degree)
        basis               % Type of basis functions. Standard is tensor 
                            % products of Legendre polynomials.
%         dim                 % Dimension of disc to facilitate e.g. 2D 
                            % simulations on horizontal slice of 3D
                            % reservoir
        
        useUnstructCubature % Bool to tell the method to use experimental 
                            % unstructured cubature class
        volumeCubature      % Cubature for volume integrals
        surfaceCubature     % Cubature for surface integrals
        
        jumpTolerance       % Tolerance for sat jumps across interfaces
        outTolerance        % Tolerance for sat outside [0,1]
        meanTolerance       % Tolerance of mean sat outside [0,1]
        limiterType         % Type of limiter
        plotLimiterProgress % 1d plot of result before and after limiter
        
        velocityInterp      % Function for mapping face fluxes to cell
                            % velocity/ies
        upwindType          % Type of upwind calculation
        
        internalConnParent  % If we only solve on subset of full grid, we
                            % must keep tract of internal connections in 
                            % the full grid.
        
    end
    
    methods
        
        %-----------------------------------------------------------------%
        function disc = DGDiscretization(model, varargin)
            
            disc = disc@WENODiscretization(model);
            
            G = disc.G;
            
            % Standard dG properties
            disc.degree = 1;
            disc.basis  = 'legendre';
%             disc.dim    = G.griddim;
            
            % Limiter tolerances
            disc.jumpTolerance = 0.2;
            disc.outTolerance  = 1e-4;
            disc.meanTolerance = 1e-4;
            disc.limiterType   = 'kill';
            disc.plotLimiterProgress = false;
            
            % Specifics for reordering
            disc.internalConnParent  = disc.internalConn;
            disc.G.parent            = G;
            
            disc = merge_options(disc, varargin{:});
            
            % Replace basis string by dG basis object
            disc.basis  = dgBasis(disc.dim, disc.degree, disc.basis);
            
            % Set up velocity interpolation
            disc.velocityInterp = velocityInterpolation(G, 'mimetic');
            disc.upwindType     = 'potential';
            
            % Create cubatures
            disc.useUnstructCubature = false;
            prescision = disc.degree + 1;
            if G.griddim == 2
                if disc.degree == 0 || disc.useUnstructCubature
                    disc.volumeCubature = Unstruct2DCubature(G, prescision, disc.internalConn);
                else
                    disc.volumeCubature = TriangleCubature(G, prescision, disc.internalConn);
                end
                disc.surfaceCubature = LineCubature(G, prescision, disc.internalConn);
            else
                if disc.degree == 0 || disc.useUnstructCubature
                    disc.volumeCubature  = Unstruct3DCubature(G, prescision, disc.internalConn);
                    disc.surfaceCubature = Unstruct2DCubature(G, prescision, disc.internalConn);
                else
                    disc.volumeCubature  = TetrahedronCubature(G, prescision, disc.internalConn);
                    disc.surfaceCubature = TriangleCubature(G, prescision, disc.internalConn);
                end
            end
            
        end
        
        %-----------------------------------------------------------------%
        function state = assignDofFromState(disc, state)
            % Assign dofs from state (typically initial state). All dofs
            % for dofNo > 0 are set to zero.
            
            % Set degree to disc.degree in all cells
            state.degree = repmat(disc.degree, disc.G.cells.num, 1);
            
            % Create vector dofPos for position of dofs in state.sdof
            state = disc.updateDofPos(state);
            
            state.nDof  = disc.getnDof(state);
            sdof        = zeros(sum(state.nDof), size(state.s,2));
            
            % Assing constant dof equal to cell saturation in each cell
            ix          = disc.getDofIx(state, 1);
            sdof(ix, :) = state.s;

            state.sdof = sdof;

        end
        
        %-----------------------------------------------------------------%
        function state = updateDofPos(disc, state)
            % Update dosfPos (position of dofs in state.sdof) based on
            % changes in state.degree, or create dofPos vector if it does
            % not exist. Dofs for cell i are found in
            %
            %   state.sdof(dofPos(:,i),:),
            %
            % Zeros are included to easily map dofs from one timestep to
            % the next.

            dp = reshape((1:disc.G.cells.num*disc.basis.nDof)', disc.basis.nDof, []);
            
            nd = disc.getnDof(state);
            subt = cumsum([0; disc.basis.nDof - nd(1:end-1)]);
            [ii, jj, v] = find(dp);

            if size(ii,1) == 1, ii = ii'; end
            if size(jj,1) == 1, jj = jj'; end
            if size(v ,1) == 1, v  =  v'; end

            cnDof = cumsum(nd);

            v = v - subt(jj);
            v(v > cnDof(jj)) = 0;
            dp = full(sparse(ii, jj, v));
            
            state.nDof   = nd;
            state.dofPos = dp;
            
        end
        
        %-----------------------------------------------------------------%
        function ix = getDofIx(disc, state, dofNo, cells, includezero)
            % Get position of dofs in state.sdof for a given cell
            %
            % PARAMETERS:
            %   state       - State with field sdof
            %   dofNo       - Dof number we want the position of. Empty
            %                 dofNo returns position of all dofs for cells
            %   cells       - Cells we want the dof position for. If empty,
            %                 positions of dofNo are returned for all cells.
            %   includeZero - Boolean indicating of we should include zeros
            %                 or not (see updateDofPos)
            %
            % RETURNS:
            %   ix - Indices into states.sdof. Dof number dofNo for cells
            %        are found in state.sdof(ix,:);

            G = disc.G;
            if nargin < 3 || (numel(dofNo) == 1 && dofNo == Inf)
                % dofNo not given, return ix for all dofs
                dofNo = 1:disc.basis.nDof;
            elseif nargin < 4 || (numel(cells) == 1 && cells == Inf)
                % Cells not given, return ix for all cels
                cells = 1:G.cells.num;
            end
            
            ix = state.dofPos(dofNo, cells);
            ix = ix(:);
            
            if nargin < 5
                includezero = false;
            end
            if ~includezero
                ix(ix == 0) = [];
            end
              
        end
        
        %-----------------------------------------------------------------%
        function [xhat, translation, scaling] = transformCoords(disc, x, cells, inverse, useParent)
            % Transfor coordinates from physical to reference coordinates. 
            %
            % PARAMETERS:
            %   x         - Coordinates in physical space
            %   cells     - Cells we want reference coordinates for, cells(ix)
            %               are used when transforming x(ix,:)
            %   inverse   - Boolean indicatiing if we are mapping 
            %               to (inverse = false) or from (inverse = true)
            %               reference coordiantes. Default = false.
            %   useParent - Boolean indicating if we are working on the
            %               full grid (G.parent) or a subgrid.
            %
            % RETURNS:
            %   xhat        - Transformed coordinates
            %   translation - Translation applied to x
            %   scaling     - Scaling applied to x
            
            G = disc.G;
            
            if nargin < 4, inverse   = false; end
            if nargin < 5, useParent = false; end
            
            if isfield(G, 'mappings') && useParent
                G = G.parent;
            end
            
            % Coordinates are centered in cell center
            translation = -G.cells.centroids(cells,:);
            if isfield(G.cells, 'dx')
                % Scaling found from dimensions of minimum bounding box
                % aligned with coordinate axes that contains the cell
                scaling = 1./(G.cells.dx(cells,:)/2);
            else
                % If it G.cells.dx is not computed, we use approximation
                dx = G.cells.volumes(cells).^(1/G.griddim);
                scaling = 1./(dx/2);
            end
            
            if ~inverse
                xhat = (x + translation).*scaling;
                xhat = xhat(:, 1:disc.dim);
                scaling     = scaling(:, 1:disc.dim);
                translation = translation(:, 1:disc.dim);
%                 assert(all(all(abs(xhat)<=1)))
            else
                xhat = x./scaling - translation;
            end
               
        end
        
        %-----------------------------------------------------------------%
        function nDof = getnDof(disc, state)
            % Get number of dofs for each cell from degree.
            
            if disc.degree < 0
                nDof = 0;
            else
                nDof = factorial(state.degree + disc.dim)...
                       ./(factorial(disc.dim).*factorial(state.degree));
            end
            
        end
        
        %-----------------------------------------------------------------%
        function state = mapDofs(disc, state, state0)
            % Map dofs from state0 to state, typically from one timestep to
            % the next, when we start with maximum number of dofs in all
            % cells.
            
            % Update dofPos
            state = disc.updateDofPos(state);
            
            if all(state.nDof == state0.nDof)
                state.sdof = state0.sdof;
            else
                sdof = zeros(sum(state.nDof), size(state0.sdof,2));
                for dofNo = 1:disc.basis.nDof
                    % We may be solving only a subset of the gridcells, so
                    % we include zeros in dofIx to keep track of where old
                    % dofs maps to new ones.
                    ix  = disc.getDofIx(state , dofNo, (1:disc.G.cells.num)', true);
                    ix0 = disc.getDofIx(state0, dofNo, (1:disc.G.cells.num)', true);
                    sdof(ix(ix0 > 0 & ix > 0),:) = state0.sdof(ix0(ix0 > 0 & ix > 0),:);
                end
                state.sdof = sdof;
            end

        end
        
        %-----------------------------------------------------------------%
        function sat = evaluateSaturation(disc, x, cells, dof, state)
            % Evaluate saturation at coordinate x as seen from cell
            %
            % PARAMETERS:
            %   x     - Coordinates in REFERENCE space
            %   cells - Cells we want saturation value in (saturation
            %               evaluated at coordinate x(ix,:) in cell(ix)
            %   dof   - degrees of freedom for the phase we want the
            %               saturation
            %   state - State we get our dofs from (only used to get dofIx)
            %
            % RETURNS:
            %   sat - Saturation values at coordinates x as seen from cells
            
            psi     = disc.basis.psi;
            nDof    = state.nDof;
            nDofMax = disc.basis.nDof;
            
            ix = disc.getDofIx(state, 1, cells);
            sat = dof(ix).*0;
            for dofNo = 1:nDofMax
                keep = nDof(cells) >= dofNo;
                ix = disc.getDofIx(state, dofNo, cells(keep));
                if all(keep)
                    sat = sat + dof(ix).*psi{dofNo}(x(keep,:));
                else
                    sat(keep) = sat(keep) + dof(ix).*psi{dofNo}(x(keep,:));
                end

            end
            
        end
        
        %-----------------------------------------------------------------%
        function state = getCellSaturation(disc, state)
            % Get average cell saturaion, typically assigned to state.s

            % Get cubature for all cells, transform coordinates to ref space
            [W, x, cellNo, ~] = disc.getCubature((1:disc.G.cells.num)', 'volume');
            x = disc.transformCoords(x, cellNo);
            
            sdof = state.sdof;
            nPh  = size(sdof,2);
            s    = zeros(disc.G.cells.num, nPh);
            for phNo = 1:nPh
                s(:,phNo) = W*disc.evaluateSaturation(x, cellNo, sdof(:,phNo), state);
            end
            s = s./disc.G.cells.volumes;
            
            state.s = s;
            
        end
        
        function dotProduct = dot(disc,u,v)
            dotProduct = sum(u.*v, 2);
        end
        
        %-----------------------------------------------------------------%
        function I = cellInt(disc, model, fun, cells, state, state0, varargin)
            % Integrate integrand over cells
            %
            % PARAMETERS:
            %   model    - Model, which contains information on how the
            %              integrand looks like
            %   fun      - Integrand function handle
            %   cells    - Cells over which we will integrate fun
            %   state, state0 - States from current and prev timestep,
            %              to be used for dofPos
            %   varargin - Variables passed to model for integrand
            %              evalauation. varargin{1} MUST be an AD object of
            %              dofs for all cells in the grid.
            %
            % RETURNS:
            %   I - Integrals int(fun*psi{dofNo}) for dofNo = 1:nDof over
            %       all cells
            
            psi      = disc.basis.psi;      % Basis functions
            grad_psi = disc.basis.grad_psi; % Gradient of basis functions
            nDof     = state.nDof;          % Number of dofs per cell
            nDofMax  = disc.basis.nDof;     % Maximum number of dofs
            
            % Empty cells means all cells in grid
            if isempty(cells)
                cells = (1:disc.G.cells.num)';
            end
            
            % Get cubature for all cells, transform coordinates to ref space
%             [W, x, cellNo, ~] = disc.volumeCubature.getCubature2(cells, 'volume');
            [W, x, cellNo, ~] = disc.getCubature(cells, 'volume');
            [x, ~, scaling]   = disc.transformCoords(x, cellNo);

            % Model evaluates integrand at cubature points, and returns
            % function handle taking psi and grad_psi at the same points as
            % argumets
            integrand = model.cellIntegrand(fun, x, cellNo, state, state0, varargin{:});
            
            I = getSampleAD(varargin{1})*0;
            for dofNo = 1:nDofMax
                keepCells = nDof(cells) >= dofNo;
                if any(keepCells)
                    ix = disc.getDofIx(state, dofNo, cells(keepCells));
                    i = W*integrand(psi{dofNo}(x), grad_psi{dofNo}(x).*scaling);
                    I(ix) = i(keepCells);
                elseif numel(cells) == disc.G.cells.num
                    warning('No cells with %d dofs', dofNo);
                end
            end
            I = disc.trimValues(I);
            
        end
        
        %-----------------------------------------------------------------%
        function I = faceFluxInt(disc, model, fun, cells, T, vT, g, mob, state, varargin)
            % Integrate integrand over all internal faces of each cell in
            % cells
            %
            % PARAMETERS:
            %   model    - Model, which contains information on how the
            %              integrand looks like
            %   fun      - Integrand function handle
            %   cells    - Cells over which we will integrate fun
            %   state    - State to be used for dofPos
            %   varargin - Variables passed to model for integrand
            %              evalauation. varargin{1} MUST be an AD object of
            %              dofs for all cells in the grid.
            %
            % RETURNS:
            %   I - Integrals int(fun*psi{dofNo}) for dofNo = 1:nDof over
            %       all cell surfaces of all cells
            
            psi     = disc.basis.psi;  % Basis functions
            nDof    = state.nDof;      % Number of dofs per cell
            nDofMax = disc.basis.nDof; % maximum number of dofs
        
            % Empty cells means all cells in grid
            if isempty(cells)
                cells = (1:disc.G.cells.num)';
            end
            
            % Get cubature for all cells, transform coordinates to ref space
%             [W, x, cellNo, faceNo] = disc.surfaceCubature.getCubature2(cells, 'surface', true);
            [W, x, cellNo, faceNo] = disc.getCubature(cells, 'surface');
            [x_c, ~, ~] = disc.transformCoords(x, cellNo);
            
            if isempty(faceNo)
                I = 0;
                return
            end
            
            % Model evaluates integrand at cubature points, and returns
            % function handle taking psi and grad_psi at the same points as
            % argumets. Inputs T, vT, g, mob are used to calculate upstram
            % directions
            integrand = model.faceIntegrand(fun, x, faceNo, cellNo, ...
                                        T, vT, g, mob, state, varargin{:});

            I = getSampleAD(varargin{1})*0;
            for dofNo = 1:nDofMax                
                keepCells = nDof(cells) >= dofNo;
                if any(keepCells)
                    ix = disc.getDofIx(state, dofNo, cells(keepCells)');
                    i  = W*integrand(psi{dofNo}(x_c));
                    I(ix) = i(keepCells);
                elseif numel(cells) == disc.G.cells.num
                    warning('No cells with %d dofs', dofNo);
                end
            end
            I = disc.trimValues(I);
            
        end
        
        %-----------------------------------------------------------------%
        function I = faceFluxIntBC(disc, model, fun, bc, state, varargin)
            % Integrate integrand over all faces where bcs are defined
            %
            % PARAMETERS:
            %   model    - Model, which contains information on how the
            %              integrand looks like
            %   fun      - Integrand function handle
            %   bc       - Boundary condition struct from schedule
            %   state    - State to be used for dofPos
            %   varargin - Variables passed to model for integrand
            %              evalauation. varargin{1} MUST be an AD object of
            %              dofs for all cells in the grid.
            %
            % RETURNS:
            %   I - All integrals int(fun*psi{dofNo}) for dofNo = 1:nDof
            %       over all bc faces for each cell
            
            G       = disc.G;          % Grid
            psi     = disc.basis.psi;  % Basis functions
            nDof    = state.nDof;      % Number of dofs per cell
            nDofMax = disc.basis.nDof; % Maximum number of dofs
            
            % Get faces and corresponding cells where BCs are defined
            faces = bc.face;
            cells = sum(G.faces.neighbors(faces,:),2);
            
            % Get cubature for each face, find corresponding cells, and
            % transform to reference coords
            [W, x, ~, faceNo] = disc.getCubature(faces, 'face');
            cellNo = sum(G.faces.neighbors(faceNo,:),2);
            [xR, ~, ~] = disc.transformCoords(x, cellNo);
            
            % Ensure that we have the right sign for the integrals
            sgn = 1 - 2*(G.faces.neighbors(faces, 1) == 0);
            W = W.*sgn;
            
            % Mappings from global face/cell numbers to bc face/cell
            % numbers
            globFace2BCface = nan(G.faces.num,1);
            globFace2BCface(faces) = 1:numel(faces);
            globCell2BCcell = nan(G.cells.num,1);
            globCell2BCcell(cells) = 1:numel(cells);
            S = sparse(globCell2BCcell(cells), 1:numel(faces), 1);
            
            % Determine if BC flux is in or out of domain from total flux
            vT = sum(state.flux,2);
            isInj = vT(faces) > 0 & sgn < 0;
            
            % Model evaluates integrand at cubature points, and returns
            % function handle taking psi at the same points as argumets
            integrand = model.faceIntegrandBC(fun, xR, faceNo, cellNo, ...
                           bc, isInj, globFace2BCface, state, varargin{:});

            I = getSampleAD(varargin{1})*0;
            for dofNo = 1:nDofMax
                keepCells = nDof(cells) >= dofNo;
                if any(keepCells)
                    ix = disc.getDofIx(state, dofNo, cells(keepCells)');
                    i  = W*integrand(psi{dofNo}(xR));
                    i = S*i;
                    I(ix) = i(keepCells);
                end
            end
            I = disc.trimValues(I);
            
        end
        
        function [flag_v, flag_G, upCells_v, upCells_G, s_v, s_G] = getSaturationUpwind(disc, faces, x, T, vT, g, mob, sdof, state)
            % Explicit calculation of upstream cells. See getSaturationUpwindDG
            [flag_v, flag_G, upCells_v, upCells_G, s_v, s_G] ...
                = getSaturationUpwindDG(disc, faces, x, T, vT, g, mob, sdof, state);
        end
        
        %-----------------------------------------------------------------%
        function [W, x, cellNo, faceNo] = getCubature(disc, elements, type)
            % Get cubature for elements. Wrapper for cubature class
            % function getCubature, with mapping of elements before and
            % after in case we are solving on a subgrid
            
            useMap = isfield(disc.G, 'mappings');
            if useMap
                % Map elements to old numbering
                maps = disc.G.mappings; 
                switch type 
                    case {'volume', 'surface'}
                        elements = maps.cellMap.new2old(elements);
                    case 'face'
                        elements = maps.faceMap.new2old(elements);
                end
            end
            
            % Get correct cubature type
            switch type 
                case 'volume'
                    cubature = disc.volumeCubature; 
                case {'surface', 'face'}
                    cubature = disc.surfaceCubature;
            end
            
            % Get cubature from cubature class
            [W, x, cellNo, faceNo] = cubature.getCubature(elements, type, ...
                              'excludeBoundary', true                   , ...
                              'internalConn'   , disc.internalConnParent, ...
                              'outwardNormal'  , true                   );
            
            if useMap
                % Map elements back to new numbering
                cellNo = maps.cellMap.old2new(cellNo);
                faceNo = maps.faceMap.old2new(faceNo);
            end
            
        end
        
        %-----------------------------------------------------------------%
        function [sMin, sMax] = getMinMaxSaturation(disc, state)
            % Get maximum and minimum saturaiton for each cell
            
            if isfield(disc.G, 'mappings')
                % We are dealing with a subgrid, make sure we get the
                % correct faces for each cell
                maps  = disc.G.mappings.cellMap;
                cells = find(maps.keep);
                G     = disc.G.parent;
                faces = G.cells.faces(mcolon(G.cells.facePos(cells), ...
                                             G.cells.facePos(cells+1)-1),1);
                s = @(x, c) disc.evaluateSaturation(x, maps.old2new(c), state.sdof(:,1), state);                
            else
                G     = disc.G;
                cells = (1:G.cells.num)';
                faces = G.cells.faces(:,1);
                s     = @(x, c) disc.evaluateSaturation(x, c, state.sdof(:,1), state);
            end
            
            [~, xSurf, cSurf, ~] = disc.getCubature((1:G.cells.num)', 'surface');
            [~, xCell, cCell, ~] = disc.getCubature((1:G.cells.num)', 'volume');
            x = [xSurf; xCell];
            c = [cSurf; cCell];
            % Get nodes for all faces of all cells
%             nodes = G.faces.nodes(mcolon(G.faces.nodePos(faces), ...
%                                          G.faces.nodePos(faces+1)-1));                                     
%             % Number of nodes for each cell
%             nfn = diff(G.faces.nodePos);
%             ncf = diff(G.cells.facePos);
%             ncn = accumarray(rldecode((1:numel(cells))', ncf(cells), 1), nfn(faces));
%             % Transform cell node coords to reference coords
%             x      = G.nodes.coords(nodes,:);
%             cellNo = rldecode(cells, ncn, 1);
            x      = disc.transformCoords(x, c, false, true);
            % Evaluate saturation
            s = s(x, c);
            % Find maxium and minimum values for each cell. Minimum values
            % returned are actually min(s,0)
%             jj = rldecode((1:numel(cells))', ncn, 1);
            s = sparse(c, (1:numel(s))', s);
            sMax = full(max(s, [], 2));
            sMin = full(min(s, [], 2));
            
        end
        
        %-----------------------------------------------------------------%
        function [jumpVal, faces, cells] = getInterfaceJumps(disc, sdof, state)
            % Get interface jumps for all internal connections in grid
            
            G     = disc.G;
            faces = find(disc.internalConn);
            cells = G.faces.neighbors(disc.internalConn,:);
            if isfield(G, 'mappings')
                % We assume we are using reordering, and thus only check
                % interface jumps for interfaces against already solved
                % cells
                order   = G.mappings.cellMap.localOrder;
                isUpstr = any(order <= order(~G.cells.ghost)',2);
                keep    = all(isUpstr(cells),2);
                faces   = faces(keep);
            end
            
            % Saturation function
            s = @(x, c) disc.evaluateSaturation(x, c, sdof, state);                
            
            % Get reference coordinates
            xF = G.faces.centroids(faces,:);            
            cells = G.faces.neighbors(faces,:);                
            cL  = cells(:,1);
            xFL = disc.transformCoords(xF, cL);
            cR  = cells(:,2);
            xFR = disc.transformCoords(xF, cR);
            
            % Find inteface jumps
            jumpVal = abs(s(xFL, cL) - s(xFR, cR));
            
        end
        
        %-----------------------------------------------------------------%
        function state = limiter(disc, state)

            G = disc.G;
            
            [jump, over, under] = deal(false(G.cells.num,1));

            if disc.meanTolerance < Inf
                % If the first dof is outside [0,1], something is wrong,
                % and we reduce to order 0
                s = state.s;
                meanOutside = any(s < 0 - disc.meanTolerance | ...
                                  s > 1 + disc.meanTolerance, 2);
%                 [smin, smax] = disc.getMinMaxSaturation(state);
%                 meanOutside = smin < 0 - disc.meanTolerance | ...
%                               smax > 1 + disc.meanTolerance;
                if any(meanOutside)
                    state = dgLimiter(disc, state, meanOutside, 'kill', 'plot', disc.plotLimiterProgress);
                end 
                              
            end
            
            if disc.outTolerance < Inf && disc.degree > 0
                if 0
                    state = dgLimiter(disc, state, true(G.cells.num,1), 'scale', 'plot', disc.plotLimiterProgress);
                else
                    state = dgLimiter(disc, state, true(G.cells.num,1), 'orderReduce', 'plot', disc.plotLimiterProgress);
                end
            end
            
            
            
            if disc.jumpTolerance < Inf
                % Cells with interface jumps larger than threshold
                [jumpVal, ~, cells] = disc.getInterfaceJumps(state.sdof(:,1), state);
                j = accumarray(cells(:), repmat(jumpVal,2,1) > disc.jumpTolerance) > 0;
                jump(cells(:))          = j(cells(:));
                jump(state.degree == 0) = false;
                if any(jump)
                    state = dgLimiter(disc, state, jump, disc.limiterType);
                end
            end
            
%             bad = jump | over | under;
%             outside = over | under;
% 
%             state.jump    = jump;
%             state.outside = over | under;
%             
%             if isfield(G, 'mappings')
%                 bad(G.cells.ghost) = false;
%             end

            
%             if any(outside)
%                 state = dgLimiter(disc, state, outside, 'kill');
%             end
%             if any(jump)
%                 state = dgLimiter(disc, state, jump, disc.limiterType);
%             end
            
            
            
            
%             state0 = state;
%             state.degree = repmat(disc.degree, G.cells.num, 1);
%             state = disc.mapDofs(state, state0);
%             state = disc.updateDisc(state);
            
%             sdof = state.sdof;
           
%             if any(bad)
%                 
%                 switch disc.limiterType
%                     
%                     case 'kill'
%                         % Simple "limiter" that reduces to zero-degree for
%                         % all bad cells
% 
% %                         state.degree(bad) = 0;
%                         ix = disc.getDofIx(stat%     if disc.degree > 0
%         ix = disc.getDofIx(state, 2:disc.basis.nDof);
%         sWdof(ix) = max(sWdof(ix), 1/(disc.degree));
%         sWdof(ix) = min(sWdof(ix), -1/(disc.degree));
%     ende, 1, bad);
% %                         sdof(ix,:) = min(max(sdof(ix,:), 0), 1);
%                         sdof(ix,:) = min(max(state.s(bad,:), 0), 1);
%                         sdof(ix,:) = sdof(ix,:)./sum(sdof(ix,:),2);
%                         state.s(bad,:) = sdof(ix,:);
%                         ix = disc.getDofIx(state, 2:nDofMax, bad);
%                         sdof(ix,:) = [];
%                         state.sdof = sdof;
%                         state.degree(bad) = 0;
%                         
%                     case 'adjust'
%                         % Reduce to dG(1) and adjust slope
%                         
% %                         meanOutside = state.s(:,1) > 1 | state.s(:,1) < 0;
% %                 
% %                         ix = disc.getDofIx(state, 1, meanOutside);
% %                         sdof(ix,:) = min(max(sdof(ix,:), 0), 1);
% %                         sdof(ix,:) = sdof(ix,:)./sum(sdof(ix,:),2);
% %                         state.degree(meanOutside) = 0;
% %                         bad(meanOutside) = false;
% 
%                         for dNo = 1:G.griddim
%                     
%                             ix0 = disc.getDofIx(state, 1, under);
%                             ix  = disc.getDofIx(state, dNo+1, under);
%                             sdof(ix) = sign(sdof(ix)).*sdof(ix0);
% %                             sdof(ix) = sign(sdof(ix)).*min(abs(sdof(ix0)), abs(sdof(ix)));
% 
%                             ix0 = disc.getDofIx(state, 1, over);
%                             ix  = disc.getDofIx(state, dNo+1, over);
%                             sdof(ix) = sign(sdof(ix)).*(1 - sdof(ix0));
% 
%                         end
%                         state.degree(bad) = 1;
%                         
%                         ix = disc.getDofIx(state, (1+G.griddim+1):nDofMax, bad);
%                         sdof(ix,:) = [];
% %                         ix = disc.getDofIx(state, G.gr
%                         state.sdof = sdof;
%                         
%                         
% %                         both = over & under;
% %                         
% %                         
% %                         
% %                         %                 disc.dofPos = disc.updateDofPos();
% %                 
% %                         sdof = sdof(:,1);
% %                         cells_o = find(over);
% %                         cells_u = find(under);
% 
%                     case 'tvb'
%                 
%                         error('Limiter type not implemented yet...');
% 
%                         
%                 end
                
%                 state = disc.getCellSaturation(state);
%                 ix = disc.getDofIx(state, 2:nDofMax, meanOutside | bad);
%                 sdof(ix,:) = [];
%                 state.sdof = s
                 
%                 
%                 % Reduce to first-order
%                 ix = disc.getDofIx(state, (1 + G.griddim + 1):nDofMax, cells);
%                 sdof(ix) = 0;
%                 
%                 % Reduce linear dofs so that saturaion is within [0,1]
% %                 ix0 = disc.getDofIx(1                , cells);
%                 for dNo = 1:G.griddim
%                     
%                     ix0 = disc.getDofIx(state, 1, cells_u);
%                     ix  = disc.getDofIx(state, dNo+1, cells_u);
%                     sdof(ix) = sign(sdof(ix)).*min(abs(sdof(ix0)), abs(sdof(ix)));
%                     
%                     ix0 = disc.getDofIx(state, 1, cells_o);
%                     ix  = disc.getDofIx(state, dNo+1, cells_o);
%                     sdof(ix) = sign(sdof(ix)).*(1 - sdof(ix0));
%                     
%                 end
%                 
%                 state.sdof(:,1) = sdof;
%                 state.sdof(:,2) = -sdof;
%                 
%                 ix0 = disc.getDofIx(state, 1, (1:G.cells.num)');
%                 state.sdof(ix0,2) = 1 - state.sdof(ix0,1);
%                 
%                 [smin, smax] = disc.getMinMaxSaturation(state);
%                 
%                 s = disc.getCellSaturation(state);
                
%             end
            
        end
        
        %-----------------------------------------------------------------%
        function v = trimValues(disc, v)
            
            tol = eps(mean(disc.G.cells.volumes));
            tol = -inf;
%             tol = 1e-7;
            ix = abs(v) < tol;
            if isa(v, 'ADI')
                v.val(ix) = 0;
            else
                v(ix) = 0;
            end
            
%             if any(ix)
%                 disp(nnz(ix));
%             end
            
        end
        
        %-----------------------------------------------------------------%
        function plotCellSaturation(disc, state, cellNo)
            
            G = disc.G;
            
            faces = G.cells.faces(G.cells.facePos(cellNo):G.cells.facePos(cellNo+1)-1);
            nodes = G.faces.nodes(mcolon(G.faces.nodePos(faces), G.faces.nodePos(faces+1)-1));
            nodes = reshape(nodes, 2, [])';
            
            swap = G.faces.neighbors(faces,1) ~= cellNo;

            nodes(swap,:) = nodes(swap, [2,1]); nodes = nodes(:,1);
            
            x = G.nodes.coords(nodes,:);
            
            x = disc.transformCoords(x, cellNo);
            
            
            if disc.dim == 1
                
                n = 100;
                xx = linspace(-1,1,n)';
                xk = xx;
                
            elseif disc.dim == 2
                
                n = 10; 
                xx = linspace(-1, 1, n);
                [xx, yy] = ndgrid(xx);
                xx = [xx(:), yy(:)];
                
                [in, on] = inpolygon(xx(:,1), xx(:,2), x(:,1), x(:,2));
                keep = in;
                xk = xx(keep,:);
                
            elseif disc.dim == 3
                
            end
                
            cellNo = repmat(cellNo, size(xk,1), 1);
            s = disc.evaluateSaturation(xk, cellNo, state.sdof, state);
            
            if disc.dim > 1
                s = scatteredInterpolant(xk, s);
                s = reshape(s(xx(:,1), xx(:,2)), n,n)';
                surf(s);
            else
                plot(xx, s);
            end
            
        end
        
    end
        
end