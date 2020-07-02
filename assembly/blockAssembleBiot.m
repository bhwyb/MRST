function assembly = blockAssembleBiot(G, props, drivingforces, eta, globtbls, globmappings, varargin)
    
    opt = struct('verbose'         , mrstVerbose, ...
                 'assemblyMatrices', false      , ...
                 'addAdOperators'  , false      , ...
                 'blocksize'       , []         , ...
                 'bcetazero'       , true       , ...
                 'useVirtual'      , true       , ...
                 'extraoutput'     , false);
        
    opt = merge_options(opt, varargin{:});
    
    % We solve the system
    %
    %  A*u = f
    %
    % where
    %
    %
    %        | A11    A12    0      A14    A15    0    |
    %        | A21    A22    0      0      0      0    |
    %  A =   | 0      0      A33    A34    0      A36  |
    %        | A41    A42    A43    A44    0      0    |
    %        | A51    0      0      0      0      0    |
    %        | 0      0      A63    0      0      0    | 
    %
    %
    %       | displacement_nfc (node face dofs, belongs to nodefacecoltbl)         |
    %       | displacement_c   (cell dofs, belongs to cellcoltbl)                  |
    %  u =  | pressure_nf      (node face dofs, belongs to nodefacetbl)            |
    %       | pressure_c       (cell dofs, belongs to celltbl)                     |
    %       | lambda1          (lagrangian multiplier for Dirichlet mechanical bc) |
    %       | lambda2          (lagrangian multiplier for Dirichlet fluid bc) |
    %
    %
    %       | exterior forces      |
    %       | volumetric forces    |
    %  f =  | exterior fluxes      |
    %       | source               |
    %       | mechanical bc values |              
    %       | fluid bc values      |

    
    useVirtual = opt.useVirtual;
    blocksize = opt.blocksize;
    
    nn = G.nodes.num;
    nblocks = floor(nn/blocksize);
    blocksizes = repmat(blocksize, nblocks, 1);
    if nn > nblocks*blocksize
        blocksizes = [blocksizes; nn - nblocks*blocksize];
    end
    nblocks = numel(blocksizes);
    blockinds = cumsum([1; blocksizes]);

    coltbl         = globtbls.coltbl;
    colrowtbl      = globtbls.colrowtbl;
    col2row2tbl    = globtbls.col2row2tbl;
    
    globcoltbl                = globtbls.coltbl;
    globcolrowtbl             = globtbls.colrowtbl;
    globcol2row2tbl           = globtbls.col2row2tbl;
    globnodetbl               = globtbls.nodetbl;
    globcellcoltbl            = globtbls.cellcoltbl;
    globnodecoltbl            = globtbls.nodecoltbl;
    globcellnodetbl           = globtbls.cellnodetbl;
    globcellnodecoltbl        = globtbls.cellnodecoltbl;
    globnodefacecoltbl        = globtbls.nodefacecoltbl;
    globcellcol2row2tbl       = globtbls.cellcol2row2tbl;
    globcellcolrowtbl         = globtbls.cellcolrowtbl;
    globcoltbl                = globtbls.coltbl;
    globcelltbl               = globtbls.celltbl;
    globfacetbl               = globtbls.facetbl;
    globnodetbl               = globtbls.nodetbl;
    globcellnodetbl           = globtbls.cellnodetbl;
    globnodefacetbl           = globtbls.nodefacetbl;
    globcellcoltbl            = globtbls.cellcoltbl;
    globnodecoltbl            = globtbls.nodecoltbl;
    globnodefacecoltbl        = globtbls.nodefacecoltbl;
    globcellnodefacetbl       = globtbls.cellnodefacetbl;
    globcellnodecoltbl        = globtbls.cellnodecoltbl;
    globcellnodecolrowtbl     = globtbls.cellnodecolrowtbl;
    globcellnodefacecolrowtbl = globtbls.cellnodefacecolrowtbl;
    globcolrowtbl             = globtbls.colrowtbl;
    globnodecolrowtbl         = globtbls.nodecolrowtbl;
    globcol2row2tbl           = globtbls.col2row2tbl;
    globcellcol2row2tbl       = globtbls.cellcol2row2tbl;
    globcellnodecol2row2tbl   = globtbls.cellnodecol2row2tbl;
    
    dim = coltbl.num;
    
    mechprops  = props.mechprops;
    fluidprops = props.fluidprops;
    coupprops  = props.coupprops;
   
    globC     = setupStiffnessTensor(mechprops, globtbls);
    globK     = fluidprops.K;
    globalpha = coupprops.alpha;
    globrho   = coupprops.rho;
    
    loadstruct = drivingforces.mechanics;
    fluidforces = drivingforces.fluid;
    bcstruct = fluidforces.bcstruct;
    src = fluidforces.src;
    
    % setup global structure for mechanic boundary condition
    globextforce = loadstruct.extforce;
    globforce = loadstruct.force;

    globmechbc = loadstruct.bc;
    globmechbcnodefacetbl = globmechbc.bcnodefacetbl;        
    globmechbcnodefacetbl = globmechbcnodefacetbl.addLocInd('bcinds');    
    globmechbcnodefacecoltbl = crossIndexArray(globmechbcnodefacetbl, coltbl, {}, ...
                                           'optpureproduct', true);
    globlinform = globmechbc.linform;
    globlinform = reshape(globlinform', [], 1);
    globlinformvals = globmechbc.linformvals;
    
    % setup global structure for fluid boundary condition (globextflux and globsrc)
    if ~isempty(bcstruct.bcneumann)
        error('not yet implemented');
    else
        nf_num = globnodefacetbl.num;
        globextflux = zeros(nf_num, 1);
    end

    if isempty(src)
        globsrc = zeros(globcelltbl.num, 1);
    else
        globsrc = src;
    end
    
    globbcdirichlet = bcstruct.bcdirichlet;
    globfluidbcnodefacetbl = globbcdirichlet.bcnodefacetbl;
    globfluidbcvals = globbcdirichlet.bcvals;

    % setup assembly matrices where the values computed at block level will be stored
    %       | B11  B12  B13      |
    %  B =  | B21  B22  B23  B24 |
    %       | B31  B32  B33      |
    %       |      B42       B44 |
    %
    %
    %       | displacement at cell          |    in globcellcoltbl
    %  u =  | pressure at cell              |    in globcelltbl
    %       | lagrange multiplier mechanics |    in globmechbcnodefacetbl
    %       | lagrange multiplier fluid     |    in globfluidbcnodefacetbl
    

    bglobtbls = {globcellcoltbl, globcelltbl, globmechbcnodefacetbl, globfluidbcnodefacetbl};
    
    nB = 4;
    B   = cell(nB, 1);
    insB = cell(nB, 1);
    rhs  = cell(nB, 1);
    bglobnums = cell(nB, 1);
    
    for i = 1 : nB
        B{i} = cell(nB, 1);
        insB{i} = cell(nB, 1);
        ni = bglobtbls{i}.num;
        bglobnums{i} = ni;
        rhs{i} = zeros(ni, 1);
        for j = 1 : nB;
            nj = bglobtbls{j}.num;
            B{i}{j} = sparse(ni, nj);
            insB{i}{j} = sparse(ni, nj);
        end
    end
    
    aglobtbls = {globnodefacecoltbl, globcellcoltbl, globnodefacetbl, globcelltbl, globmechbcnodefacetbl, globfluidbcnodefacetbl};
    nA = 6;
    A = cell(nA, 1);
    for i = 1 : nA
        A{i} = cell(nA, 1);
        insA{i} = cell(nA, 1);
        ni = aglobtbls{i}.num;
        for j = 1 : nA
            nj = aglobtbls{j}.num;
            A{i}{j} = sparse(ni, nj);
            insA{i}{j} = sparse(ni, nj);
        end
    end

    % we compute the number of nodes per cells
    map = TensorMap();
    map.fromTbl = globcellnodetbl;
    map.toTbl = globcelltbl;     
    map.mergefds = {'cells'};
    map = map.setup();
    
    nnodespercell = map.eval(ones(globcellnodetbl.num, 1));
    
    map = TensorMap();
    map.fromTbl = globnodefacetbl;
    map.toTbl = globfacetbl;     
    map.mergefds = {'faces'};
    map = map.setup();
    
    nnodesperface = map.eval(ones(globnodefacetbl.num, 1));
    
    
    for iblock = 1 : nblocks

        %% Construction of tensor g (as defined in paper eq 4.1.2)
        nodes = [blockinds(iblock) : (blockinds(iblock + 1) - 1)]';

        clear nodetbl;
        nodetbl.nodes = nodes;
        nodetbl = IndexArray(nodetbl);

        if opt.verbose
            fprintf('Assembling block %d/%d (%d nodes)\n', iblock, nblocks, nodetbl.num);
        end
        
        [tbls, mappings] = setupStandardBlockTables(G, nodetbl, globtbls, 'useVirtual', useVirtual);

        celltbl     = tbls.celltbl;
        colrowtbl   = tbls.colrowtbl;
        nodetbl     = tbls.nodetbl;
        facetbl     = tbls.facetbl;
        cellnodetbl = tbls.cellnodetbl;
        nodefacetbl = tbls.nodefacetbl;
        cellcoltbl  = tbls.cellcoltbl;
        nodecoltbl  = tbls.nodecoltbl;
        
        nodefacecoltbl        = tbls.nodefacecoltbl;
        cellnodefacetbl       = tbls.cellnodefacetbl;
        cellnodecoltbl        = tbls.cellnodecoltbl;
        cellnodecolrowtbl     = tbls.cellnodecolrowtbl;
        cellnodefacecoltbl    = tbls.cellnodefacecoltbl;
        cellnodefacecolrowtbl = tbls.cellnodefacecolrowtbl;
        nodecolrowtbl         = tbls.nodecolrowtbl;
        cellcol2row2tbl       = tbls.cellcol2row2tbl;
        cellnodecol2row2tbl   = tbls.cellnodecol2row2tbl;
        cellcolrowtbl         = tbls.cellcolrowtbl;
        
        globcell_from_cell = mappings.globcell_from_cell;
    
        % Obtain stiffness values for the block
        map = TensorMap();
        map.fromTbl = globcellcol2row2tbl;
        map.toTbl = cellcol2row2tbl;
        map.mergefds = {'cells', 'coldim1', 'coldim2', 'rowdim1', 'rowdim2'};
        
        map.pivottbl = cellcol2row2tbl;
        c_num     = celltbl.num;             % shortcut
        gc_num    = globcellcol2row2tbl.num; % shortcut
        cc2r2_num = cellcol2row2tbl.num;     % shortcut
        c2r2_num  = col2row2tbl.num;         % shortcut
        [c2r2, i] = ind2sub([c2r2_num, c_num], (1 : cc2r2_num)');
        map.dispind1 = sub2ind([c2r2_num, gc_num], c2r2, globcell_from_cell(i));
        map.dispind2 = (1 : cc2r2_num)';
        map.issetup = true;
        
        C = map.eval(globC);

        % Obtain permeability values for the block
        map = TensorMap();
        map.fromTbl = globcellcolrowtbl;
        map.toTbl = cellcolrowtbl;
        map.mergefds = {'cells', 'coldim', 'rowdim'};
        
        map.pivottbl = cellcolrowtbl;
        ccr_num = cellcolrowtbl.num; % shortcut
        cr_num = colrowtbl.num;      % shortcut
        [cr, i] = ind2sub([cr_num, c_num], (1 : ccr_num)');
        map.dispind1 = sub2ind([cr_num, gc_num], cr, globcell_from_cell(i));
        map.dispind2 = (1 : ccr_num)';
        map.issetup = true;
        
        K = map.eval(globK);


        %% Assembly mechanical part
        
        % We collect the degrees of freedom in the current block that belongs to the boundary.
        mechbcnodefacetbl = crossIndexArray(globmechbcnodefacetbl, nodefacetbl, {'nodes', 'faces'});
        
        mechbcterm_exists = true;
        if mechbcnodefacetbl.num == 0
            mechbcterm_exists = false;
        end
                
        if mechbcterm_exists
            
            mechbcinds = mechbcnodefacetbl.get('bcinds');
            mechbcnodefacecoltbl = crossIndexArray(mechbcnodefacetbl, coltbl, {}, 'optpureproduct', true);
            
            linformvals = globlinformvals(mechbcinds, :);

            map = TensorMap();
            map.fromTbl = globmechbcnodefacecoltbl;
            map.toTbl = mechbcnodefacecoltbl;
            map.mergefds = {'bcinds', 'coldim', 'nodes', 'faces'};
            map = map.setup();
            
            linform = map.eval(globlinform);
            linform = reshape(linform, dim, [])';
            
            % we need to remove 'bcinds' field because otherwise they will interfer with later assignment.
            mechbcnodefacetbl2 = replacefield(mechbcnodefacetbl, {{'bcinds', ''}});
            mechbc = struct('bcnodefacetbl', mechbcnodefacetbl2, ...
                            'linform'      , linform      , ...
                            'linformvals'  , linformvals);
        end

        % We get the part of external and volumetric force that are active in the block
        map = TensorMap();
        map.fromTbl = globnodefacecoltbl;
        map.toTbl = nodefacecoltbl;
        map.mergefds = {'nodes', 'faces', 'coldim'};
        map = map.setup();

        extforce = map.eval(globextforce);
        
        map = TensorMap();
        map.fromTbl = globcellcoltbl;
        map.toTbl = cellcoltbl;
        map.mergefds = {'cells', 'coldim'};
        map = map.setup();

        force = map.eval(globforce);
        
        % We collect the degrees of freedom in the current block that belongs to the boundary for the fluid part.
        
        fluidbcnodefacetbl = crossIndexArray(globfluidbcnodefacetbl, nodefacetbl, {'nodes', 'faces'});
        
        bcterm_exists = true;
        if fluidbcnodefacetbl.num == 0
            bcterm_exists = false;
        end
        
        if bcterm_exists
            map = TensorMap();
            map.fromTbl = globfluidbcnodefacetbl;
            map.toTbl = fluidbcnodefacetbl;
            map.mergefds = {'faces', 'nodes'};
            map = map.setup();
            
            fluidbcvals = map.eval(globfluidbcvals);

            clear bcdirichlet;
            bcdirichlet.bcnodefacetbl = fluidbcnodefacetbl;
            bcdirichlet.bcvals = fluidbcvals;
        else
            bcdirichlet = [];
        end
        
        % We get the part of external and volumetric sources that are active in the block
        map = TensorMap();
        map.fromTbl = globnodefacetbl;
        map.toTbl = nodefacetbl;
        map.mergefds = {'nodes', 'faces'};
        map = map.setup();
        
        extflux = map.eval(globextflux);
        
        map = TensorMap();
        map.fromTbl = globcelltbl;
        map.toTbl = celltbl;
        map.mergefds = {'cells'};
        map = map.setup();

        src = map.eval(globsrc);

        % Assemble mechanical part
        
        map = TensorMap();
        map.fromTbl = globfacetbl;
        map.toTbl = facetbl;
        map.mergefds = {'faces'};
        map = map.setup();
        
        nnpf = map.eval(nnodesperface);
        
        opts = struct('eta', eta, ...
                      'bcetazero', opt.bcetazero);
        [mechmat, mechbcvals] = coreMpsaAssembly(G, C, mechbc, nnpf, tbls, mappings, opts);
        
        % Assemble fluid part

        dooptimize = useVirtual;
        opts = struct('eta', eta, ...
                      'bcetazero', opt.bcetazero, ...
                      'dooptimize', dooptimize);
        [fluidmat, fluidbcvals, extra] = coreMpfaAssembly(G, K, bcdirichlet, tbls, mappings, opts);
        
        
        atbls = {nodefacecoltbl, cellcoltbl, nodefacetbl, celltbl, mechbcnodefacetbl, fluidbcnodefacetbl};
        nlocA = 6;
        locA = cell(nlocA, 1);
        for i = 1 : nlocA
            locA{i} = cell(4, 1);
            ni = atbls{i}.num;
            for j = 1 : nlocA
                nj = atbls{j}.num;
                locA{i}{j} = sparse(ni, nj);
            end
        end


        %% Assemble coupling terms (finite volume and consistent divergence operators)
        
        map = TensorMap();
        map.fromTbl = globcelltbl;
        map.toTbl = celltbl;
        map.mergefds = {'cells'};
        map = map.setup();
        
        alpha = map.eval(globalpha);
        rho = map.eval(globrho);
        nnpc = map.eval(nnodespercell);
        coupassembly = assembleCouplingTerms(G, eta, alpha, nnpc, tbls, mappings);
        
        % Recover terms from mpsa assembly
        
        invA11 = mechmat.invA11;
        locA{1}{1} = mechmat.A11;
        locA{1}{2} = mechmat.A12;
        locA{2}{1} = mechmat.A21;
        locA{2}{2} = mechmat.A22;
        locA{1}{5} = -mechmat.D;
        locA{5}{1} = -locA{1}{5}';
        
        % Recover terms from mpfa assembly
        
        invA33 = fluidmat.invA11;
        locA{3}{3} = fluidmat.A11;
        locA{3}{4} = fluidmat.A12;
        locA{4}{3} = fluidmat.A21;
        locA{4}{4} = fluidmat.A22;
        locA{3}{6} = -fluidmat.D;
        locA{6}{3} = -locA{3}{6}';

        % Recover the coupling terms
        
        locA{1}{4} = coupassembly.divfv;
        locA{1}{4} = -locA{1}{4}'; % We use the gradient which is the transpose of minus div
        locA{4}{1} = coupassembly.divconsnf;
        locA{4}{2} = coupassembly.divconsc;
    
        % We add the diagonal term for the mass conservation equation
        prod = TensorProd();
        prod.tbl1 = celltbl;
        prod.tbl2 = globcelltbl;
        prod.tbl3 = celltbl;
        prod.mergefds = {'cells'};
        prod = prod.setup();
        
        rho = prod.eval(rho, G.cells.volumes);
        
        % This matrix could be easily assembled directly (not using tensor assembly)
        celltbl = tbls.celltbl;
        prod = TensorProd();
        prod.tbl1 = celltbl;
        prod.tbl2 = celltbl;
        prod.tbl3 = celltbl;
        prod.mergefds = {'cells'};
        prod = prod.setup();
        
        A44b_T = SparseTensor();
        A44b_T = A44b_T.setFromTensorProd(rho, prod);
        locA{4}{4} = locA{4}{4} + A44b_T.getMatrix();
        
        % boundary conditions for the full system
        fullrhs = cell(6, 1);
        fullrhs{1} = extforce;
        fullrhs{2} = force;
        fullrhs{3} = extflux;
        fullrhs{4} = src;
        fullrhs{5} = mechbcvals;
        fullrhs{6} = fluidbcvals;

        %% We proceed with the local reduction
        
        %          | locB{1}{1}  locB{1}{2}  locB{1}{3}             |
        %  locB =  | locB{2}{1}  locB{2}{2}  locB{2}{3}  locB{2}{4} |
        %          | locB{3}{1}  locB{3}{2}  locB{3}{3}             |
        %          |             locB{4}{2}              locB{4}{4} |
        
        
        btbls  = {cellcoltbl, celltbl, mechbcnodefacetbl, fluidbcnodefacetbl};        
        bnums  = cell(4, 1);
        locB   = cell(4, 4);
        locrhs = cell(4, 1);
        
        for i = 1 : 4
            locB{i} = cell(4, 1);
            ni = btbls{i}.num;
            bnums{i} = ni;
            locrhs{i} = zeros(ni, 1);
            for j = 1 : 4;
                ni = btbls{j}.num;
                locB{i}{j} = sparse(ni, nj);
            end
        end
    
        % 1. row : Momentum equation
        A21invA11 = locA{2}{1}*invA11;
        locB{1}{1} = -A21invA11*locA{1}{2} +  locA{2}{2};
        locB{1}{2} = -A21invA11*locA{1}{4};
        locB{1}{3} = -A21invA11*locA{1}{5};

        % 2. row : Fluid mass conservation
        A41invA11 = locA{4}{1}*invA11;
        A43invA33 = locA{4}{3}*invA33;
        locB{2}{1} = -A41invA11*locA{1}{2} + locA{4}{2};
        locB{2}{2} = -A41invA11*locA{1}{4} - A43invA33*locA{3}{4} + locA{4}{4};
        locB{2}{3} = -A41invA11*locA{1}{5};
        locB{2}{4} = -A43invA33*locA{3}{6};

        % 3. row : Mechanic BC
        A51invA11 = locA{5}{1}*invA11;
        locB{3}{1} = -A51invA11*locA{1}{2};
        locB{3}{2} = -A51invA11*locA{1}{4};
        locB{3}{3} = -A51invA11*locA{1}{5};

        % 4. row : Fluid BC
        A63invA33 = locA{6}{3}*invA33;
        locB{4}{2} = -A63invA33*locA{3}{4};
        locB{4}{4} = -A63invA33*locA{3}{6};

        % Assembly of right hand side
        f = fullrhs; % shortcut
        locrhs{1} = f{2} - A21invA11*f{1};
        locrhs{2} = f{4} - A41invA11*f{1} - A43invA33*f{3};
        locrhs{3} = f{5} - A51invA11*f{1};
        locrhs{4} = f{6} - A63invA33*f{3};
        
        % We store in the global matrices wthe values that have been computed at the block level
        
        btbls = {cellcoltbl, celltbl, mechbcnodefacetbl, fluidbcnodefacetbl};        
        
        % setup the index mappings (l2ginds)
        fds = {{'cells', 'coldim'}, {'cells'}, {'faces', 'nodes', 'bcinds'}, {'faces', 'nodes'}};
        l2ginds = cell(4, 1);
        bnums = cell(4, 1);
        for i = 1 : 4
            bnums{i} = btbls{i}.num;
            map = TensorMap();
            map.fromTbl = bglobtbls{i};
            map.toTbl = btbls{i};
            map.mergefds = fds{i};
            
            l2ginds{i} = map.getDispatchInd();
        end
        
        for i = 1 : 4
            for j = 1 : 4
                [indi, indj, v] = find(locB{i}{j});
                indi = l2ginds{i}(indi);
                indj = l2ginds{j}(indj);
                ni = bglobnums{i};
                nj = bglobnums{j};
                insB{i}{j} = sparse(indi, indj, v, ni, nj); 
                B{i}{j} = B{i}{j} + insB{i}{j};
            end
            [indi, ~, v] = find(locrhs{i});
            indi = l2ginds{i}(indi);
            ni = bglobnums{i};
            insrhs = sparse(indi, 1, v, ni, 1);
            rhs{i} = rhs{i} + insrhs;
        end
        
        % setup the index mappings (l2ginds)
        fds = {{'nodes', 'faces', 'coldim'} , ...
               {'cells', 'coldim'}          , ...
               {'nodes', 'faces'}           , ...
               {'cells'}                    , ...
               {'faces', 'nodes', 'bcinds'} , ...
               {'faces', 'nodes'}};
        
        al2ginds = cell(nA, 1);
        bnums = cell(nA, 1);
        for i = 1 : nA
            map = TensorMap();
            map.fromTbl  = aglobtbls{i};
            map.toTbl    = atbls{i};
            map.mergefds = fds{i};
            
            al2ginds{i} = map.getDispatchInd();
        end
        
        for i = 1 : nA
            for j = 1 : nA
                [indi, indj, v] = find(locA{i}{j});
                indi = al2ginds{i}(indi);
                indj = al2ginds{j}(indj);
                ni = aglobtbls{i}.num;
                nj = aglobtbls{j}.num;
                insA{i}{j} = sparse(indi, indj, v, ni, nj); 
                A{i}{j} = A{i}{j} + insA{i}{j};
            end
        end
        
    end
    
    % We concatenate the matrices
    for i = 1 : nB
        B{i} = horzcat(B{i}{:});
    end
    B = vertcat(B{:});
    rhs = vertcat(rhs{:});

    assembly = struct('B'  , B, ...
                      'rhs', rhs);
    assembly.A = A;
    
    if opt.addAdOperators

        fluxop = fluidassembly.adoperators.fluxop;
        
        % Setup face node dislpacement operator
        fndisp{1} = -invA11*locA{1}{2};
        fndisp{2} = -invA11*locA{1}{4};
        fndisp{3} = -invA11*locA{1}{5};
        
        facenodedispop = @(u, p, lm, extforce) facenodedispopFunc(u, p, lm, extforce, fndisp);
        
        % Setup stress operator
        aver = cellAverageOperator(tbls, mappings);
        stress{1} = C1;
        stress{2} = C2;
        stressop = @(unf, uc) stressopFunc(unf, uc, stress, aver);

        % Setup divKgrad operator
        divKgrad{1} = - A43invA33*locA{3}{4} + locA{4}{4};
        divKgrad{2} = - A43invA33*locA{3}{6};
        divKgradrhs = f{4} - A43invA33*f{3};
        
        divKgradop = @(p, lf) divKgradopFunc(p, lf, divKgrad, divKgradrhs);

        % Setup consistent divergence operator (for displacement, includes value of Biot coefficient alpha)
        % The divergence is volume weighted
        divu{1} = - A41invA11*locA{1}{2} + locA{4}{2};
        divu{2} = - A41invA11*locA{1}{4};
        divu{3} = - A41invA11*locA{1}{5};
        divu{4} = A41invA11;
        
        divuop = @(u, p, lm, extforce) divuopFunc(u, p, lm, extforce, divu);

        % Setup momentum balance operator 
        moment{1} = B11;
        moment{2} = B12;
        moment{3} = B13;
        % We have : right-hand side for momentum equation = f{2} - A21invA11*f{1}. Hence, we set
        moment{4} = A21invA11;
        momentrhs = f{2};
        
        momentop = @(u, p, lm, extforce) momentopFunc(u, p, lm, extforce, moment, momentrhs);
        
        % Setup dirichlet boundary operator for mechanics
        mechdir{1} = B31;
        mechdir{2} = B32;
        mechdir{3} = B33;
        mechdirrhs = redrhs{3};
        
        mechDirichletop = @(u, p, lm) mechdiropFunc(u, p, lm, mechdir, mechdirrhs);
        
        % Setup dirichlet boundary operator for flow
        fluiddir{1} = B42;
        fluiddir{2} = B44;
        fluiddirrhs = redrhs{4};
        
        fluidDirichletop = @(p, lf) fluiddiropFunc(p, lf, fluiddir, fluiddirrhs);
        
        adoperators = struct('fluxop'          , fluxop          , ...
                             'facenodedispop'  , facenodedispop  , ...
                             'stressop'        , stressop        , ...
                             'divKgradop'      , divKgradop      , ...
                             'divuop'          , divuop          , ...
                             'momentop'        , momentop        , ...
                             'fluidDirichletop', fluidDirichletop, ...
                             'mechDirichletop' , mechDirichletop);

        assembly.adoperators = adoperators;
        
    end    
end


function fndisp = facenodedispopFunc(u, p, lm, extforce, fndisp)
    fndisp = fndisp{1}*u + fndisp{2}*p + fndisp{3}*lm + extforce;
end

function stress = stressopFunc(unf, uc, stress, aver)
    
    % get stress at each cell-node region (corner)
    stress = stress{1}*unf + stress{2}*uc;
    stress = aver*stress;
    
end

function divKgrad = divKgradopFunc(p, lf, divKgrad, divKgradrhs)
    divKgrad = divKgrad{1}*p + divKgrad{2}*lf - divKgradrhs;
end

function divu = divuopFunc(u, p, lm, extforce, divu)
    divu = divu{1}*u + divu{2}*p + divu{3}*lm + divu{4}*extforce;
end

function moment = momentopFunc(u, p, lm, extforce, moment, momentrhs)
    moment = moment{1}*u + moment{2}*p + moment{3}*lm + moment{4}*extforce - momentrhs;
end

function mechdir = mechdiropFunc(u, p, lm, mechdir, mechdirrhs)
    mechdir = mechdir{1}*u + mechdir{2}*p + mechdir{3}*lm - mechdirrhs;
end

function fluiddir = fluiddiropFunc(p, lf, fluiddir, fluiddirrhs)
    fluiddir = fluiddir{1}*p + fluiddir{2}*lf - fluiddirrhs;
end
% 