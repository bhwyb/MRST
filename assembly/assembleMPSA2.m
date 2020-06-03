function assembly = assembleMPSA2(G, prop, loadstruct, eta, tbls, mappings, varargin)
%% Assembly of MPSA-weak
%%
%% Reference paper:
%% Finite volume methods for elasticity with weak symmetry
%% Keilegavlen, Eirik and Nordbotten, Jan Martin
%% International Journal for Numerical Methods in Engineering
%% 2017

    % the solution is given by the system
    %
    % A = [[A11, A12, -D];
    %      [A21, A22,  0];
    %      [D' , 0  ,  0]];
    %
    % u = [u (displacement at nodefacecoltbl);
    %      u (displacement at cellcoltbl);
    %      lagmult (forces in the linear directions at the boundary)];
    %
    % f = [extforce (force at nodefacecoltbl);
    %      force    (volumetric force at cellcoltbl);
    %      bcvals   (for the linear form at the boundary)];
    %
    % A*u = f
    %
    % Note: extforce is sparse and should only give contribution at facets
    % that are at the boundary
    %
    % By construction of the method, the matrix A11 is block-diagonal. Hence,
    % we invert it directly and reduce to a cell-centered scheme.
    
    
    opt = struct('bcetazero'       , true , ...
                 'assemblyMatrices', false, ...
                 'extraoutput'     , false);
    opt = merge_options(opt, varargin{:});
    
    % Recover IndexArrays
    coltbl                = tbls.coltbl;
    celltbl               = tbls.celltbl;
    nodetbl               = tbls.nodetbl;
    cellnodetbl           = tbls.cellnodetbl;
    nodefacetbl           = tbls.nodefacetbl;
    cellcoltbl            = tbls.cellcoltbl;
    nodecoltbl            = tbls.nodecoltbl;
    nodefacecoltbl        = tbls.nodefacecoltbl;
    cellnodefacetbl       = tbls.cellnodefacetbl;
    cellnodecoltbl        = tbls.cellnodecoltbl;
    cellnodecolrowtbl     = tbls.cellnodecolrowtbl;
    cellnodefacecoltbl    = tbls.cellnodefacecoltbl;
    cellnodefacecolrowtbl = tbls.cellnodefacecolrowtbl;
    colrowtbl             = tbls.colrowtbl;
    nodecolrowtbl         = tbls.nodecolrowtbl;
    col2row2tbl           = tbls.col2row2tbl;
    cellcol2row2tbl       = tbls.cellcol2row2tbl;
    cellnodecol2row2tbl   = tbls.cellnodecol2row2tbl;
    
    cell_from_cellnode         = mappings.cell_from_cellnode;
    node_from_cellnode         = mappings.node_from_cellnode;
    cellnode_from_cellnodeface = mappings.cellnode_from_cellnodeface;
    nodeface_from_cellnodeface = mappings.nodeface_from_cellnodeface;
    
    % Some shortcuts
    c_num     = celltbl.num;
    n_num     = nodetbl.num;
    cnf_num   = cellnodefacetbl.num;
    cnfc_num  = cellnodefacecoltbl.num;
    cn_num    = cellnodetbl.num;
    cncr_num  = cellnodecolrowtbl.num;
    nf_num    = nodefacetbl.num;
    nfc_num   = nodefacecoltbl.num;
    cnfcr_num = cellnodefacecolrowtbl.num;
    d_num     = coltbl.num;
    
    dim = coltbl.num;

    %% Construction of tensor g (as defined in paper eq 4.1.2), belongs to cellnodefacecoltbl
    g = computeConsistentGradient(G, eta, tbls, mappings, 'bcetazero', opt.bcetazero);
    
    %% Setup of the facet normals, belongs to cellnodefacecoltbl
    facetNormals = computeFacetNormals(G, cellnodefacetbl);

    %% Setup of stiffness tensor, belongs to cellcol2row2tbl
    C = setupStiffnessTensor(prop, tbls);

    %% Compute number of cell per node
    map = TensorMap();
    map.fromTbl = cellnodetbl;
    map.toTbl = nodetbl;
    map.mergefds = {'nodes'};
    map = map.setup();
    
    ncellpernode = map.eval(ones(cellnodetbl.num, 1));
    
    cellnodefacecol2rowtbl = crossIndexArray(cellnodefacecolrowtbl, coltbl, {}, 'crossextend', {{'coldim', {'coldim1', 'coldim2'}}});
    nodefacecol2rowtbl     = projIndexArray(cellnodefacecol2rowtbl, {'nodes', 'faces', 'coldim1', 'coldim2', 'rowdim'});
    cellnodeface2col2tbl   = crossIndexArray(cellnodefacecoltbl, cellnodefacecoltbl, {'cells', 'nodes'}, 'crossextend', {{'faces', ...
                        {'faces1', 'faces2'}}, {'coldim', {'coldim1', 'coldim2'}}});
    nodeface2col2tbl       = projIndexArray(cellnodeface2col2tbl, {'nodes', 'faces1', 'faces2', 'coldim1', 'coldim2'});
    cellnodefacecol2tbl    = replacefield(cellnodefacecolrowtbl, {{'coldim', 'coldim1'}, {'rowdim', 'coldim2'}});
    cellcol2tbl            = crossIndexArray(cellcoltbl, coltbl, {}, 'crossextend', {{'coldim', {'coldim1', 'coldim2'}}});
    
    prod = TensorProd();
    prod.tbl1 = cellcol2row2tbl;
    prod.tbl2 = cellnodefacecoltbl;
    prod.tbl3 = cellnodefacecol2rowtbl;
    prod.replacefds1 ={{'rowdim1', 'rowdim'}};
    prod.replacefds2 ={{'coldim', 'rowdim2'}};
    prod.mergefds = {'cells'};
    prod.reducefds = {'rowdim2'};
    prod = prod.setup();
    
    Cg = prod.eval(C, g);
    
    prod = TensorProd();
    prod.tbl1 = cellcol2row2tbl;
    prod.tbl2 = cellnodefacecoltbl;
    prod.tbl3 = nodefacecol2rowtbl;
    prod.replacefds1 ={{'rowdim1', 'rowdim'}};
    prod.replacefds2 ={{'coldim', 'rowdim2'}};
    prod.reducefds = {'cells', 'rowdim2'};
    prod = prod.setup();
    
    CAverg = prod.eval(C, g);
    
    % We divide by the number of cell per node
    prod = TensorProd();
    prod.tbl1 = nodetbl;
    prod.tbl2 = nodefacecol2rowtbl;
    prod.tbl3 = nodefacecol2rowtbl;
    prod.mergefds = {'nodes'};
    prod = prod.setup();
    
    CAverg = prod.eval(1./ncellpernode, CAverg);
    
    %% We include the fix at the boundary when the symmetry condition cannot be imposed.
    
    switch dim
      case 2
        minncellpernode = 1;
      case 3
        minncellpernode = 2;
    end
    
    bcfix = zeros(nodetbl.num, 1);
    bcfix(ncellpernode < minncellpernode) = 1;
    
    bcfix1 = 0.5*bcfix + 0.5;
    prod = TensorProd();
    prod.tbl1 = nodetbl;
    prod.tbl2 = cellnodefacecol2rowtbl;
    prod.tbl3 = cellnodefacecol2rowtbl;
    prod.mergefds = {'nodes'};
    prod = prod.setup();
    
    Cg = prod.eval(bcfix1, Cg);
    
    bcfix2 = -0.5*bcfix + 0.5;
    prod = TensorProd();
    prod.tbl1 = nodetbl;
    prod.tbl2 = nodefacecol2rowtbl;
    prod.tbl3 = nodefacecol2rowtbl;
    prod.mergefds = {'nodes'};
    prod = prod.setup();
    
    CAverg = prod.eval(bcfix2, CAverg);
    
    %% We take the transpose of CAverg
    
    map = TensorMap();
    map.fromTbl = nodefacecol2rowtbl;
    map.toTbl = nodefacecol2rowtbl;
    map.replaceFromTblfds = {{'coldim1', 'rowdim', 'interchange'}};
    map.mergefds = {'nodes', 'faces', 'coldim1', 'coldim2', 'rowdim'};
    map = map.setup();
    
    CAverg = map.eval(CAverg);
    
    %% We map CAverg to cellnodefacecol2rowtbl and add it to Cg
    % (note that a coefficient 0.5 has already be introduced when we included the bc fix)
    
    map = TensorMap();
    map.fromTbl = nodefacecol2rowtbl;
    map.toTbl = cellnodefacecol2rowtbl;
    map.mergefds = {'nodes', 'faces', 'coldim1', 'coldim2', 'rowdim'};
    map = map.setup();
    
    CAverg = map.eval(CAverg);
    
    Cg = Cg + CAverg;
    
    %% We multiply Cg with facetNormals
    
    prod = TensorProd();
    prod.tbl1 = cellnodefacecoltbl;
    prod.tbl2 = cellnodefacecol2rowtbl;
    prod.tbl3 = cellnodeface2col2tbl;
    prod.replacefds1 = {{'coldim', 'rowdim'}, {'faces', 'faces1'}};
    prod.replacefds2 = {{'faces', 'faces2'}};
    prod.mergefds = {'cells', 'nodes'};
    prod.reducefds = {'rowdim'};
    prod = prod.setup();
    
    nCg = prod.eval(facetNormals, Cg);
    
    %% We setup A11
    
    map = TensorMap();
    map.fromTbl = cellnodeface2col2tbl;
    map.toTbl = nodeface2col2tbl;
    map.mergefds = {'nodes', 'faces1', 'faces2', 'coldim1', 'coldim2'};
    map = map.setup();
    
    A11 = map.eval(nCg);
    
    prod = TensorProd();
    prod.tbl1 = nodeface2col2tbl;
    prod.tbl2 = nodefacecoltbl;
    prod.tbl3 = nodefacecoltbl;
    prod.replacefds1 = {{'faces1', 'faces'}, {'coldim1', 'coldim'}};
    prod.replacefds2 = {{'faces', 'faces2'}, {'coldim', 'coldim2'}};
    prod.mergefds = {'nodes'};
    prod.reducefds = {'faces2', 'coldim2'};
    prod = prod.setup();
    
    A11_T = SparseTensor();
    A11_T = A11_T.setFromTensorProd(A11, prod);
    A11 = A11_T.getMatrix();
    
    %% We setup A12
    
    map = TensorMap();
    map.fromTbl = cellnodeface2col2tbl;
    map.toTbl = cellnodefacecol2tbl;
    map.replaceFromTblfds = {{'faces1', 'faces'}};
    map.mergefds = {'cells', 'nodes', 'faces', 'coldim1', 'coldim2'};
    map = map.setup();
    
    % note the minus sign
    A12 = map.eval( - nCg);
    
    prod = TensorProd();
    prod.tbl1 = cellnodefacecol2tbl;
    prod.tbl2 = cellcoltbl;
    prod.tbl3 = nodefacecoltbl;
    prod.replacefds1 = {{'coldim1', 'coldim'}};
    prod.replacefds2 = {{'coldim', 'coldim2'}};
    prod.reducefds = {'cells', 'coldim2'};
    prod = prod.setup();
    
    A12_T = SparseTensor();
    A12_T = A12_T.setFromTensorProd(A12, prod);
    A12 = A12_T.getMatrix();
    
    %% We setup A21
    
    map = TensorMap();
    map.fromTbl = cellnodeface2col2tbl;
    map.toTbl = cellnodefacecol2tbl;
    map.replaceFromTblfds = {{'faces2', 'faces'}};
    map.mergefds = {'cells', 'nodes', 'faces', 'coldim1', 'coldim2'};
    map = map.setup();
    
    % note the minus sign
    A21 = map.eval( - nCg);
    
    prod = TensorProd();
    prod.tbl1 = cellnodefacecol2tbl;
    prod.tbl2 = nodefacecoltbl;
    prod.tbl3 = cellcoltbl;
    prod.replacefds1 = {{'coldim2', 'coldim'}};
    prod.replacefds2 = {{'coldim', 'coldim1'}};
    prod.reducefds = {'nodes', 'faces', 'coldim1'};
    prod = prod.setup();
    
    A21_T = SparseTensor();
    A21_T = A21_T.setFromTensorProd(A21, prod);
    A21 = A21_T.getMatrix();
    
    %% We setup A22
    
    map = TensorMap();
    map.fromTbl = cellnodeface2col2tbl;
    map.toTbl = cellcol2tbl;
    map.mergefds = {'cells', 'coldim1', 'coldim2'};
    map = map.setup();
    
    A22 = map.eval(nCg);
    
    prod = TensorProd();
    prod.tbl1 = cellcol2tbl;
    prod.tbl2 = cellcoltbl;
    prod.tbl3 = cellcoltbl;
    prod.replacefds1 = {{'coldim1', 'coldim'}};
    prod.replacefds2 = {{'coldim', 'coldim2'}};
    prod.mergefds = {'cells'};
    prod.reducefds = {'coldim2'};
    prod = prod.setup();
    
    A22_T = SparseTensor();
    A22_T = A22_T.setFromTensorProd(A22, prod);
    A22 = A22_T.getMatrix();
    
    % Uses the block structure for the local reduction
    % We count the number of degrees of freedom that are connected to the same
    % node.
    [nodes, sz] = rlencode(nodefacecoltbl.get('nodes'), 1);
    opt.invertBlocks = 'mex';
    bi = blockInverter(opt);
    invA11 = bi(A11, sz);


    % We enforce the boundary conditions as Lagrange multipliers

    bc = loadstruct.bc;
    if ~isfield(bc, 'bcnodefacetbl')
        bc = setupFaceBC(bc, G, tbls);
    end
    [D, bcvals] = setupNodeFaceBc(bc, G, tbls);
    
    extforce = loadstruct.extforce;
    force = loadstruct.force;

    fullrhs{1} = extforce;
    fullrhs{2} = force;
    fullrhs{3} = bcvals;
    

    
    matrices = struct('A11', A11, ...
                      'A12', A12, ...
                      'A21', A21, ...
                      'A22', A22, ...
                      'D'  , D  , ...
                      'invA11', invA11);
    matrices.fullrhs = fullrhs;
    
    % We reduced the system (shur complement) using invA11
    % We obtain system of the form
    %
    % B*u = rhs
    %
    % where
    %
    % B = [[B11, B12];
    %      [B21, B22]];
    %
    % u = [u (displacement at cellcoltbl);
    %      lagmult];
    %
    % rhs = [-A21*invA11*extforce;  +  [force;
    %        -D'*invA11*extforce  ]     bcvals]
    
    B11 = A22 - A21*invA11*A12;
    B12 = A21*invA11*D;
    B21 = -D'*invA11*A12;
    B22 = D'*invA11*D;

    B = [[B11, B12]; ...
         [B21, B22]];
    
    rhs{1} = -A21*invA11*extforce + force; 
    rhs{2} = -D'*invA11*extforce + bcvals;
    
    rhs = vertcat(rhs{:});

    % Assembly of operator to compute u_{nodefacecoltbl} from solution of the system
    % (which consists of the concatenation of u_{cellcol} and lagmult) and
    % extforce which is a force in nodefacecoltbl
    %
    % We have  u_{nodefacecoltbl} = R1*sol + R2*extforce
    %
    % where R1 = invA11*[-A12, D] and R2 = invA11
    
    R1 = invA11*[-A12, D];
    R2 = invA11;
    
    % The divergence operator (integrated over the volume)
    % is given by 
    %
    %  div[c] = sum (m[f,s] u_[f,n,i] n[c,f,i])
    %
    % where u:solution, n:normal, m:area
    % indices : c:cell, f:face, n:node.
    
    % The facetNormals are already weighted with respect to area
    
    prod = TensorProd();
    prod.tbl1 = cellnodefacecoltbl;
    prod.tbl2 = nodefacecoltbl;
    prod.tbl3 = celltbl;
    prod.reducefds = {'faces', 'nodes', 'coldim'};
    % prod = prod.setup();
    
    prod.pivottbl = cellnodefacecoltbl;
    prod.dispind1 = (1 : cnfc_num)';
    [c, i] = ind2sub([d_num, cnf_num], (1 : cnfc_num)');
    prod.dispind2 = sub2ind([d_num, nf_num], c, nodeface_from_cellnodeface(i));
    prod.dispind3 = cell_from_cellnode(cellnode_from_cellnodeface(i));
    prod.issetup = true;
    
    div_T = SparseTensor;
    div_T = div_T.setFromTensorProd(facetNormals, prod);
    div = div_T.getMatrix();
    
    
    assembly = struct('B'       , B       , ...
                      'rhs'     , rhs     , ...
                      'g'       , g       , ...
                      'extforce', extforce, ...
                      'R1'      , R1      , ...
                      'R2'      , R2);
    
    if opt.assemblyMatrices
        assembly.matrices = matrices;
    end
    
    if opt.extraoutput
        assembly.divop = @(sol) mpsaDivOperator(sol, extforce, R1, R2, div);
    end
    
end




