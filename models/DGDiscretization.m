classdef DGDiscretization < WENODiscretization
    
    properties
        degree
        basis
        dofPos
%         limiter
%         cellIntegrator
%         faceIntegrator
    end
    
    methods
        %-----------------------------------------------------------------%
        function disc = DGDiscretization(model, dim, varargin)
            
            disc = disc@WENODiscretization(model, dim, 'interpolateReference', false);
            
            disc.degree  = 1;
            disc.basis   = 'legendre';
%             disc.limiter = 'tvb';
            disc         = merge_options(disc, varargin{:});
            
%             disc.basis = DGBasisFunctions(disc.G, disc.degree);
            
            disc.basis   = dgBasis(disc.G.griddim, disc.degree, disc.basis);
            
            disc.dofPos = reshape((1:disc.G.cells.num*disc.basis.nDof)', disc.basis.nDof, []);
            
%             disc.limiter = dgLimiter(disc     , disc.limiter);
            
%             [x, w, nq, ii, jj, cellNo] = makeCellIntegrator(disc.G, (1:disc.G.cells.num)', disc.degree*2);
%             disc.cellIntegrator = struct('points'  , x     , ...
%                                          'wheights', w     , ...
%                                          'numPts'  , nq    , ...
%                                          'ii'      , ii    , ...
%                                          'jj'      , jj    , ...
%                                          'cellNo'  , cellNo);
%                                      
%             [x, w, nq, ii, jj, cellNo, faceNo] = makeFaceIntegrator(disc.G, (1:disc.G.cells.num)', disc.degree);
%             disc.faceIntegrator = struct('points'  , x     , ...
%                                          'wheights', w     , ...
%                                          'numPts'  , nq    , ...
%                                          'ii'      , ii    , ...
%                                          'jj'      , jj    , ...
%                                          'cellNo'  , cellNo, ...
%                                          'faceNo'  , faceNo);
            
        end
        
        %-----------------------------------------------------------------%
        function [xhat, translation, scaling] = transformCoords(disc, x, cells)
            
            G = disc.G;
            translation = -G.cells.centroids(cells,:);
            if isfield(G.cells, 'dx')
                scaling = 1./(G.cells.dx(cells,:)/2);
            else
                scaling = 1./(G.cells.diameters(cells)/(2*sqrt(G.griddim)));
            end
            
            xhat = (x + translation).*scaling;
               
        end
        
        %-----------------------------------------------------------------%
        function v = trimValues(disc, v)
            
            tol = eps(mean(disc.G.cells.volumes));
            ix = abs(v) < tol;
            if isa(v, 'ADI')
                v.val(ix) = 0;
            else
                v(ix) = 0;
            end
            
        end
        
        %-----------------------------------------------------------------%
        function ix = getDofIx(disc, dofNo, cells)
            
%             if size(cells,2) == 1
%                 cells = cells';
%             end
%             if size(dofNo,1) == 1
%                 dofNo = dofNo';
%             end
%             
%             nDof = disc.basis.nDof;
%             ix = reshape((cells-1)*nDof + dofNo, [], 1);
              ix = disc.dofPos(dofNo, cells);
              ix = ix(:);
        end
            
        
        %-----------------------------------------------------------------%
        function s = evaluateSaturation(disc, x, cells, dof)
            
            psi  = disc.basis.psi;
            nDof = disc.basis.nDof;
            
            s = 0;
            for dofNo = 1:nDof
                ix = disc.getDofIx(dofNo, cells);
                s = s + dof(ix).*psi{dofNo}(x);
            end
            
        end
        
        %-----------------------------------------------------------------%
        function state = getCellSaturation(disc, state)
            
            [x, w, nq, ii, jj, cellNo] = makeCellIntegrator(disc.G, (1:disc.G.cells.num)', disc.degree);
            W = sparse(ii, jj, w);

            x = disc.transformCoords(x, cellNo);
            
            sdof = state.sdof;
            nPh = size(sdof,2);
            s = zeros(disc.G.cells.num, nPh);
            for phNo = 1:nPh
%                 s(:, phNo) = disc.evaluateSaturation(repmat([0,0], disc.G.cells.num, 1), (1:disc.G.cells.num)', sdof(:,phNo));
                s(:,phNo) = (W*disc.evaluateSaturation(x, cellNo, sdof(:,phNo)))./disc.G.cells.volumes;
            end
            
            state.s = s;
            
        end
        
        %-----------------------------------------------------------------%
        function state = limiter(disc, state)
            
            G = disc.G;
            nDof = disc.basis.nDof;
            
            faces = G.cells.faces(:,1);
            nodes = G.faces.nodes(mcolon(G.faces.nodePos(faces), G.faces.nodePos(faces+1)-1));
            
            nfn = diff(G.faces.nodePos);
            ncn = accumarray(rldecode((1:G.cells.num)', diff(G.cells.facePos), 1), nfn(faces));
            
            x = G.nodes.coords(nodes,:);
            
            cells = rldecode((1:G.cells.num)', ncn, 1);
            x = disc.transformCoords(x, cells);
            
            s = disc.evaluateSaturation(x, cells, state.sdof);
            
            jj = rldecode((1:G.cells.num)', ncn, 1);
            s = sparse(jj, (1:numel(s))', s);
            
            smax = full(max(s, [], 2));
            smin = full(min(s, [], 2));
            
            over  = smax > 1;
            under = smin < 0;
            bad = over | under;
            
            sdof = state.sdof(:,1);
            sdof0 = sdof;
            
            if any(bad)
                
                cells = find(bad);
                cells_o = find(over);
                cells_u = find(under);
                
                % Reduce to first-order
                ix = disc.getDofIx((1 + G.griddim + 1):nDof, cells);
                sdof(ix) = 0;
                
                % Reduce linear dofs so that saturaion is within [0,1]
%                 ix0 = disc.getDofIx(1                , cells);
                for dNo = 1:G.griddim
                    
                    ix0 = disc.getDofIx(1, cells_u);
                    ix  = disc.getDofIx(dNo+1, cells_u);
                    sdof(ix) = sign(sdof(ix)).*min(abs(sdof(ix0)), abs(sdof(ix)));
                    
                    ix0 = disc.getDofIx(1, cells_o);
                    ix  = disc.getDofIx(dNo+1, cells_o);
                    sdof(ix) = sign(sdof(ix)).*(1 - sdof(ix0));
                    
                end
                
                state.sdof(:,1) = sdof;
                state.sdof(:,2) = -sdof;
                
                ix0 = disc.getDofIx(1, (1:G.cells.num)');
                state.sdof(ix0,2) = 1 - state.sdof(ix0,1);
                
            end
            
        end
        
        %-----------------------------------------------------------------%
        function I = cellInt(disc, integrand, cells)
        
            G    = disc.G;
            psi  = disc.basis.psi;
            nDof = disc.basis.nDof;
            
            [x, w, nq, ii, jj, cellNo] = makeCellIntegrator(G, cells, disc.degree+1);
            W = sparse(ii, jj, w);
            
            [x, ~, ~] = disc.transformCoords(x, cellNo);
            
            I = integrand(repmat([0,0], numel(cells).*nDof, 1), ones(numel(cells)*nDof, 1), 1);
            for dofNo = 1:nDof
                ix = disc.getDofIx(dofNo, (1:numel(cells))');
                p = psi{dofNo}(x);
                I(ix) = W*integrand(x, cellNo, p);
            end
            
            I = disc.trimValues(I);
            
        end
        
        %-----------------------------------------------------------------%
        function I = cellIntDiv(disc, integrand, cells)
            
            G        = disc.G;
            grad_psi = disc.basis.grad_psi;
            nDof     = disc.basis.nDof;
            
            [x, w, nq, ii, jj, cellNo] = makeCellIntegrator(G, cells, disc.degree+1);
            W = sparse(ii, jj, w);
            
            [x, ~, scaling] = disc.transformCoords(x, cellNo);
            
            I = integrand(repmat([0,0], numel(cells)*nDof, 1), ones(numel(cells)*nDof, 1), [1,1]);
            for dofNo = 1:nDof
                ix = disc.getDofIx(dofNo, (1:numel(cells))');
                gp = grad_psi{dofNo}(x).*scaling;
                I(ix) = W*(integrand(x, cellNo, gp));
            end
            
            I = disc.trimValues(I);
            
        end
        
        %-----------------------------------------------------------------%
        function I = faceIntDiv(disc, integrand, cells, upc)
            
            G        = disc.G;
            psi      = disc.basis.psi;
            nDof     = disc.basis.nDof;

            [x, w, nq, ii, jj, cellNo, faceNo] = makeFaceIntegrator(G, cells, disc.degree+1);
            W = sparse(ii, jj, w);

            upCells_v = G.faces.neighbors(:,2);
            intf      = find(disc.internalConn);
            upCells_v(intf(upc)) = disc.N(upc,1);
            upCells_v = upCells_v(faceNo);    
            upCells_G = upCells_v;
            
            [x_c, ~, ~] = disc.transformCoords(x, cellNo);
            [x_v, ~, ~] = disc.transformCoords(x, upCells_v);
            [x_G, ~, ~] = disc.transformCoords(x, upCells_G);
            
            x0 = repmat([0,0], G.cells.num.*nDof, 1);
            [c0, f0] = deal(ones(G.cells.num*nDof, 1));
            I = integrand(x0, x0, x0, c0, c0, c0, f0, 1);
            for dofNo = 1:nDof
                ix = disc.getDofIx(dofNo, (1:numel(cells))');
                p = psi{dofNo}(x_c);
                I(ix) = W*(integrand(x_c, x_v, x_G, cellNo, upCells_v, upCells_G, faceNo, p));
            end
            
            I = disc.trimValues(I);
            
        end        
        
    end
        
end