function [qf, qf_vol] = calculateQF_vec(G)
%
%
% SYNOPSIS:
%   function [qf, qf_vol] = calculateQF_vec(G)
%
% DESCRIPTION:  Calculate elementary integrals that are used to assemble the
% stiffness matrix for the 2D case. 
%
% PARAMETERS:
%   G - Grid structure
%
% RETURNS:

%
%   qf     -    Elementary assembly integrals : One (2D) vector value in each
%               cell, which corresponds to the two components of the integral of
%               the basis function in each coordinate over the faces (see
%               (74) in [Gain et al], then faces there correspond to edges here).
%   qf_vol -    Elementary assembly integrals : one scalar value for each
%               node, wich corresponds to the weights that are used to
%               compute th L^2 projection, see VEM_linElast.m
%
%
% EXAMPLE:
%
% SEE ALSO:
%


    assert(G.griddim == 2); 
    
    qf = zeros(size(G.cells.nodes, 1), 2); 
    cellno = rldecode([1 : G.cells.num]', diff(G.cells.nodePos)); % #ok

    cells   = 1 : G.cells.num;
    lcells = rldecode(cells', diff(G.cells.nodePos)');
    
    % For each cell, indices of the first to the second-to-last node 
    inodes1 = mcolon(G.cells.nodePos(cells), G.cells.nodePos(cells + 1) - 2)'; 
    
    % For each cell, indices of the second to the last node 
    inodes2 = mcolon(G.cells.nodePos(cells) + 1, G.cells.nodePos(cells + 1) - 1)'; 

    % For each cell, indices of each face 
    ifaces  = mcolon(G.cells.facePos(cells), G.cells.facePos(cells + 1) - 1)'; 
    
    % For each cell, indices of each face 
    faces   = G.cells.faces(ifaces, 1); 
    
    % Orienting normals ('N') so that they always point out of the current cell and
    % into the neighbor cell
    sign    = 2 * (G.faces.neighbors(faces, 1) == cellno) - 1; 
    N       = bsxfun(@times, G.faces.normals(faces', :), sign); 

    % For each cell node, add up the (scaled) normals of the two adjacent faces and
    % divide by two.
    relvec = G.faces.centroids(faces, :) - G.cells.centroids(lcells, :);
    tetvols = sum(N.*relvec, 2);
    qf_vol = zeros(numel(G.cells.nodes), 1);
    qf_vol(inodes1) = qf_vol(inodes1, :) + tetvols(inodes1); 
    qf_vol(inodes2) = qf_vol(inodes2) +  tetvols(inodes1);
    qf_vol(G.cells.nodePos(cells)) = qf_vol(G.cells.nodePos(cells)) + ...
        tetvols(G.cells.nodePos(cells + 1) - 1);
    qf_vol(G.cells.nodePos(cells + 1) - 1) = qf_vol(G.cells.nodePos(cells + 1) - 1) + ...
        tetvols(G.cells.nodePos(cells + 1) - 1);                                    
    qf_vol = qf_vol/4;
    
    qf(inodes1, :) = qf(inodes1, :) + N(inodes1, :); 
    qf(inodes2, :) = qf(inodes2, :) + N(inodes1, :);
    qf(G.cells.nodePos(cells), :) = qf(G.cells.nodePos(cells), :) + ...
        N(G.cells.nodePos(cells + 1) - 1, :); 
    qf(G.cells.nodePos(cells + 1) - 1, :) = qf(G.cells.nodePos(cells + 1) - 1, :) + ...
        N(G.cells.nodePos(cells + 1) - 1, :);
    qf = qf / 2;

end

