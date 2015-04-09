function Kup = upAbsPerm(block, varargin)
opt = struct(...
    'dp',         1*barsa, ...
    'dims',       1:3 ...
    );
opt = merge_options(opt, varargin{:});

% Handle input
dims  = opt.dims; % Dimensions to upscale
ndims = length(dims);
dp    = opt.dp .* ones(ndims,1);
G     = block.G;
rock  = block.rock;
isPeriodic = block.periodic;

% Initial state
state0 = initResSol(G, 100*barsa, 1);

% Allocate space for fluxes
if isPeriodic
    V = nan(ndims, ndims); % matrix
else
    V = nan(ndims,1); % vector
end

% Setup solver
fluidPure  = initSingleFluid('mu' ,1, 'rho', 1);
if isPeriodic
    bcp = block.bcp;
    T = computeTransGp(G.parent, G, rock);
    psolver = @(state0, bcp) incompTPFA(state0, G, ....
        T, fluidPure, 'bcp', bcp);
else
    T = computeTrans(G, rock);
    psolver = @(state0, bc) incompTPFA(state0, block.G, ...
        T, fluidPure, 'bc', bc);
end

% Loop over dimensions, apply pressure drop and compute fluxes
for i = 1:ndims
    
    % Set boundary conditions
    if isPeriodic
        bcp.value(:) = 0;
        bcp.value(bcp.tags == dims(i)) = dp(i);
        bc = opt.bcp;
    else
        bc = addBC([], block.faces{dims(i)}{1}, 'pressure', dp(i) );
        bc = addBC(bc, block.faces{dims(i)}{2}, 'pressure', 0 );
    end
    
    % Solve pressure equation
    warning('off','mrst:periodic_bc');
    state1 = psolver(state0, bc);
    warning('on','mrst:periodic_bc');
    
    if isPeriodic
        % Store flux in j-direction caused by pressure drop in d-direction
        for j = 1:ndims
            faces = bcp.face(bcp.tags==dims(j));
            sign  = bcp.sign(bcp.tags==dims(j));
            V(j,i) = sum(state1.flux(faces, 1).*sign) / ...
                block.areas(dims(j),2);
        end
    else
        faces = block.faces{dims(i)}{2};
        sign  = ones(numel(faces), 1);
        sign(G.faces.neighbors(faces,1)==0) = -1;
        V(i) = sum(state1.flux(faces, 1).*sign) / block.areas(dims(i),2);
    end
    
end

% Compute upscaled permeability
L = block.lengths(dims);
if isPeriodic
    Pinv = diag(L(:)./dp(:)); % matrix
    Kup   = - V*Pinv;
else
    Kup   = V.*(L(:)./dp(:)); % vector
end
Kup = Kup';


end


