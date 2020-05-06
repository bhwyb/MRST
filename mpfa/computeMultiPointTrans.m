function [T, T_noflow] = computeMultiPointTrans(G, rock, varargin)
% Compute multi-point transmissibilities. 
%
% Two versions available : 'legacy' (default) and 'tensor Assembly'.
%
% This legacy version is faster. It is limited to a mesh with grid cells where
% corners have the same number of faces as the spatial dimension (this is always
% the case in 2D but not in 3D). The tensor assembly version
% (computeMultiPointTransTensorAssembly) can handle the other cases but is
% slower (the implementation will be optimized in the future to run faster).
%
   opt = struct('useTensorAssembly', false, ...
                'blocksize', [], ...
                'neumann', false);
   
   [opt, extra] = merge_options(opt, varargin{:});
   
   if ~opt.useTensorAssembly
       [T, T_noflow] = computeMultiPointTransLegacy(G, rock, extra{:});
   else
       blocksize = opt.blocksize;
       if opt.neumann
           mpfastruct = computeNeumannMultiPointTrans(G, rock, 'blocksize', blocksize, extra{:})
           T = mpfastruct;
       else
           mpfastruct = computeMultiPointTransTensorAssembly(G, rock, 'blocksize', blocksize, extra{:});
           T = mpfastruct;
       end
   end       

end
