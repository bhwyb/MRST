function reginx = getRegMap(val, REGNUM, REGINX, varargin)
   opt = struct('cellInx', []);
   opt = merge_options(opt, varargin{:});
   nt  = numel(REGINX);
   
   if isempty(val)
      % Allow for empty val
      if isempty(opt.cellInx)
         N = numel(REGNUM);
      else
         N = numel(opt.cellInx);
      end
   elseif isa(val, 'ADI') 
      N = numel(val.val);
   else
      N = size(val,1);
   end
   
   
   if isempty(opt.cellInx),
      
      if nt == 1,

         reginx = { ':' };

      else
         
         % Entire domain.

         if N ~= numel(REGNUM),
            % Do not use NUMEL in case of ADI.

            error('Region reference for input undefined');
         end

         reginx = REGINX;
      
      end

   else
      % Reference to (small) subset of all cells

      cellInx = opt.cellInx;
      regnum  = REGNUM(cellInx);

      if numel(cellInx) > 1

         if N ~= numel(cellInx),
            % Do not use NUMEL in case of ADI.

            error('Number of cell indices must be same as input values');
         end

         reginx = arrayfun(@(x) find(x == regnum), 1 : nt, ...
                           'UniformOutput', false);

      elseif numel(cellInx) == 1,
         % Allow single input (for exploring single cell functions).

%             reginx = { repmat(regnum, size(val)) };
         reginx         = cell(1, nt);
         reginx(regnum) = {(1:N)'};

      else

         error('Got empty cellInx input. This is not happening...');

      end
      
   end
   
end
