function [x_out, output, trunc_mats] = WB_projection(A, b, nvec, thr, tau, regpar_c, P, options, trunc_options, trunc_mats)
%
% [x_out, output, trunc_mats] = WB_projection(A, b, nvec, thr, tau, regpar_c, P, options, trunc_options, trunc_mats)
%
% WB_projection is a Golub-Kahan-based warm-basis hybrid projection method that can
% exploit compression and recycling techniques in order to solve a broad
% class of inverse problems where memory requirements or high computational
% cost may otherwise be prohibitive.
%
% Inputs:
%                A : either (a) a full or sparse matrix
%                           (b) a matrix object that performs matrix*vector
%                                 and matrix'*vector operations
%                b : rhs vector
%         regpar_c : user-specified nonnegative regularization parameter
%                    for the warm-basis coefficient. WGCV is not used
%                    for this scalar subproblem.
%                P : left preconditioner, P_left, OR
%                  : cell containing left and right preconditioner (optional)
%                          {P_left, P_right}
%                   Note: Preconditioning is not yet implemented for the
%                   recycling process
%
% options : structure with the following fields (optional)
%         InSolv - solver for the inner problem: [none | TSVD | {Tikhonov}]
%         RegPar - a value or method to find the regularization parameter:
%                       [non-negative scalar | GCV | {WGCV} | optimal]
%                   Note: 'optimal' requires x_true
%          Omega - if RegPar is 'WGCV', then omega must be
%                       [non-negative scalar | {adapt}]
%           Iter - maximum number of Lanczos iterations:
%                       [ positive integer | {min(m,n,100)} ]
%         x_true - True solution : [ array | {off} ]
%                Returns error norms with respect to x_true at each iteration
%                and is used to compute 'optimal' regularization parameters
%         BegReg - Begin regularization after this iteration:
%                   [ positive integer | {2} ]
%         Reorth - it is the option for HyBR rather than HyBR recycle
%             Vx - extra space needed for finding optimal reg. parameters
%         FlatTol - Tolerance for detecting flatness in the GCV curve as a
%                    stopping criteria
%                   [ non-negative scalar | {10^-6}]
%         MinTol - Window of iterations for detecting a minimum of the GCV curve
%                    as a stopping criteria
%                   [ positive integer | {3}]
%         ResTol - Residual tolerance for stopping the LBD iterations,
%                    similar to the stopping criteria from [1]: [atol, btol]
%                   [non-negative scalar  | {[10^-6, 10^-6]}]
%       StopRule - iteration stopping rule: 'maxit', 'gcv', 'dp',
%                  'gcv_dp', or 'all'. Default is 'gcv_dp'.
%                  options.Iter is always enforced as the maximum.
%       NoiseLevel - noise level for discrepancy principle. If <1, it is
%                    treated as relative noise level; otherwise absolute.
%       DPTau - safety factor for discrepancy principle, default 1.01.
%       Note: options is a structure created using the function 'HyBRset'
%               (see 'HyBRset' for more details)
%
% trunc_options : structure with additional parameters for truncation
%        nOuter - number of outer iterations
%        nInner - maximum storage of solution vector space
%        max_mm  - maminimum number of vectors to save at compression
%        compress  - method for compression
%
% trunc_mats : recycled matrices, W, Y, and R
%                W : basis for solution search space - dimension k = trunc_options.nOuter;
%                Y : basis for rhs search space
%                R : R-factor, AW = YR
%                    AW = YR
%                    approx solution tilde(x) = Ws_k
%                x : initial solution for the case that the bases are
%                    given. x can be [] or some appropriate initial base
%
% Outputs:
%      x_out : computed solution
%     output : structure with the following fields:
%      iterations - stopping iteration (options.Iter | GCV-determined)
%         GCVstop - GCV curve used to find stopping iteration
%            Enrm - relative error norms (requires x_true)
%            Rnrm - relative residual norms
%            Xnrm - relative solution norms
%             U,V - Lanczos basis vectors
%               B - bidiagonal matrix from LBD
%            flag - a flag that describes the output/stopping condition:
%                       1 - flat GCV curve
%                       2 - min of GCV curve (within window of MinTol its)
%                       3 - performed max number of iterations
%                       4 - achieved residual tolerance
%                       5 - discrepancy principle
%           alpha - regularization parameter at (output.iterations) its
%           Alpha - vector of all regularization parameters computed
%     truncmats : structure containing recycled matrices, W, Y, R, and x
%
% References:
%   [1] Chung, Nagy and O'Leary, "A Weighted-GCV Method for Lanczos-Hybrid
%       Regularization", ETNA 28 (2008), pp. 149-167.
%   [2]  Chung, de Sturler, and Jiang. "Hybrid Projection Methods with
%           Recycling for Inverse Problems". SISC, 2020.
%
% J. Chung, E. de Sturler, and J. Jiang, 2020
 
%% Initialization
% Added optional stopping controls through the options structure.
defaultopt = struct('InSolv','tikhonov','RegPar','wgcv','Omega',...
  'adapt', 'Iter', [], 'Reorth','off','x_true', 'off', 'BegReg', 2,...
  'Vx' , [], 'FlatTol', 10^-6, 'MinTol', 4, 'ResTol', [10^-6, 10^-6],...
  'StopRule', 'gcv_dp', 'NoiseLevel', [], 'DPTau', 1.01);
 
% If input is 'defaults,' return the default options in x_out
if nargin==1 && nargout <= 1 && isequal(A,'defaults')
  x_out = defaultopt;
  return;
end
 
% Check required inputs and optional arguments.
if nargin < 6
  error('WB_projection:NotEnoughInputs', ...
    'A, b, nvec, thr, tau, and regpar_c are required.');
end
if isempty(regpar_c) || ~isscalar(regpar_c) || ~isnumeric(regpar_c) || regpar_c < 0
  error('WB_projection:InvalidRegParC', ...
    'regpar_c must be a user-specified nonnegative scalar.');
end
if nargin < 7 || isempty(P)
  P = {[], []};
end
if nargin < 8 || isempty(options)
  options = defaultopt;
end
if nargin < 9
  trunc_options = [];
end
if nargin < 10
  trunc_mats = [];
end
 
if isempty(trunc_options) % Additional parameters for truncation
  nOuter = 1;
  nInner = 40;
else
  nOuter = trunc_options.nOuter;
  nInner = trunc_options.nInner;
end
 
if iscell(P) % Preconditioner
  P_le = P{1};
  P_ri = P{2};
else
  P_le = P;
  P_ri = [];
end
 
% Get options:
[m, n] = size(A);
defaultopt.Iter = min([m, n, 50]);
options = HyBRset(defaultopt, options);
 
solver = HyBRget(options,'InSolv',[],'fast');
regpar = HyBRget(options,'RegPar',[],'fast');
omega = HyBRget(options,'Omega',[],'fast');
maxiter = HyBRget(options,'Iter',[],'fast');
x_true = HyBRget(options,'x_true',[],'fast');
regstart = HyBRget(options,'BegReg',[],'fast');
degflat = HyBRget(options,'FlatTol',[],'fast');
mintol = HyBRget(options,'MinTol',[],'fast');
restol = HyBRget(options,'ResTol',[],'fast');
 
% Additional stopping-control options.
% Do not use HyBRget here because some HyBRget implementations cannot
% query user-defined fields such as StopRule/NoiseLevel/DPTau.
if isstruct(options) && isfield(options,'StopRule') && ~isempty(options.StopRule)
  stoprule = options.StopRule;
else
  stoprule = 'gcv_dp';
end
 
if isstruct(options) && isfield(options,'NoiseLevel')
  noiselevel = options.NoiseLevel;
else
  noiselevel = [];
end
 
if isstruct(options) && isfield(options,'DPTau') && ~isempty(options.DPTau)
  dptau = options.DPTau;
else
  dptau = 1.01;
end
 
% Parse stopping rule without changing the function interface.
% StopRule can be a string, e.g., 'maxit', 'gcv', 'dp', 'gcv_dp', 'all',
% or a cell array, e.g., {'gcv','dp'}. MaxIter is always enforced by options.Iter.
if iscell(stoprule)
  stopstr = '';
  for ii_stop = 1:numel(stoprule)
    stopstr = [stopstr '_' lower(stoprule{ii_stop})]; %#ok<AGROW>
  end
else
  stopstr = lower(stoprule);
end
useGCVstop = ~isempty(strfind(stopstr,'gcv')) || strcmp(stopstr,'all');
useDPstop = ~isempty(strfind(stopstr,'dp')) || strcmp(stopstr,'all');
 
% Discrepancy principle threshold. If NoiseLevel < 1, treat it as a
% relative noise level and use delta = NoiseLevel * ||b||. Otherwise,
% treat NoiseLevel as an absolute noise norm.
if isempty(noiselevel)
  dp_delta = [];
elseif noiselevel < 1
  dp_delta = noiselevel * norm(b);
else
  dp_delta = noiselevel;
end
 
adaptWGCV = ischar(regpar) && strcmpi(regpar,'wgcv') && ischar(omega) && strcmpi(omega,'adapt');
 
notrue = isempty(x_true);
 
%--------------------------------------------
%  The following is needed for RestoreTools:
%
if isa(A, 'psfMatrix')
  bSize = size(b);
  b = b(:);
  A.imsize = bSize;
  if ~notrue
    
    x_true = x_true(:);
  end
end
%
%  End of new stuff needed for RestoreTools
%--------------------------------------------
 
% Set-up output parameters:
outputparams = nargout>1;
if outputparams
  output.iterations = maxiter;
  output.GCVstop = [];
  output.Enrm = ones(maxiter,1);
    output.E_nor = ones(maxiter,1);
   output.E_nor_inne = ones(maxiter,1);
  output.Rnrm = ones(maxiter,1);
  output.Xnrm = ones(maxiter,1);
  output.Alpha = ones(maxiter,1);
  output.Alpha_c = ones(maxiter,1);
  output.U = [];
  output.V = [];
  output.T = [];
  output.G = [];
  output.Z = [];
  output.flag = 3;
  output.alpha = 0;
  output.E_opt = ones(maxiter,1);
  output.StopRule = stoprule;
  output.StopReason = 'maxit';
  output.DPthreshold = [];
  if ~isempty(dp_delta)
    output.DPthreshold = dptau * dp_delta;
  end
 
  % Runtime tracking.
  % CumTime(k) records the elapsed wall-clock time, in seconds, after the
  % kth projected iteration has produced a reconstruction. IterTime(k)
  % records the incremental time since the previous recorded iteration.
  output.CumTime = nan(maxiter,1);
  output.IterTime = nan(maxiter,1);
  output.TotalTime = nan;
end
 
% Test for a left preconditioner and define solver:
if isempty(P_le)
  beta = norm(b);
  U = (1 / beta)*b;
  handle = @FGK_l1;
  if ~isempty(P_ri)
    handle = @PLBD;
  end
else
  U = P_le\b;
  beta = norm(U); U = U / beta;
  handle = @PLBD;
end
 
 
% check/setup reycling
if isempty(trunc_mats)
  no_recycle = 1; % build recycling matrices at start
else
  no_recycle = 0;
  % W and S must be defined; check size
  if isempty(trunc_mats.W) || size(trunc_mats.W, 2) > (nInner - 1)
    fprintf('trunc_mats.W must exist \n');
    fprintf('number of recycling vectors must be less than max_mm+2 \n\n');
    return
  else
    W = trunc_mats.W; % W is assumed to have orthonormal columns
    kk = size(W,2);
    if isempty(trunc_mats.x) % no initial solution is given
      x_out = zeros(size(A,2),1);
    else                      % initial solution is given
      x_out = trunc_mats.x;
      eta = W'*x_out; xhat = x_out - W*eta; etahat = norm(xhat);
      xhat = xhat / etahat;
      W = [W xhat];
      kk = kk+1;
    end
  end
  if isempty(trunc_mats.Y)
    Y = zeros(size(A,1),size(W,2));
    for j = 1:kk
      Y(:,j) = A*W(:,j);
    end
    [Y,R] = qr(Y,0);
  else
    Y = trunc_mats.Y;
    R = trunc_mats.R;
  end
end
 
%% Main Code Begins Here
runtime_tic = tic;
last_recorded_time = 0;
 
T = []; G = []; V = []; GCV = []; Omega = []; Z = [];
terminate = 0;
if useGCVstop && ischar(regpar) && (strcmpi(regpar,'wgcv') || strcmpi(regpar,'gcv'))
  terminate = 1;
end
gcv_warning = 0;
iter = 0;
L = ones(size(A,2),1);
 
% Incremental QR factors for the correction basis Z.
% In the WB/recycling branch, alpha is selected from the correction
% subproblem (G,Z), not from the augmented [W,Z] problem.
QZ = []; RZ = [];
 
for outer = 1:nOuter
  if no_recycle
    % first run is standard GK with nInner iterations
    for inner = 1:nInner
      iter = iter+1;
      [U,T,G, V, Z] = feval(handle, A, U, T,G, V, Z, L, P_le, P_ri, options,1);
       [QZ,RZ] = local_qr_append_col(QZ,RZ,Z(:,end));
       Rs = RZ;
      if strcmp(regpar,'optimal')
        options.Zx = Z;
      end
      if inner >= 2 % Otherwise skip
        vector = (beta*eye(size(G,2)+1,1)); % assumes b is first vector in V
        switch solver
          case{'tsvd', 'tikhonov'}% Solve projected problem using TSVD/Tikhonov at each iteration
            [Ub, Sb, ~] = svd(G);   % full U is needed by WGCV/GCVstopfun
            if adaptWGCV %Use the adaptive, weighted GCV method
                 if iter>1
                Omega(iter) = min(1, findomega(Ub'* vector,diag(Sb), solver));
              else
                Omega(1) = 1;
              end
               alpha0 = -0.5;               
              options.Omega = mean(Omega);
              errhan = @(p)WGCV_l1(p,G,Rs,vector,options.Omega);
               alpha = fminunc(errhan,alpha0);
               
            else
                alpha = regpar;
            end
            IR = alpha^2*(Rs'*Rs);
            GIR = G'*G + IR;
            C = GIR\G';
            y = C*vector;
            % Compute the GCV value used to find the stopping criteria
            GCV(iter-1) = GCVstopfun(alpha, Ub(1,:)', diag(Sb), beta, m, n, solver);
            
            % Determine if GCV wants us to stop
            if iter > 2 && terminate
              %%-------- If GCV curve is flat, we stop -----------------------
              if abs(GCV(iter-1) - GCV(iter-2)) / max(abs(GCV(regstart-1)), eps) < degflat
                x_out =Z*y; % Return the solution at (i-1)st iteration
                % Test for a right preconditioner:
                if ~isempty(P_ri)
                  x_out = P_ri \ x_out;
                end
                if notrue || useGCVstop %Set all the output parameters and return
                  if outputparams
                    output.U = U;
                    output.V = V;
                    output.Z = Z;
                    output.G = G;
                    output.T = T;
                    output.GCVstop = GCV(:);
                    output.iterations = iter-1;
                    output.flag = 1;
                    output.StopReason = 'gcv_flat';
                    output.alpha = alpha; % Reg Parameter at the (i-1)st iteration
                  end
                  if exist('h','var') && ishandle(h), close(h); end
                  %--------------------------------------------
                  %  The following is needed for RestoreTools:
                  %
                  if isa(A, 'psfMatrix')
                    x_out = reshape(x_out, bSize);
                  end
                  %
                  %  End of new stuff needed for RestoreTools
                  %--------------------------------------------
                  return;
                else % Flat GCV curve means stop, but continue since have x_true
                  if outputparams
                    output.iterations = iter-1; % GCV says stop at (i-1)st iteration
                    output.flag = 1;
                    output.StopReason = 'gcv_flat';
                    output.alpha = alpha; % Reg Parameter at the (i-1)st iteration
                  end
                end
                terminate = 0; % Solution is already found!
                
                %%--- Have warning : Avoid bumps in the GCV curve by using a
                %    window of (mintol+1) iterations --------------------
              elseif gcv_warning && length(GCV) > iterations_save + mintol %Passed window
                if all(GCV(iterations_save) < GCV(iterations_save+1:end))
                  % We should have stopped at iterations_save.
                  x_out = x_save;
                  % Test for a right preconditioner:
                  if ~isempty(P_ri)
                    x_out = P_ri \ x_out;
                  end
                  if notrue || useGCVstop %Set all the output parameters and return
                    if outputparams
                      output.U = U;
                      output.V = V;
                      output.G = G;
                      output.T = T;
                       output.Z = Z;
                      output.GCVstop = GCV(:);
                      output.iterations = iterations_save;
                      output.flag = 2;
                      output.StopReason = 'gcv_min';
                      output.alpha = alpha_save;
                    end
                    if exist('h','var') && ishandle(h), close(h); end
                    %--------------------------------------------
                    %  The following is needed for RestoreTools:
                    %
                    if isa(A, 'psfMatrix')
                      x_out = reshape(x_out, bSize);
                    end
                    %
                    %  End of new stuff needed for RestoreTools
                    %--------------------------------------------
                    return;
                  else % GCV says stop at iterations_save, but continue since have x_true
                    if outputparams
                      output.iterations = iterations_save;
                      output.flag = 2;
                      output.StopReason = 'gcv_min';
                      output.alpha = alpha_save;
                    end
                  end
                  terminate = 0; % Solution is already found!
                  
                else % It was just a bump... keep going
                  gcv_warning = 0;
                  x_out = [];
                  iterations_save = maxiter;
                  alpha_save = 0;
                end
                
                %% ----- No warning yet: Check GCV function---------------------
              elseif ~gcv_warning
                if GCV(iter-2) < GCV(iter-1) %Potential minimum reached.
                  gcv_warning = 1;
                  % Save data just in case.
                  x_save = Z*y;
                  iterations_save = iter-1;
                  alpha_save = alpha;
                end
              end
            end
            
          case 'none'
            y = G \ vector;
        end
 
        % Record cumulative and per-iteration runtime after the current
        % projected solution has been computed.
        if outputparams
          current_time = toc(runtime_tic);
          output.CumTime(iter,1) = current_time;
          output.IterTime(iter,1) = current_time - last_recorded_time;
          output.TotalTime = current_time;
          last_recorded_time = current_time;
        end
 
        ri = vector - G*y;
        ri_nrm = norm(ri);
        Vy = Z*y;
        r_out = b - A*Vy;
        rnrm = norm(r_out);
        dxj = 2*sqrt(Vy(:).^2+eps);
        dxj(dxj<thr) = tau;
        L = dxj(:).^(1/2);
        
        if outputparams
            temp_Vy = Vy(:);
             temp_Vy(temp_Vy<0) = 0;
          if ~notrue
            
            temp_true = x_true(:);
            output.E_nor_inne(iter,1) = norm((temp_Vy(:)-min(temp_Vy(:)))/(max(temp_Vy(:))-min(temp_Vy(:)))-temp_true(:)/max(temp_true(:)))/norm(temp_true(:)/max(temp_true(:)));
 
            temp_Vy = (temp_Vy(:)-min(temp_Vy(:)))/(max(temp_Vy(:))-min(temp_Vy(:)));
             output.E_nor(iter,1) = norm(temp_Vy(:)-temp_true(:))/norm(temp_true(:));
              temp_Vy = reshape(temp_Vy,nvec);
              temp_true = reshape(temp_true,nvec);
              % Assume temp_Vy and temp_true are given 3D matrices of size [55,55,15]
 
% Define slice sizes
slices = [ceil(nvec(3)/4), ceil(nvec(3)/4), ceil(nvec(3)/4), nvec(3)-ceil(nvec(3)/4)*3];  % Total sum is 15
num_slices = length(slices);  % Number of partitions
 
% Starting index
start_idx = 1;
 
for i_slice = 1:num_slices
    % Define end index for the current slice
    end_idx = start_idx + slices(i_slice) - 1;
    
    % Extract corresponding slices from temp_Vy and temp_true
    Vy_part = temp_Vy(:,:,start_idx:end_idx);
    true_part = temp_true(:,:,start_idx:end_idx);
    
    % Compute relative error for the current slice
    error_norm = norm(Vy_part(:) - true_part(:), 2);  % L2 norm of error
    
    % Avoid division by zero
    output.E_nor_slice(iter,i_slice) = error_norm;
    % Update start index for next slice
    start_idx = end_idx + 1;
end
             
            output.Enrm(iter,1) = norm(Vy(:)-x_true(:))/norm(x_true(:));
            x_opt = V*(V\x_true);
            output.E_opt(iter,1) = norm( x_opt(:)-x_true(:) )/norm(x_true(:));
          end
 
          output.Rnrm(iter,1) = norm(A*temp_Vy(:)/max(temp_Vy(:))-b);
          output.Xnrm(iter,1) = norm(Vy);
          output.Alpha(iter,1) = alpha;
        end
        
        % Optional discrepancy-principle stopping on the full residual.
        if useDPstop && ~isempty(dp_delta) && rnrm <= dptau*dp_delta
          x_out = Vy;
          if outputparams
            output.iterations = iter;
            output.flag = 5;
            output.StopReason = 'DP';
            output.alpha = alpha;
          end
          return
        end
 
        % Explicit MaxIter stopping via options.Iter.
        if iter >= maxiter
          x_out = Vy;
          if outputparams
            output.iterations = iter;
            output.flag = 3;
            output.StopReason = 'maxit';
            output.alpha = alpha;
          end
          return
        end
 
        if ri_nrm < restol(1)*beta
           Vy =Z*y; % FIX - already above
          x_out = Vy; % FIX - already above
          if outputparams
            output.iterations = iter;
            output.flag = 4;
            output.StopReason = 'restol';
          end
          return
        end
      end
    end
     output.G = G;
      output.T = T;
    output.V = V;
    output.U = U;
    output.Z = Z;
    
    % --------- Select a recycle space W --------------------------
    y_1 = y; % regularized solution y
    Vy = Z*y_1;
    %%%%%%%%%%%%%% Perform Compression %%%%%%%%%%%%%%%%%%%
    [kk,W,Y,R] = compression(A,G,Z,trunc_options,Vy,vector,y,[]);
    
    x_out = Vy;
    no_recycle = 0;
    
  else % no_reycle = false, but possible first outer iteration
    if outer == 1 % set up
      r_out = b - A*x_out;
    else
      
      Vupd = W(:,end);
      r_out = b - A*Vupd;
    end
    
    % build up space again
    % initialize next inner
    r_upd = r_out;
    
    zeta = (Y'*r_upd);
    btil = r_upd - Y*zeta;
    betaInn = norm(btil); U = btil/betaInn;
    G = []; T = []; V = []; H = []; GCV = []; Omega = []; Z = [];
    handle = @recyclingFGK_l1_2; %% recyclingGKB process
    
    % Restart incremental QR factors for the correction basis in this
    % recycling/WB cycle. We do not maintain QR([W,Z]) because the
    % alternating WB-IPM treats the warm-basis coefficient and correction
    % subproblems separately.
    QZ = []; RZ = [];
    
    for inner = 1:nInner-kk
      iter = iter+1;
      [U, T,G, V, H,Z] = feval(handle, A, U, T,G, V, H,Z,L, P_le, P_ri, W, Y, options);
      
      [QZ,RZ] = local_qr_append_col(QZ,RZ,Z(:,end));
      Rs0 = RZ;
      if strcmp(regpar,'optimal')
        options.Vx = Z;
      end
 
      mm = inner;
 
      % Split the augmented right-hand side according to the alternating
      % WB-IPM formulation. The correction regularization parameter is
      % selected from the Z-subproblem only.
      vector_c = zeta;
      vector_z = (betaInn*eye(size(G,2)+1,1)); % betaInn*e1
 
      % Keep BB/vector only for diagnostics, optional GCV stopping output,
      % the no-regularization fallback, and the compression step. They are
      % no longer used for WGCV parameter selection.
      BB = zeros(kk+mm+1,kk+mm);
      BB(1:kk,1:kk) = R;
      BB(1:kk,kk+1:kk+mm) = H;
      BB(kk+1:kk+mm+1,kk+1:kk+mm) = G;
      vector = zeros(mm+kk+1,1);
      vector(1:kk) = vector_c;
      vector(kk+1:kk+mm+1) = vector_z;
 
      y = zeros(kk+mm,1);
      switch solver
        case{'tsvd', 'tikhonov'} % Solve the correction subproblem in Z
          [Ub,Sb,~] = svd(G);   % full U is needed by WGCV/GCVstopfun
 
          if adaptWGCV
            if iter > 1
              Omega(iter-1) = min(1, findomega(Ub'*vector_z,diag(Sb),solver));
            else
              Omega(1) = 1;
            end
            alpha0 = -0.5;
            options.Omega = mean(Omega);
            errhan = @(p)WGCV_l1(p,G,Rs0,vector_z,options.Omega);
            alpha = fminunc(errhan,alpha0);
          else
            alpha = regpar;
          end
 
          % Correction solve:
          %   min_d ||G*d - vector_z||^2 + alpha^2 ||RZ*d||^2.
          IR0 = alpha^2*(Rs0'*Rs0);
          GIR = G'*G + IR0;
          d = GIR\(G'*vector_z);
          y(kk+1:kk+mm) = d;
 
          % Warm-basis coefficient update. The user-specified regpar_c is
          % used directly; WGCV is reserved for the correction subproblem.
          c_right = vector_c - H*d;
          alpha_c = regpar_c;
 
          if kk == 1
            y(1:kk) = R*c_right/(R^2 + alpha_c^2);
          else
            y(1:kk) = (R'*R + alpha_c^2*eye(kk)) \ (R'*c_right);
          end
 
          if outputparams
            output.Alpha(iter) = alpha;
            output.Alpha_c(iter) = alpha_c;
          end
          % Compute the GCV value used to find the stopping criteria.
          % Since alpha is selected for the correction subproblem, betaInn
          % is the corresponding right-hand-side norm.
          if iter > 1
            GCV(iter-1) = GCVstopfun(alpha, Ub(1,:)', diag(Sb), betaInn, m, n, solver);
          end
          % Determine if GCV wants us to stop
          if iter > 2 && terminate
            %%-------- If GCV curve is flat, we stop -----------------------
            if abs(GCV(iter-1) - GCV(iter-2)) / max(abs(GCV(regstart-1)), eps) < degflat
              x_out = [W,Z]*y; % Return the solution at (i-1)st iteration
              % Test for a right preconditioner:
              if ~isempty(P_ri)
                x_out = P_ri \ x_out;
              end
              if notrue || useGCVstop %Set all the output parameters and return
                if outputparams
                  output.U = [Y,U];
                  output.V = [W,V];
                  output.Z = [W,Z];
                  output.B = BB;
                  output.GCVstop = GCV(:);
                  output.iterations = iter-1;
                  output.flag = 1;
                  output.StopReason = 'gcv_flat';
                  output.alpha = alpha; % Reg Parameter at the (i-1)st iteration
                end
                if exist('h','var') && ishandle(h), close(h); end
                %--------------------------------------------
                %  The following is needed for RestoreTools:
                %
                if isa(A, 'psfMatrix')
                  x_out = reshape(x_out, bSize);
                end
                %
                %  End of new stuff needed for RestoreTools
                %--------------------------------------------
                return;
              else % Flat GCV curve means stop, but continue since have x_true
                if outputparams
                  output.iterations = iter-1; % GCV says stop at (i-1)st iteration
                  output.flag = 1;
                  output.StopReason = 'gcv_flat';
                  output.alpha = alpha; % Reg Parameter at the (i-1)st iteration
                end
              end
              terminate = 0; % Solution is already found!
              
              %%--- Have warning : Avoid bumps in the GCV curve by using a
              %    window of (mintol+1) iterations --------------------
            elseif gcv_warning && length(GCV) > iterations_save + mintol %Passed window
              if all(GCV(iterations_save) < GCV(iterations_save+1:end))
                % We should have stopped at iterations_save.
                x_out = x_save;
                % Test for a right preconditioner:
                if ~isempty(P_ri)
                  x_out = P_ri \ x_out;
                end
                if notrue || useGCVstop %Set all the output parameters and return
                  if outputparams
                    output.U = [Y,U];
                    output.V = [W,V];
                    output.Z = [W,Z];
                    output.B = BB;
                    output.GCVstop = GCV(:);
                    output.iterations = iterations_save;
                    output.flag = 2;
                    output.StopReason = 'gcv_min';
                    output.alpha = alpha_save;
                  end
                  if exist('h','var') && ishandle(h), close(h); end
                  %--------------------------------------------
                  %  The following is needed for RestoreTools:
                  %
                  if isa(A, 'psfMatrix')
                    x_out = reshape(x_out, bSize);
                  end
                  %
                  %  End of new stuff needed for RestoreTools
                  %--------------------------------------------
                  return;
                else % GCV says stop at iterations_save, but continue since have x_true
                  if outputparams
                    output.iterations = iterations_save;
                    output.flag = 2;
                    output.StopReason = 'gcv_min';
                    output.alpha = alpha_save;
                  end
                end
                terminate = 0; % Solution is already found!
                
              else % It was just a bump... keep going
                gcv_warning = 0;
                iterations_save = maxiter;
                alpha_save = 0;
              end
              
              %% ----- No warning yet: Check GCV function---------------------
            elseif ~gcv_warning
              if GCV(iter-2) < GCV(iter-1) %Potential minimum reached.
                gcv_warning = 1;
                % Save data just in case.
                x_save = [W,Z]*y;
                iterations_save = iter-1;
                alpha_save = alpha;
              end
            end
          end
        case 'none' % Solve projected augmented problem with no regularization
          y = BB \ vector;
      end      
            % Record cumulative and per-iteration runtime after the current
      % projected solution has been computed.
      if outputparams
        current_time = toc(runtime_tic);
        output.CumTime(iter,1) = current_time;
        output.IterTime(iter,1) = current_time - last_recorded_time;
        output.TotalTime = current_time;
        last_recorded_time = current_time;
      end
 
      x1 = y(1:kk); x2 = y(kk+1:mm+kk); % x1 = chat and x2 = d in notes
      ri = vector - BB*y;
      ri_nrm = norm(ri);      
      
      
      if outer > 1
        c = x1;
        d = x2;
      else
        c = x1;
        d = x2;
      end
 
      x_out = W*c + Z*d; % x_out is the adding vector
     z_out = Z*d;
      r_out = b - A*x_out;
      rnrm = norm(r_out);
        dxj = 2*sqrt(z_out(:).^2+eps);
        dxj(dxj<thr) = tau;
        L = dxj(:).^(1/2);
            if outputparams && iter > 1
           temp_out = x_out(:);
           temp_out(temp_out<0) = 0;
        if ~notrue
             
            
             temp_Vy = temp_out;
              temp_true = x_true(:);
              output.E_nor_inne(iter-1,1) = norm((temp_Vy(:)-min(temp_Vy(:)))/(max(temp_Vy(:))-min(temp_Vy(:)))-temp_true(:)/max(temp_true(:)))/norm(temp_true(:)/max(temp_true(:)));
            temp_Vy = (temp_Vy(:)-min(temp_Vy(:)))/(max(temp_Vy(:))-min(temp_Vy(:)));
             temp_Vy = reshape(temp_Vy,nvec);
              temp_true = reshape(temp_true,nvec);
              % Assume temp_Vy and temp_true are given 3D matrices of size [55,55,15]
 
% Define slice sizes
slices = [ceil(nvec(3)/4), ceil(nvec(3)/4), ceil(nvec(3)/4), nvec(3)-ceil(nvec(3)/4)*3];  % Total sum is 15
num_slices = length(slices);  % Number of partitions
 
% Starting index
start_idx = 1;
 
for i_slice = 1:num_slices
    % Define end index for the current slice
    end_idx = start_idx + slices(i_slice) - 1;
    
    % Extract corresponding slices from temp_Vy and temp_true
    Vy_part = temp_Vy(:,:,start_idx:end_idx);
    true_part = temp_true(:,:,start_idx:end_idx);
    
    % Compute relative error for the current slice
    error_norm = norm(Vy_part(:) - true_part(:), 2);  % L2 norm of error
    
    % Avoid division by zero
 
    output.E_nor_slice(iter,i_slice) = error_norm;
    % Update start index for next slice
    start_idx = end_idx + 1;
end
             output.E_nor(iter-1,1) = norm(temp_Vy(:)-temp_true(:))/norm(temp_true(:));
          output.Enrm(iter-1,1) = norm(x_out(:)-x_true(:))/norm(x_true(:));
          % projection of x_true onto solution space
          x_opt = W*(W'*x_true) + V*(V'*x_true);
          output.E_opt(iter-1,1) = norm( x_opt(:) - x_true(:) )/norm(x_true(:));
        end
        output.Rnrm(iter-1,1) = norm(A*temp_out(:)/max(temp_out(:))-b);
        output.Xnrm(iter-1,1) = norm(x_out);
      end
      
      % Optional discrepancy-principle stopping on the full residual.
      if useDPstop && ~isempty(dp_delta) && rnrm <= dptau*dp_delta
        trunc_mats.W = W;
        trunc_mats.R = R;
        trunc_mats.Y = Y;
        if outputparams
          output.iterations = iter;
          output.flag = 5;
          output.StopReason = 'DP';
          output.alpha = alpha;
        end
        return
      end
 
      % Explicit MaxIter stopping via options.Iter.
      if iter >= maxiter
        trunc_mats.W = W;
        trunc_mats.R = R;
        trunc_mats.Y = Y;
        if outputparams
          output.iterations = iter;
          output.flag = 3;
          output.StopReason = 'maxit';
          output.alpha = alpha;
        end
        return
      end
 
      if ri_nrm < restol(1)*beta
        trunc_mats.W = W;
        trunc_mats.R = R;
        trunc_mats.Y = Y;
        if outputparams
          output.iterations = iter;
          output.flag = 4;
          output.StopReason = 'restol';
        end
        return
      end
      
    end
    
    % --------- Select a recycle space W (do compression) ---------------
    y_1 = [c;d]; % regularized solution, c is coefficient of input bases W, d is the coefficient of augmented bases
    
    y = y_1;
    x1 = y_1(1:kk); x2 = y_1(kk+1:mm+kk); 
    
    if outer > 1
      c = x1;
      d = x2;
    else
      c = x1;
      d = x2;
    end
    x_out = W*c + Z*d;
    x_out(x_out<0)=0;
    WV = [W Z];
    %%%%%%%%%%%%%% Perform Compression %%%%%%%%%%%%%%%%%%%
    [kk,W,Y,R] = compression(A,BB,WV,trunc_options,x_out,vector,y,W);
  end
  
end
 
trunc_mats.W = W;
trunc_mats.R = R;
trunc_mats.Y = Y;
output.iterations = iter;
 
output.Enrm = output.Enrm(1:iter-1);
output.E_nor = output.E_nor(1:iter-1);
output.E_nor_inne = output.E_nor_inne(1:iter-1);
 
output.E_opt = output.E_opt(1:iter-1);
output.Rnrm = output.Rnrm(1:iter-1);
output.Xnrm = output.Xnrm(1:iter-1);
output.Alpha = output.Alpha(1:iter-1);
output.Alpha_c = output.Alpha_c(1:iter-1);
if isfield(output,'CumTime')
  output.CumTime = output.CumTime(1:iter);
  output.IterTime = output.IterTime(1:iter);
  if iter > 0 && ~isempty(output.CumTime)
    output.TotalTime = output.CumTime(iter);
  end
end
end
 
% -----------------------SUBFUNCTION---------------------------------------
function [Qnew,Rnew] = local_qr_append_col(Q,R,z)
%LOCAL_QR_APPEND_COL  Incremental thin-QR update after appending one column.
% Given an existing thin QR factorization A = Q*R, return the factorization
% of [A z].  This avoids recomputing qr([A z],0) at every iteration.
% A second modified Gram-Schmidt pass is used for numerical stability.
if isempty(Q)
  r = norm(z);
  if r <= 10*eps*max(1,norm(z))
    Qnew = zeros(size(z));
    Rnew = 0;
  else
    Qnew = z/r;
    Rnew = r;
  end
  return;
end
 
h = Q'*z;
v = z - Q*h;
% second pass for stability
h2 = Q'*v;
h = h + h2;
v = v - Q*h2;
r = norm(v);
 
if r <= 10*eps*max(1,norm(z))
  % Rare near-breakdown fallback: recover the original matrix and call full QR.
  % This preserves robustness without changing the algorithmic logic.
  [Qnew,Rnew] = qr([Q*R z],0);
else
  q = v/r;
  Qnew = [Q q];
  Rnew = [R h; zeros(1,size(R,2)) r];
end
end
 
% -----------------------SUBFUNCTION---------------------------------------
function omega = findomega(bhat, s, insolv)
%
%  
%  This function computes a value for the omega parameter.
%
%  The method: Assume the 'optimal' regularization parameter to be the
%  smallest singular value.  Then we take the derivative of the GCV
%  function with respect to alpha, evaluate it at alpha_opt, set the
%  derivative equal to zero and then solve for omega.
%
%  Input:   bhat -  vector U'*b, where U = left singular vectors
%              s -  vector containing the singular values
%         insolv -  inner solver method for HyBR
%
%  Output:     omega - computed value for the omega parameter.
 
%
%   First assume the 'optimal' regularization parameter to be the smallest
%   singular value.
%
 
%
% Compute the needed elements for the function.
%
m = length(bhat);
n = length(s);
switch insolv
  case 'tsvd'
    k_opt = n;
    omega = (m*bhat(k_opt)^2) / (k_opt*bhat(k_opt)^2 + 2*bhat(k_opt+1)^2);
    
  case 'tikhonov'
    t0 = sum(abs(bhat(n+1:m)).^2);
    alpha = s(end);
    s2 = abs(s) .^ 2;
    alpha2 = alpha^2;
    
    tt = 1 ./ (s2 + alpha2);
    
    t1 = sum(s2 .* tt);
    t2 = abs(bhat(1:n).*alpha.*s) .^2;
    t3 = sum(t2 .* abs((tt.^3)));
    
    t4 = sum((s.*tt) .^2);
    t5 = sum((abs(alpha2*bhat(1:n).*tt)).^2);
    
    v1 = abs(bhat(1:n).*s).^2;
    v2 = sum(v1.* abs((tt.^3)));
    
    %
    % Now compute omega.
    %
    omega = (m*alpha2*v2)/(t1*t3 + t4*(t5 + t0));
    
  otherwise
    error('Unknown solver');
end
end
 
%% ---------------SUBFUNCTION ---------------------------------------
function G = GCVstopfun(alpha, u, s, beta, m, n, insolv)
%
%  G = GCVstopfun(alpha, u, s, beta, n, insolv)
%  This function evaluates the GCV function G(i, alpha), that will be used
%     to determine a stopping iteration.
%
% Input:
%   alpha - regularization parameter at the kth iteration of HyBR
%       u - P_k^T e_1 where P_k contains the left singular vectors of B_k
%       s - singular values of bidiagonal matrix B_k
%    beta - norm of rhs b
%     m,n - size of the ORIGINAL problem (matrix A)
%  insolv - solver for the projected problem
%
 
k = length(s);
beta2 = beta^2;
 
switch insolv
  case 'tsvd'
    t2 = (abs(u(alpha+1:k+1))).^2;
    G = n*beta2*(sum(t2))/((m - alpha)^2);
  case 'tikhonov'
    s2 = abs(s) .^ 2;
    alpha2 = alpha^2;
    
    t1 = 1 ./ (s2 + alpha2);
    t2 = abs(alpha2*u(1:k) .* t1) .^2;
    t3 = s2 .* t1;
    
    num = beta2*(sum(t2) + abs(u(k+1))^2)/n;
    den = ( (m - sum(t3))/n )^2;
    G = num / den;
    
  otherwise
    error('Unknown solver');
end
end

