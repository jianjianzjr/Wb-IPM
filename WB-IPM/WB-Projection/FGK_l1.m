function [U, T,G, V, Z] = FGK_l1(A, U, T,G, V,Z,L, P_le, P_ri, options, save)
%
%     [U, B, V] = LBD(A, U, B, V, P_le, P_ri, options)
%
%  Perform one step of Lanczos bidiagonalization with or without
%  reorthogonalization, WITHOUT preconditioner here.
%
% Input:
%          A - matrix
%       U, V - accumulation of vectors
%          B - bidiagonal matrix
% P_le, P_ri - input ignored
%    options - structure from HyBR (see HyBRset)
%
% Output:
%       U, V - updated "orthogonal" matrix
%          B - updated bidiagonal matrix
%
%  Refs:
%   [1] Paige and Saunders, "LSQR an algorithm for sparse linear
%       equations an sparse least squares", ACG Trans. Gath Software,
%       8 (1982), pp. 43-71.
%   [2] Bjorck, Grimme and Van Dooren, "An implicit shift bidiagonalization
%       algorithm for ill-posed systems", BIT 34 (11994), pp. 520-534.
%
%   J.Chung and J. Nagy 3/2007

% Determine if we need to do reorthogonalization or not.
if nargin < 8
  save = 0;
end
reorth = strcmp(HyBRget(options,'Reorth'), {'on'});

% m = size(U,1);
k = size(G,2)+1;

%  if k == 1
    v = Atransp_times_vec(A, U(:,k)); v = v(:);
%  else
    % if k>1
%      v = Atransp_times_vec(A, U(:,k)); v = v(:);
    for i = 1:k-1
        
        T(i,k)=V(:,i)'*v;
        v = v - T(i,k)*V(:,i);
    end
% end
    T(k,k) = norm(v);
    v = v / T(k,k);
    %
%     z = diag(L).*v;
    z = double(L) .* double(v);
    u = A_times_vec(A, z); u = u(:);
    for i = 1:k
        G(i,k) = U(:,i)'*u;
        u = u - G(i,k)*U(:,i);
    end
    G(k+1,k) = norm(u);
    u = u / G(k+1,k);
    
    U(:,k+1) = u;
    V(:,k) = v;
    Z(:,k) = z;
     

