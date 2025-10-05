function [U,T,G, V, H, Z] = recyclingFGK_l1_2(A, U, T,G, V, H,Z,L ,P_le, P_ri, W, Y, options)
%
%     [U, B, V, H] = recyclingGKB(A, U, B, V, H, P_le, P_ri, options)
%
%  Perform one step of recycling GKB without reorthogonalization and 
%   without preconditioner.
%
% Input:
%          A - matrix
%       U, V - accumulation of vectors
%          H - ADD definitions
%          B - bidiagonal matrix
% P_le, P_ri - inputs ignored
%       W, Y - Additional vectors for recycling
%    options - structure from HyBR (see HyBRset), ignored here
%
% Output:
%       U, V - updated "orthogonal" matrix
%          B - updated bidiagonal matrix
%          H - updated upper triangular portion of the matrix
%
%  Reference:
%   Chung, de Sturler, and Jiang. "Hybrid Projection Methods with
%           Recycling for Inverse Problems". SISC, 2020.
%
% J. Chung, E. de Sturler, and J. Jiang, 2020

k = size(G,2)+1;

%  if k == 1
%     Au = A'*U(:,k);
% %   WtAu = W'*Au;
% %   v = Au - W*WtAu;
%   v = Au;
%     v = Atransp_times_vec(A, U(:,k)); v = v(:);
% Au =A'*U(:,k); 
% Ytu = Y'*U(:,k);
% YYtu = Y*Ytu;
% % AuY = Y'*Au; 
% v = A_times_vec(A',U(:,k)- YYtu);
%  else
%       Au = A'*U(:,k);
% %   WtAu = W'*Au;
% %   v = Au - W*WtAu;
% v = Au;
Ytu = Y'*U(:,k);
YYtu = Y*Ytu;
% AuY = Y'*Au; 
v = A_times_vec(A',U(:,k)- YYtu);

    % if k>1
    for i = 1:k-1
        T(i,k)=V(:,i)'*v;
        v = v - T(i,k)*V(:,i);
    end
% end
    T(k,k) = norm(v);
    v = v / T(k,k);
    %
    invLv = double(L) .* double(v);
    z= invLv - W*(W'*invLv);
    
    Az =A_times_vec(A, z); YtAz = Y'*Az; H(:,k) = YtAz;
u = Az - Y*YtAz;
%     u = A_times_vec(A, z); u = u(:);
    for i = 1:k
        G(i,k) = U(:,i)'*u;
        u = u - G(i,k)*U(:,i);
    end
    G(k+1,k) = norm(u);
    u = u / G(k+1,k);
    
    U(:,k+1) = u;
    V(:,k) = v;
    Z(:,k) = z;
