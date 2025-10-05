function err = WGCV_l1(alpha,M,R,vector,omega)
m = size(M,2);
IR = alpha^2*(R'*R);
GIR = M'*M + IR;
% vector = zeros(m+1,1);
% vector(1) = 1;
% vector = beta*vector;
Gtb = M'*vector;
f = GIR\Gtb;
part_r = M*f - vector;
whole_r = part_r'*part_r;

MC = GIR\(M'*M);
tr  = trace(MC);
err = whole_r/(m - omega*tr)^2;
% yx = Z*f;
% n = size(Z,1)/2;
% s1 = Q*yx(1:n);
% s2 = L*yx(n+1:end);
% s = s1+s2;
% err = norm(s(land) - xtrue(land));
end