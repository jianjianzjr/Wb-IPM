function err = WGCV_l1(alpha,M,R,vector,omega)
m = size(M,1);   % number of projected data equations, e.g. k+1
IR = alpha^2*(R'*R);
GIR = M'*M + IR;

Gtb = M'*vector;
f = GIR\Gtb;

part_r = M*f - vector;
whole_r = part_r'*part_r;

MC = GIR\(M'*M);
tr  = trace(MC);

err = whole_r/(m - omega*tr)^2;
end
