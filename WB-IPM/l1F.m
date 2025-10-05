function [f] = l1F(A,b,lambda,N)
%l1 regularization: find the solution to the problem 
%                  argmin_x{F(x):= ||Ax-b||^2+lambda*||x||_1}
%                  through the use of FISTA, a natural extension of the
%                  gradient-based method
%  Inputs:
%         L: smallest Lipschitz constant of gradient(f),
%            L(f)=2*lambda_max(A'A)
%         A: design matrix
%         b: measurements/response vector
%         lambda: regularization weight
%         N: number of iterations
% Output:
%         f: minimizer of the reularization problem
M=size(A,2);
x=zeros(M,1);
y=zeros(M,1);
t=1;
L=2*eigs(A'*A,1);
alpha=lambda/L;
ATA=A'*A;
ATb=A'*b;
for k=1:N

    z=y-2/L*(ATA*y-ATb);
    xx=tau(alpha,z);
    tt=(1+sqrt(1+4*t^2))/2;
    y=xx+(t-1)/tt*(xx-x);
    x=xx;
    t=tt;
end
    f=y;
%     z=x-2/L*(ATA*x-ATb);
%     x=tau(alpha,z);
% 
% end
%     f=x;
    
end

