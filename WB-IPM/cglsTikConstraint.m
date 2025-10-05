function [X,rho,eta,F] = cglsTikConstraint(A,b,K,lambda,L,options)
%CGLSAIR Conjugate gradient algorithm applied implicitly to the normal equations
%
% X = cgls(A,b,K)
%
% Performs max(K) steps of the conjugate gradient algorithm applied
% implicitly to the normal equations A'*A*x = A'*b.
%


% Initialization.
k = max(K);
if (k < 1), error('Number of steps k must be positive'), end
n = size(A,2); X = zeros(n,length(K));

%Constraint conditions
lbound = options.lbound;
ubound = options.ubound;

% Prepare for CG iteration.
x = zeros(n,1);
d = A'*b;
r = b;
normr2 = d'*d;
temp =[A;lambda.*L];
% Iterate.
ksave = 0;
for j=1:k

  % Update x and r vectors.
  Ad = A*d; 
  temp1 =temp*d; 
  alpha = normr2/(temp1'*temp1);
  x  = x + alpha*d;
  r  = r - alpha*Ad;
  s  = A'*r;

  % Update d vector.
  normr2_new = s'*s;
  beta = normr2_new/normr2;
  normr2 = normr2_new;
  d = s + beta.*d;
  
  % Enforce any lower and upper bounds (scalars or xk-sized vectors).
  if ~isempty(lbound)
    x = max(x,lbound(:));
  end
  if ~isnan(ubound)
    x = min(x,ubound(:));
  end
  
  % Save, if wanted.
  if any(K==j)
      ksave = ksave + 1;
      X(:,ksave) = x;
  end
  
  % Stopping rules based on redurial
  if norm(A*x-b)<3.26*0.05*norm(b)
      break;
  end

end
X = X(:,1:ksave);
