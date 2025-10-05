function x_true = get_x_true(nx, nu, ell, nt, nu_t, ell_t)
%
% A code to generate an approximate sample from the prior
% Julianne
%
%

if nargin < 4
    nt = 1;
end
if nt == 1
    %% Covariance kernel for spatial prior
    xmin = [0 0];           % Coordinates of left corner
    xmax = [1 1];           % Coordinates of right corner
    nvec = [nx, nx];         % Number of points
    theta = [1, 1];
    sigma2 = 1;
    
    % nu = .5; ell = .25;
    f1 = @(r) sigma2*matern(r,nu,ell);
    k = @(r)f1(r);
    % alpha = 100; l = 0.01;
    % f2 = @(r) rational_quadratic(r,alpha,l);
    %     l = .1;
    %     f2 = @(r) exp(-(r.^2)/(2*l^2));
    %
    %     k = @(r)f1(r)+10*f2(r);
    
    % Define prior covariance matrix
    Qr = createrow(xmin,xmax,nvec,k,theta);
    Qfun = @(x) toeplitzproduct(x, Qr, nvec);
    Q = funMat(Qfun,Qfun,nvec.^2);
    
    % Use Preconditioner
    h = 1./nx;
    P = gallery('poisson',nx)/(h.^2);% + ell.^2*speye(nx.^2);
    if nu == 0.5
        G =  chol(P,'upper');
    elseif nu == 1.5
        G = P;
    elseif nu == 2.5
        G = P^2;
    else
        error('To use preconditioner, check nu')
    end
    params.maxiter = 200;   params.tol = 1.e-6;
    
    % Generate one prior sample with preconditioner
    eps = randn(prod(nvec),1);
    [x12,relres] = krylov_sqrt(Q, G, eps, params.maxiter, params.tol);
    x_true = x12 + abs(min(x12(:))); % positive
    x_true = x_true/max(x_true(:)); % normalize
    
    x_true = reshape(x_true,nx,nx);
    
else
    %% Covariance kernel for spatial prior
    xmin = [0 0];           % Coordinates of left corner
    xmax = [1 1];           % Coordinates of right corner
    nvec = [nx, nx];         % Number of points
    theta = [1, 1];
    sigma2 = 1;
    %     nu = .5; ell = .25;
    k = @(r) sigma2*matern(r,nu,ell);
    
    Qr = createrow(xmin,xmax,nvec,k,theta);
    Qfun = @(x) toeplitzproduct(x, Qr, nvec);
    Q1 = funMat(Qfun,Qfun,nvec.^2);
    
    % Temporal prior
    %     nu_t = 2.5; ell_t = .1;
    k = @(r) matern(r,nu_t,ell_t);
    Qt = fullmatrix(0, 1, nt, k, 1);
    figure, subplot(1,2,1), imagesc(reshape(Qr,nvec)), title('Spatial kernel')
    subplot(1,2,2), plot(Qt(:,floor(nt/2))), title('Temporal kernel')
    
    Q = kronMat(Qt,Q1);
    
    % Get preconditioner
    h = 1./nx;
    P = gallery('poisson',nx)/(h.^2);% + ell.^2*speye(nx.^2);
    if nu == 0.5
        G1 =  chol(P,'upper');
    elseif nu == 1.5
        G1 = P;
    elseif nu == 2.5
        G1 = P^2;
    else
        error('To use preconditioner, check nu')
    end
    Gt = chol(inv(Qt),'upper');
    G = kronMat(Gt,G1);
    
    % Generate one prior sample with preconditioner
    eps = randn(prod(nvec)*nt,1);
    params.maxiter = 200;   params.tol = 1.e-6;
    [x12,relres] = krylov_sqrt(Q, G, eps, params.maxiter, params.tol);
    x_true = x12 + abs(min(x12(:))); % positive
    x_true = x_true/max(x_true(:)); % normalize
    
    x_true = reshape(x_true,nx,nx,nt);
end

