function k = rational_quadratic(r,alpha,l)
    %Implementation of the Matern covariance kernel
    % r is the radial distance
    % nu is a parameter that controls smoothness of the stochastic process
    % ell controls the correlation length

    %scale = sqrt(2*nu)*r/ell;
    fact  = 1 + r.^2/(2*alpha*l^2);
    
    k = fact.^(-alpha);
    
%     %Fix the scaling issue at r = 0; K_nu(0) numerically evalates to inf
%     k(find(isnan(k))) = 1;
    
%     %For nu = inf, the square exponential covariance kernel
%     if alpha == inf
%         k = exp(-((r/ell).^2)/2);
%     end
        
end