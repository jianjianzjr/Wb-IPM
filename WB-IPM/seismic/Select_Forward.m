function [A,bn,R,nlevel] = Select_Forward(prob,n,x_true,nlevel)

switch prob
    case 1
        % Sheprical tomography from IRtool
        n_t = 20;
        angles = linspace(0,360,n_t+1);
%         angles = linspace(0,90,n/2+1);
        angles = angles(2:end);
        opt = PRset('angles',angles,'numCircles',90);
        [A,~,~,ProbInfo] = PRspherical(n,opt);
        ProbInfo
    case 2
        % raytracing
        nx = n; ny = n;
        ns =round(n/2);
        %     ns =round(n/4);
        nr = round(n/2);
        
        %     ns = 25; nr = 25;
        [~,~,A] = raytracing(nx,ny,ns, nr);
    case 3
        % PSF
        [PSF, center] = getPSF(6, n);
        A = psfMatrix(PSF,center,'reflexive');
    case 4
        % raytracing
        nx = n; ny = n;
        ns = 25; nr = 25;
        % ns = 50; nr = 50;
        [~,~,A] = raytracing(nx,ny,ns,nr);
    case 5
%         n_t = 25;
        n_t = 60;
%         maxang = 180;
        maxang = 360;
        angles = linspace(0,maxang,n_t+1);
        angles = angles(2:end);
        opt = PRset('angles',angles,'numCircles',90);
        [A,~,~,ProbInfo] = PRtomo(n,opt);
        ProbInfo
end

b = A*x_true(:);
b = b(:);

%Noise covariance
m = size(b,1);
R = speye(m,m);

%Adding noise to observation (As nlevle->0 , approx sol get closer to nonapprox sol)
% nlevel = 0.01;
[N,sigma] = WhiteNoise(b(:),nlevel);
bn = b + N;