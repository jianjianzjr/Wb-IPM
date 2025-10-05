function [source,receiver,H] = raytracing(nx,ny,ns, nr)
     grid3D.nx = nx; Dx = 10; grid3D.hx = Dx/grid3D.nx;
     grid3D.ny = ny; Dy = 10; grid3D.hy = Dy/grid3D.ny;
     grid3D.nz = 1;  Dz = 1;  grid3D.hz = Dz/grid3D.nz;
     grid3D.minBound = [0, 0, 0]';%default
     grid3D.maxBound = [10,10,1]';

     grid3D.G = zeros(grid3D.ny,grid3D.nx,grid3D.nz);

     xs = zeros(1,ns);
     ys = linspace(0.5,9.5,ns);
     zs = zeros(1,ns);
     xr = 10*ones(1,nr);
     yr = linspace(0.5,9.5,nr);
     zr = zeros(1,nr);

     source = [xs;ys;zs]; %[zeros(1,4);3*ones(1,4);0:2:6];
     receiver = [xr;yr;zr];
     nSource = size(source,2);
     nReceiver = size(receiver,2);
     m = grid3D.nx * grid3D.ny * grid3D.nz;
     n = nSource * nReceiver;
    % 
     H = zeros(n,m);
     verbose = 0;
     k = 1;
     tic
     for i = 1:nSource
         for j = 1:nReceiver
            G=amanatidesWooAlgorithm(source(:,i), ...
                    receiver(:,j), grid3D, verbose);
            %hold on
            H(k,:) = reshape(G,1,m);
            k=k+1;
         end
     end
     hold off
     toc
end