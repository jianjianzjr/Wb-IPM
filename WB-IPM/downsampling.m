function output =  downsampling(x,eta)

    % Assume 'x' is your original 3D matrix of size 55x55x15
    grd = size(x); % Example data
    recon_grd = ceil(grd/eta);
    % Initialize the downsized matrix
    downsizedX = zeros(recon_grd);

    % Downsample each 2D slice
    for i = 1:grd(3)
        downsizedX(:, :, i) = imresize(x(:, :, i), [recon_grd(1) recon_grd(2)]);
    end

    % Now interpolate along the third dimension to get from 15 to 10 slices
    % Prepare a grid for original and target slice indices
   [Yq, Xq, Zq] = ndgrid(1:size(downsizedX, 1), 1:size(downsizedX, 2), linspace(1, size(downsizedX, 3),  recon_grd(3)));


    % Use interp3 for downsizing along the third dimension
    output = interp3(downsizedX, Xq, Yq, Zq, 'cubic');

    % 'finalX' is now the downsampled 3D matrix of size 37x37x10
end