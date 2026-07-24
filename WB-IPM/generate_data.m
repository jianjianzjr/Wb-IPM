 
output_root = '.\data\';
 
generate_FMT_data(output_root, 1, 0.10);
 
function generate_FMT_data(output_root, dataN, noise_level)
%GENERATE_FMT_DATA Generate paired 3D FMT data using Toast++.
%
%   generate_FMT_data(output_root, dataN, noise_level)
%
% The function:
%   1. builds a 54 x 54 x 14 mm slab mesh;
%   2. places a 10 x 10 source grid on the bottom surface;
%   3. places a 55 x 55 detector grid on the top surface;
%   4. generates 1--3 random rectangular or ellipsoidal inclusions;
%   5. solves the excitation and emission diffusion equations;
%   6. forms measurements = det_Em ./ det_Ex;
%   7. adds Gaussian noise to the complete unmasked measurement tensor;
%   8. saves paired ground-truth and measurement files.
%
% Required toolboxes:c
%   Toast++ MATLAB toolbox
%   iso2mesh (for meshabox)
%
% Example:
%   output_root = ...
%       './data/';
%   generate_FMT_data(output_root, 500, 0.10);
 
    if nargin < 1 || isempty(output_root)
        output_root = fullfile(pwd, 'FMT_dataset');
    end
 
    if nargin < 2 || isempty(dataN)
        dataN = 500;
    end
 
    if nargin < 3 || isempty(noise_level)
        noise_level = 0.10;
    end
 
    if exist('meshabox', 'file') ~= 2
        error('iso2mesh is required: meshabox was not found.');
    end
 
    if exist('toastMesh', 'file') ~= 2 && ...
            exist('toastMesh', 'class') ~= 8
        error('Toast++ MATLAB toolbox is not on the MATLAB path.');
    end
 
    rng(1);
 
    %% Output folders
 
    gt_dir = fullfile(output_root, 'gt');
    measurements_dir = fullfile(output_root, 'measurements');
 
    if ~exist(gt_dir, 'dir')
        mkdir(gt_dir);
    end
 
    if ~exist(measurements_dir, 'dir')
        mkdir(measurements_dir);
    end
 
    %% Slab geometry and reconstruction grid
 
    phantom_dim = [54, 54, 14];
    recon_grd = phantom_dim + 1;
 
    start_point = [0, 0, 0];
    end_point = phantom_dim;
    max_element_volume = 0.5;
 
    [node, ~, element] = meshabox( ...
        start_point, end_point, max_element_volume, 1);
 
    element = element(:, 1:4);
    eltp = ones(size(element, 1), 1) * 3;
 
    toast_mesh = toastMesh(node, element, eltp);
    grid_mesh_basis = toastBasis(toast_mesh, recon_grd);
 
    %% Source and detector positions
 
    source_number = [10, 10];
    source_edge = round(phantom_dim(1) / 3);
 
    source_x = linspace( ...
        source_edge, phantom_dim(1) - source_edge, source_number(1));
 
    source_y = linspace( ...
        source_edge, phantom_dim(2) - source_edge, source_number(2));
 
    [source_x_grid, source_y_grid] = ndgrid(source_x, source_y);
 
    source_position = [ ...
        source_x_grid(:), ...
        source_y_grid(:), ...
        zeros(numel(source_x_grid), 1)];
 
    detector_number = [55, 55];
    detector_x = linspace(0, phantom_dim(1), detector_number(1));
    detector_y = linspace(0, phantom_dim(2), detector_number(2));
 
    [detector_x_grid, detector_y_grid] = ndgrid( ...
        detector_x, detector_y);
 
    detector_position = [ ...
        detector_x_grid(:), ...
        detector_y_grid(:), ...
        phantom_dim(3) * ones(numel(detector_x_grid), 1)];
 
    toast_mesh.SetQM(source_position, detector_position);
 
    %% Optical properties and FEM system matrix
 
    n_node = size(node, 1);
 
    mua = 0.0055 * ones(n_node, 1);
    mus = 0.97 * ones(n_node, 1);
    refractive_index = 1.4 * ones(n_node, 1);
 
    laser_power = 10;
    laser_width = 1;
    laser_frequency = 0;
 
    detector_width = 1;
 
    qvec = laser_power * toast_mesh.Qvec( ...
        'Neumann', 'Gaussian', laser_width);
 
    mvec = toast_mesh.Mvec( ...
        'Gaussian', detector_width, refractive_index);
 
    system_K = dotSysmat( ...
        toast_mesh, mua, mus, refractive_index, laser_frequency);
 
    system_K = real(system_K);
 
    %% Excitation forward solution
 
    phi_Ex = system_K \ qvec;
    det_Ex = mvec.' * phi_Ex;
 
    n_laser = size(qvec, 2);
 
    if n_laser ~= prod(source_number)
        error('The number of Toast++ source vectors is inconsistent.');
    end
 
    if size(det_Ex, 1) ~= prod(detector_number)
        error('The number of Toast++ detector vectors is inconsistent.');
    end
 
    %% Ground-truth grid
 
    [x_grid, y_grid, z_grid] = ndgrid( ...
        0:phantom_dim(1), ...
        0:phantom_dim(2), ...
        0:phantom_dim(3));
 
    fluorophore_concentration = 100;
 
    min_xy_radius = 3;
    max_xy_radius = 7;
    min_z_radius = 1;
    max_z_radius = 3;
 
    %% Generate paired data
 
    for sample_id = 1:dataN
        ground_truth_xyz = zeros(recon_grd);
        n_inclusions = randi([1, 3]);
 
        for inclusion_id = 1:n_inclusions
            radius_x = randi([min_xy_radius, max_xy_radius]);
            radius_y = randi([min_xy_radius, max_xy_radius]);
            radius_z = randi([min_z_radius, max_z_radius]);
 
            center_x = randi([ ...
                radius_x + 1, phantom_dim(1) - radius_x - 1]);
 
            center_y = randi([ ...
                radius_y + 1, phantom_dim(2) - radius_y - 1]);
 
            center_z = randi([ ...
                radius_z + 1, phantom_dim(3) - radius_z - 1]);
 
            if rand < 0.5
                inclusion = ...
                    ((x_grid - center_x) / radius_x).^2 + ...
                    ((y_grid - center_y) / radius_y).^2 + ...
                    ((z_grid - center_z) / radius_z).^2 <= 1;
            else
                inclusion = ...
                    abs(x_grid - center_x) <= radius_x & ...
                    abs(y_grid - center_y) <= radius_y & ...
                    abs(z_grid - center_z) <= radius_z;
            end
 
            ground_truth_xyz(inclusion) = ...
                fluorophore_concentration;
        end
 
        %% Emission forward solution
 
        fluoDis_m = grid_mesh_basis.Map( ...
            'B->M', ground_truth_xyz(:));
 
        Q_fluo = fluoDis_m * ones(1, n_laser);
        Em_qvec = Q_fluo .* phi_Ex;
 
        phi_Em = system_K \ Em_qvec;
        det_Em = mvec.' * phi_Em;
 
        %% Excitation-normalized measurements
 
        measurements_array = det_Em ./ det_Ex;
 
        measurements_tmp = zeros( ...
            detector_number(1), detector_number(2), n_laser);
 
        for source_id = 1:n_laser
            measurements_tmp(:, :, source_id) = reshape( ...
                measurements_array(:, source_id), detector_number);
        end
 
        measurements = permute(measurements_tmp, [3, 2, 1]);
 
        %% Add noise to the complete, unmasked measurements
 
        if noise_level > 0
            measurement_noise = randn(size(measurements));
 
            measurement_noise = measurement_noise ...
                / norm(measurement_noise(:)) ...
                * norm(measurements(:)) ...
                * noise_level;
 
            measurements = measurements + measurement_noise;
        end
 
        %% Save data in the orientation used by main_demo.m
 
        ground_truth = permute(ground_truth_xyz, [3, 2, 1]);
 
        save( ...
            fullfile(gt_dir, [sprintf('%04d', sample_id) '.mat']), ...
            'ground_truth', ...
            '-v7.3');
 
        save( ...
            fullfile(measurements_dir, ...
            [sprintf('%04d', sample_id) '.mat']), ...
            'measurements', ...
            '-v7.3');
 
        fprintf('Generated sample %04d of %04d.\n', ...
            sample_id, dataN);
    end
end

