%% Problem setup

data_path = './data/';
load(fullfile(data_path, 'weighting_matrix.mat'));
load(fullfile(data_path, 'mea_mask_array.mat'));

recon_grd = [55, 55, 15];
mask = mea_mask_array;

%% Reconstruction

root = './data/';
attn_path = '/predicts/';
index = 47;

for i = index
    fprintf('Processing case %04d...\n', i);

    measurements_path = [root '\measurements\' sprintf('%04d', i) '.mat'];
    measurements_data = load(measurements_path);
    fn = fieldnames(measurements_data);
    measurements = measurements_data.(fn{1});

    nlevel = 0.1;
    e_full = randn(size(measurements));
    e_full = e_full / norm(e_full(:)) * norm(measurements(:)) * nlevel;
    measurements = measurements + e_full;

    gt_path = [root '\gt\' sprintf('%04d', i) '.mat'];
    gt_data = load(gt_path);
    fn = fieldnames(gt_data);
    gt = gt_data.(fn{1});
    gt = permute(gt, [3, 2, 1]);

    pred_path = [root attn_path sprintf('%04d', i) '-pred.mat'];
    pred_data = load(pred_path);
    fn = fieldnames(pred_data);
    pred = pred_data.(fn{1});
    pred = permute(pred, [3, 2, 1]);

    if any(isnan(pred), 'all') || any(isinf(pred), 'all')
        warning('Case %04d contains NaN or Inf values and is skipped.', i);
        continue;
    end

    [laser_n, xx, yy] = size(measurements);
    pixel_n = xx * yy;

    temp_measurements = permute(measurements, [3, 2, 1]);
    measurements_tmp = zeros(pixel_n, laser_n);

    for j = 1:laser_n
        measurements_tmp(:, j) = reshape(temp_measurements(:, :, j), [], 1);
    end

    measure_array = zeros(pixel_n * laser_n, 1);
    for j = 1:laser_n
        idx = (j - 1) * pixel_n + (1:pixel_n);
        measure_array(idx) = measurements_tmp(:, j);
    end

    measure_array = measure_array(mask);
    bn = double(measure_array(:));

    thr = 1e-4;
    tau = 1e-5;
    maxit_array = 50;

    for maxit = maxit_array
        regpar = 'wgcv';
        alpha_c = 0.01;

        trunc_options.nOuter = 1;
        trunc_options.nInner = maxit;
        trunc_options.max_mm = 200;
        trunc_options.compress = 'SVD';

        input = HyBRset( ...
            'InSolv', 'Tikhonov', ...
            'x_true', gt(:), ...
            'Iter', trunc_options.nInner, ...
            'RegPar', regpar);

        trunc_mats = [];

        [x_HyBR_non, HyBR_output_non, trunc_mats] = ...
            HyBRrecycle_l1_2( ...
                weighting_Matrix, bn, recon_grd, thr, tau, alpha_c, ...
                [], input, trunc_options, trunc_mats);

        trunc_mats.Y = [];
        trunc_mats.R = [];
        trunc_mats.x = [];
        trunc_mats.W = double(pred(:) / norm(pred(:)));

        input = HyBRset( ...
            'InSolv', 'Tikhonov', ...
            'x_true', gt(:), ...
            'Iter', trunc_options.nInner, ...
            'RegPar', regpar);

        [x_HyBR, HyBR_output, trunc_mats] = ...
            HyBRrecycle_l1_2( ...
                weighting_Matrix, bn, recon_grd, thr, tau, alpha_c, ...
                [], input, trunc_options, trunc_mats);

        HyBR_E_nor_non = HyBR_output_non.E_nor;
        HyBR_E_nor = HyBR_output.E_nor;

        pred(pred < 0) = 0;

        pred_error_Enor = norm( ...
            gt(:) / max(gt(:)) - ...
            (pred(:) - min(pred(:))) / ...
            (max(pred(:)) - min(pred(:)))) ...
            / norm(gt(:) / max(gt(:)));


        HyBR_E_nor = [pred_error_Enor; HyBR_E_nor];

        fprintf('Prediction error (E_nor): %.16f\n', pred_error_Enor);
        fprintf('WB-IPM error (E_nor) at %d iterations: %.16f\n', ...
            maxit, HyBR_E_nor(end));

        x_HyBR_non = reshape(x_HyBR_non, recon_grd);
        x_HyBR = reshape(x_HyBR, recon_grd);

        Nr = 1;
        Nc = 1;
        res_fact = [1, 1, 1];

        nfig = 99;
        mua_grd = gt;
        mua_grd_temp = mua_grd;

        for j = 1:size(mua_grd, 3)
            mua_grd_temp(:, :, j) = flip(mua_grd(:, :, j)', 2);
        end

        mua_grd_temp(mua_grd_temp < 0) = 0;
        SubPlotMap(mua_grd_temp, 'GT', nfig, Nr, Nc, 1, res_fact);
        colormap('hot');

        nfig = 100;
        mua_grd = pred;
        mua_grd_temp = mua_grd;

        for j = 1:size(mua_grd, 3)
            mua_grd_temp(:, :, j) = flip(mua_grd(:, :, j)', 2);
        end

        mua_grd_temp(mua_grd_temp < 0) = 0;
        SubPlotMap(mua_grd_temp, 'Prediction', nfig, Nr, Nc, 1, res_fact);
        colormap('hot');

        nfig = 101;
        x_HyBR_non_neg = x_HyBR_non;
        x_HyBR_non_neg(x_HyBR_non_neg < 0) = 0;
        x_HyBR_non_norm = ...
            (x_HyBR_non_neg - min(x_HyBR_non_neg(:))) / ...
            (max(x_HyBR_non_neg(:)) - min(x_HyBR_non_neg(:)));

        mua_grd = x_HyBR_non_norm;
        mua_grd_temp = mua_grd;

        for j = 1:size(mua_grd, 3)
            mua_grd_temp(:, :, j) = flip(mua_grd(:, :, j)', 2);
        end

        mua_grd_temp(mua_grd_temp < 0) = 0;
        SubPlotMap(mua_grd_temp, 'HyBR(non)', nfig, Nr, Nc, 1, res_fact);
        colormap('hot');

        nfig = 102;
        x_HyBR_neg = x_HyBR;
        x_HyBR_neg(x_HyBR_neg < 0) = 0;
        x_HyBR_norm = ...
            (x_HyBR_neg - min(x_HyBR_neg(:))) / ...
            (max(x_HyBR_neg(:)) - min(x_HyBR_neg(:)));

        mua_grd = x_HyBR_norm;
        mua_grd_temp = mua_grd;

        for j = 1:size(mua_grd, 3)
            mua_grd_temp(:, :, j) = flip(mua_grd(:, :, j)', 2);
        end

        mua_grd_temp(mua_grd_temp < 0) = 0;
        SubPlotMap(mua_grd_temp, 'HyBR', nfig, Nr, Nc, 1, res_fact);
        colormap('hot');

        figure(2);
        plot(HyBR_E_nor_non(1:end-1), ...
            'Color', 'blue', 'LineWidth', 2);
        hold on;
        plot(HyBR_E_nor(1:end-1), ...
            'Color', 'red', 'LineWidth', 2);
        legend('HyBR(non)', 'HyBR', 'FontSize', 15);
        xlabel('Iterations', 'FontSize', 15);
        ylabel('Relative error', 'FontSize', 15);
        hold off;
    end
end
