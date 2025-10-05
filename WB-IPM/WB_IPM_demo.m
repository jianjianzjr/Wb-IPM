%%
root = 'D:\ZJR\11_initial_guess\InitialGuess_pytorch-master\data\simulation_FMT\test_545414_circle';
attn_path = [root '\5_mse_em_noattn_0_1\100\'];
close all;
% attn_nonws_path = ['\9_Dnorm_0\100\'];

thr = 1e-4;
tau = 1e-2;
maxit = 30;
recon_grd = [55,55,15];
[m,n] = size(weighting_Matrix);
mask = mea_mask_array;
trunc_options.nOuter = 1; % number of outer iterations
trunc_options.nInner = maxit; % maximum storage of solution vector space
trunc_options.max_mm = 200; % maminimum number of vectors to save at compression
trunc_options.compress = 'SVD'; 

index = 23;
%

for i = index
    %
    gt_path = [root '/gt/' sprintf('%04d', i),'.mat'];
    gt = load(gt_path);
    fn = fieldnames(gt);
    gt = gt.(fn{1});
    gt = permute(gt,[3,2,1]);
    
    pred_path = [attn_path,sprintf('%04d', i),'-pred.mat'];
    pred = load(pred_path);
    fn = fieldnames(pred);
    pred = pred.(fn{1});
    attn_pred = permute(pred,[3,2,1]);

    measurements_path = [root '/measurements/' sprintf('%04d', i) '.mat'];
    measurements = load(measurements_path);
    fn = fieldnames(measurements);
    measurements = measurements.(fn{1});
    
    [laser_n,xx,yy] = size(measurements);
    pixel_n = xx*yy;
    measure_array = zeros(pixel_n*laser_n,1);
    temp_measurements = permute(measurements,[3,2,1]);
    measurements_tmp = zeros([pixel_n,laser_n]);
    for j = 1:laser_n
        measurements_tmp(:,j) = reshape(temp_measurements(:,:,j), [],1);
    end
    
    for j= 1:laser_n
        measure_array(((j-1)*pixel_n+1):(pixel_n*j))=measurements_tmp(:,j);
    end

    tmp_measure_array = measure_array(mask);
    clear measure_array
    measure_array = tmp_measure_array;
    clear tmp_measure_array
    
    b = double(measure_array);
    
    nlevel= 0.1;
    e = randn(size(b,1),1);
    e=e/norm(e)*norm(b)*nlevel;
    delta = norm(e);
    bn = b(:) + e;
    
    nfig = 1;
    Nr = 1;Nc = 1;
    res_fact = [1,1,1];
    mua_grd =gt;
    mua_grd_temp = mua_grd;
    for j = 1:size(mua_grd,3)
        mua_grd_temp(:,:,j) = flip(mua_grd(:,:,j)',2);
    end
    mua_grd_temp(mua_grd_temp<0)=0;
    SubPlotMap(mua_grd_temp,'GT',nfig,Nr,Nc,1,res_fact);
    colormap('hot')
    set(gca, 'FontSize', 16);

    
    nfig =2;
    Nr = 1;Nc = 1;
    res_fact = [1,1,1];
    mua_grd =attn_pred;
    mua_grd_temp = mua_grd;
    for j = 1:size(mua_grd,3)
        mua_grd_temp(:,:,j) = flip(mua_grd(:,:,j)',2);
    end
    mua_grd_temp(mua_grd_temp<0)=0;
    SubPlotMap(mua_grd_temp,'Attention U-Net',nfig,Nr,Nc,1,res_fact);
    colormap('hot')     
    set(gca, 'FontSize', 16);

trunc_options.nInner = maxit;
trunc_mats = [];
input = HyBRset('InSolv', 'Tikhonov', 'x_true', gt(:),'Iter', trunc_options.nInner,'RegPar','wgcv');

[x_fHybr,fHybr_info] = WB_Projection(weighting_Matrix, bn,recon_grd,thr,tau,[],input, trunc_options, trunc_mats);
x_fHybr = reshape(x_fHybr,recon_grd);
x_fHybr(x_fHybr<0)  = 0;
nfig =111;
Nr = 1;Nc = 1;
res_fact = [1,1,1];
mua_grd =x_fHybr;
mua_grd_temp = mua_grd;
for j = 1:size(mua_grd,3)
    mua_grd_temp(:,:,j) = flip(mua_grd(:,:,j)',2);
end
mua_grd_temp(mua_grd_temp<0)=0;
SubPlotMap(mua_grd_temp,'fHybr',nfig,Nr,Nc,1,res_fact);
colormap('hot')     
set(gca, 'FontSize', 16);

%% WB-IPM
trunc_mats.Y = [];
trunc_mats.R = [];
trunc_mats.x = [];
trunc_options.nInner = maxit;
trunc_mats.W = [double(attn_pred(:)/(norm(attn_pred(:))))];

input = HyBRset('InSolv', 'Tikhonov', 'x_true', gt(:),'Iter', trunc_options.nInner,'RegPar','wgcv');

[x_WBIPM, WBIPM_info] = WB_Projection(weighting_Matrix, bn,recon_grd,thr,tau,[],input, trunc_options, trunc_mats);
x_WBIPM = reshape(x_WBIPM,recon_grd);
x_WBIPM(x_WBIPM<0)=0;
     
attn_pred = (attn_pred-min(attn_pred(:)))./ (max(attn_pred(:))-min(attn_pred(:)));
pred_rela_err = norm(attn_pred(:)-gt(:))/norm(gt(:));
%  
nfig =112;
Nr = 1;Nc = 1;
res_fact = [1,1,1];
x_WBIPM = reshape(x_WBIPM,size(gt));
x_WBIPM = (x_WBIPM-min(x_WBIPM(:)))/(max(x_WBIPM(:))-min(x_WBIPM(:)));
mua_grd = x_WBIPM;
mua_grd_temp = mua_grd;
for j = 1:size(mua_grd,3)
mua_grd_temp(:,:,j) = flip(mua_grd(:,:,j)',2);
end
mua_grd_temp(mua_grd_temp<0)=0;
% mua_grd_temp = mua_grd_temp-min(mua_grd_temp(:));
SubPlotMap(mua_grd_temp,'WB-IPM',nfig,Nr,Nc,1,res_fact);
colormap('hot')
%%
figure(200)
hold on
semilogy(fHybr_info.E_nor, '-.*', 'Color', [0 0 1], 'LineWidth', 3);
semilogy([pred_rela_err;WBIPM_info.E_nor], '-.*', 'Color', [1 0 0], 'LineWidth', 3);
hold off
legend('fHybr','WB-IPM','Fontsize',18);
xlabel('Iterations','Fontsize',18);
ylabel('Relative error','Fontsize',18);
set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',1);
end