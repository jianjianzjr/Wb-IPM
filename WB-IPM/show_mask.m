% Generate input data

% MATLAB doesn't need to concatenate for this operation

% Get the shape of the array
array_shape = [100,55,55];

% Initialize masks
mask_ideal = ones(array_shape);
mask_appro = ones(array_shape);
epsilon = 2;

% Loop over each channel
epoch =215;
mask_epoch = squeeze(mask(epoch,:,:));
for i = 1:array_shape(1)
    [Y, X] = ndgrid(1:array_shape(2), 1:array_shape(3));
    
    % Calculate the distance from the center
    distance_from_center = ((X - mask_epoch(i,1)).^2) ./ mask_epoch(i,3).^2 + ...
                           ((Y - mask_epoch(i,2)).^2) ./ mask_epoch(i,4).^2;
    
    % Initialize mask slices
    mask_slice1 = ones(array_shape(2), array_shape(3));
    
    % Set values outside the circle to 0
    mask_slice1(distance_from_center > 1) = 0;
    mask_ideal(i,:,:) = mask_slice1;
    
    % Compute mask using the tanh function
    mask_appro(i,:,:) = (tanh((1 - distance_from_center) ./ epsilon) + 1) ./ 2;
    
    % Assign to the corresponding layer in mask1 and mask2
end

nfig = 203;
Nr = 1;Nc = 1;
res_fact = [1,1,1];
mua_grd = permute(mask_ideal,[3,2,1]);
mua_grd_temp = mua_grd;

for i = 1:size(mua_grd,3)
    mua_grd_temp(:,:,i) = flip(mua_grd(:,:,i)',2);
end
SubPlotMap(mua_grd_temp,'GT',nfig,Nr,Nc,1,res_fact);
colormap('hot')

nfig = 204;
Nr = 1;Nc = 1;
res_fact = [1,1,1];
mua_grd = permute(mask_appro,[3,2,1]);
mua_grd_temp = mua_grd;

for i = 1:size(mua_grd,3)
    mua_grd_temp(:,:,i) = flip(mua_grd(:,:,i)',2);
end
% mua_grd_temp(mua_grd_temp<0)=0;
% mua_grd_temp = mua_grd_temp-min(mua_grd_temp(:));
SubPlotMap(mua_grd_temp,'Predict',nfig,Nr,Nc,1,res_fact);
colormap('hot')
