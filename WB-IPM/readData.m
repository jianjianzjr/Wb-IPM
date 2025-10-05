%% load raw data
testSequence = '.\testData6';
% rawDataPath = [testSequence,'\RawData_undistortion\'];
rawDataPath = [testSequence,'\RawData_no_filter\'];
x_range = [0.2,0.8];
y_range = [0.2,0.8];
nx = 10;
ny = 10;
[det_x,det_z] = meshgrid(linspace(x_range(1),x_range(2),nx),linspace(y_range(1),y_range(2),ny));
scan_points_xy = [reshape(det_x,[],1) reshape(det_z,[],1)];
scan_points_xy_1 = [scan_points_xy,ones(size(scan_points_xy,1),1)];
% load([testSequence,'\T_p2c_undistortion.mat']);
load([testSequence,'\T_p2c.mat']);
scan_points_camera_undistortion = scan_points_xy_1* T_p2c;
% load([testSequence,'\cameraParams.mat']);
width = 2048;
height = 2048;
count = size(scan_points_xy,1);
raw = zeros(width, height, count);
% ambient_img = imread([testSequence,'\ambientlight_undistortion.png']);
ambient_img = imread([testSequence,'\ambientlight_no_filter.png']);
tic
for i=1:count
% imgName = [rawDataPath,num2str(scan_points_xy(i,1)), ',', num2str(scan_points_xy(i,2)),'_undistortion.png'];
imgName = [rawDataPath,num2str(scan_points_xy(i,1)), ',', num2str(scan_points_xy(i,2)),'.png'];
tmp = imread(imgName);
% [tmp_undst,newOrigin] = undistortImage(tmp,cameraParams);
tmp_undst = tmp;
raw(:,:,i) = tmp_undst - ambient_img;
end
toc
%% extract measurement
x_range = [0.2,0.8];
y_range = [0.2,0.8];
nx = 10;
ny = 10;
[mea_x,mea_y] = meshgrid(linspace(x_range(1),x_range(2),nx),linspace(y_range(1),y_range(2),ny));
mea_xy = [reshape(mea_x,[],1) reshape(mea_y,[],1)];
mea_xy_1 = [mea_xy,ones(size(mea_xy,1),1)];
load([testSequence,'\T_p2c.mat']);
mea_camera_undistortion = mea_xy_1* T_p2c;
% figure;imshow(raw(:,:,1),[]);hold on
% plot(mea_camera_undistortion(:,1),mea_camera_undistortion(:,2),'ro')
det_num = nx*ny;
raw_mea = zeros(det_num, count);
for i=1:count
raw(1:20,:,i) = 0;
tmp = raw(:,:,i);
for j=1:det_num
x = round(mea_camera_undistortion(j,1));
y = round(mea_camera_undistortion(j,2));
tmp2 = tmp(y-4:y+4, x-4:x+4);
raw_mea(j,i) = mean(tmp2(:));
end 
end
% figure;
% for i=1:count
% imshow(raw(:,:,i),[]);
% pause(0.5)
% end
% save([testSequence, 