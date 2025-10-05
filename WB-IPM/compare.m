load('D:\1 code\STIFT_v1\pro\1_data\det_Em.mat')
load('D:\1 code\STIFT_v1\pro\1_data\det_Em_simu.mat')

grd = [61,31];
Em = reshape(det_Em(:,1),grd);
Em_simu = reshape(det_Em_simu(:,1),grd);
for i = 2:size(det_Em,2)
    temp = reshape(det_Em(:,i),grd);
    Em = [Em;temp];
    temp_simu = reshape(det_Em_simu(:,i),grd);
    Em_simu = [Em_simu;temp_simu];
end
% figure
% for i = 1:size(Em,2)
%     temp = Em(:,i)/max(Em(:,i));
%     temp_simu = Em_simu(:,i)/max(Em_simu(:,i));   
%     subplot(size(Em,2),1,i);
%     plot(temp);
%     hold on 
%     plot(temp_simu);
%     hold off
%     legend('real data','simulation');
% end

temp = Em(:,ceil(end/2))/max(Em(:,ceil(end/2)));
temp_simu = Em_simu(:,ceil(end/2))/max(Em_simu(:,ceil(end/2)));
figure
plot(temp);
hold on 
plot(temp_simu);
hold off
legend('real data','simulation');

%% mesh


    