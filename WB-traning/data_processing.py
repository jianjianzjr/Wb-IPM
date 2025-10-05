import os
import matplotlib.pyplot as plt
import numpy as np
# 定义文件夹路径
# folder_path = './data/simulation_FMT/test/gt/'  # 替换为你的文件夹路径
#
# #遍历文件夹中的文件
# for filename in os.listdir(folder_path):
#     # 检查文件是否是文件而不是文件夹
#     if os.path.isfile(os.path.join(folder_path, filename)):
#         # 获取文件的扩展名
#         file_extension = os.path.splitext(filename)[1]
#
#         # 提取文件名（不包含扩展名）
#         file_name_without_extension = os.path.splitext(filename)[0]
#
#         # 检查文件名的长度，如果不足4位，前面补零
#         if len(file_name_without_extension) < 4:
#             new_file_name = file_name_without_extension.zfill(4) + file_extension
#             new_filepath = os.path.join(folder_path, new_file_name)
#
#             # 重命名文件
#             os.rename(os.path.join(folder_path, filename), new_filepath)
#
#             print(f'Renamed: {filename} to {new_file_name}')


##################################################################################################

log_dir = './log_545414_rec4cir1/6_mse_em_thres_0.05_le_f2/'
log_dir0 = './log_545414_rec4cir1/6_mse_em_thres_0.05_le_f2/log/'
mse_training_path0 = os.path.join(log_dir0,
                                     'mse_training.npy')
mse_training_epoch0 = np.load(mse_training_path0)
mse_training_path = os.path.join(log_dir,
                                     'mse_training.npy')
mse_training_epoch = np.load(mse_training_path)
mse_training_epoch[0:600] = mse_training_epoch0

mse_validation_path0 = os.path.join(log_dir0,
                                     'mse_validation.npy')
mse_validation_epoch0 = np.load(mse_validation_path0)
mse_validation_path = os.path.join(log_dir,
                                     'mse_validation.npy')
mse_validation_epoch = np.load(mse_validation_path)
mse_validation_epoch[0:600] = mse_validation_epoch0

mse_test_path0 = os.path.join(log_dir0,
                                     'mse_test.npy')
mse_test_epoch0 = np.load(mse_test_path0)
mse_test_path = os.path.join(log_dir,
                                     'mse_test.npy')
mse_test_epoch = np.load(mse_test_path)
mse_test_epoch[0:600] = mse_test_epoch0

threshold_path0 = os.path.join(log_dir0,
                                     'threshold.npy')
threshold_epoch0 = np.load(threshold_path0)
threshold_path = os.path.join(log_dir,
                                     'threshold.npy')
threshold_epoch = np.load(threshold_path)
threshold_epoch[0:600] = threshold_epoch0

plt.plot(np.log(mse_training_epoch), label='Training')
# plt.plot(np.log(loss_training_epoch), label='Loss Training')
plt.plot(np.log(mse_validation_epoch), label='Validation')
plt.plot(np.log(mse_test_epoch), label='Test')
plt.xlabel('Epoch')
plt.ylabel('Log(MSE)')
plt.legend()
# plt.title('曲线图')
plt.show()

plt.plot(threshold_epoch, label='Threshold')
plt.xlabel('Epoch')
plt.ylabel('Threshold')
plt.legend()
plt.show()
#
# mse_validation_path = os.path.join(log_dir,
#                                    'mse_validation.npy')
# mse_validation_epoch_2 = np.load(mse_validation_path)
# mse_validation_path = os.path.join(log_dir,
#                                    'mse_validation_500.npy')
# mse_validation_epoch_1 = np.load(mse_validation_path)
# mse_validation_epoch_2[0:500] = mse_validation_epoch_1
# mse_validation_epoch = mse_validation_epoch_2
# np.save(mse_validation_path, 'mse_validation_epoch')
#
# mse_training_path = os.path.join(log_dir,
#                                      'threshold.npy')
# mse_training_epoch_2 = np.load(mse_training_path)
# #
# # # plot mse curve with epoches
# #
# # # 绘制曲线
# # plt.plot(mse_training_epoch, label='Training')
# plt.plot(mse_training_epoch_2)
# plt.xlabel('Epoch')
# plt.ylabel('Threshold')
# plt.legend()
# # plt.title('曲线图')
# plt.show()


# mse_training_path = os.path.join(log_dir,
#                                      'mse_training.npy')
# mse_training_epoch = np.load(mse_training_path)
# loss_training_path = os.path.join(log_dir,
#                                  'loss_training.npy')
# loss_training_epoch = np.load(loss_training_path)
# mse_validation_path = os.path.join(log_dir,
#                                    'mse_validation.npy')
# mse_validation_epoch = np.load(mse_validation_path)
#
# # plot mse curve with epoches
#
# # 绘制曲线
#
# plt.plot(np.log(mse_training_epoch), label='MSE Training')
# # plt.plot(np.log(loss_training_epoch), label='Loss Training')
# plt.plot(np.log(mse_validation_epoch), label='MSE Validation')
# plt.xlabel('Epoch')
# plt.ylabel('Log(MSE)')
# plt.legend()
# # plt.title('曲线图')
# plt.show()


# ###### change npy data to the format mat
# import numpy as np
# from scipy.io import savemat
# log_dir =
# thres_training_path = os.path.join(log_dir,
#                                       'threshold.npy')
# # Load your .npy file
# npy_data = np.load(thres_training_path)
#
# # Convert to a dictionary format that savemat expects
# # The key 'data' is the name of the variable you will see in MATLAB
# mat_data = {'mask': npy_data}
#
# # Save to a .mat file
# savemat('mask.mat', mat_data)
