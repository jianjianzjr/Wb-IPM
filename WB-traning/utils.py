import torch
import os
import h5py
import re
import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np

def load_data_phantomToSimu(data_path, device, mode):
    data = h5py.File(data_path)
    phantom = data['phantom']
    simu = data['simu']
    simu = np.expand_dims(simu, axis=2)
    phantom = np.expand_dims(phantom, axis=2)
    phantom = np.transpose(phantom, [3, 2, 0, 1])
    simu = np.transpose(simu, [3, 2, 0, 1])

    training_images_count = round(simu.shape[0]*0.9)
    if mode == 'train':
        simu_training = torch.tensor(simu[0:training_images_count]).float().to(device)
        phantom_training = torch.tensor(phantom[0:training_images_count]).float().to(device)
        simu_validation = torch.tensor(simu[training_images_count:]).float().to(device)
        phantom_validation = torch.tensor(phantom[training_images_count:]).float().to(device)
        return simu_training, phantom_training, simu_validation, phantom_validation
    elif mode == 'eval':
        return simu, phantom,
    else:
        raise ValueError('mode should be train or test')

def load_weighting_matrix(data_path, device):
    file_path = os.path.join(data_path, 'weighting_Matrix.mat')
    data = h5py.File(file_path)
    if 'weighting_Matrix' in data:
        weight_Matrix = data['weighting_Matrix']
    weight_Matrix = torch.tensor(np.array(weight_Matrix)).float().to(device)

    file_path = os.path.join(data_path, 'mea_mask_array.mat')
    data = h5py.File(file_path)
    if 'mea_mask_array' in data:
        mea_mask_array = data['mea_mask_array']
    mea_mask_array = torch.tensor(np.array(mea_mask_array)).float().to(device)
    return weight_Matrix.t(), mea_mask_array.t()

def load_data(data_path, device, mode):
    # if mode == 'train':
    gt_data_path = os.path.join(data_path, 'gt/')
    gt_files = [f for f in os.listdir(gt_data_path) if f.endswith('.mat')]
    measure_data_path = os.path.join(data_path, 'emissions/')
    measure_files = [f for f in os.listdir(measure_data_path) if f.endswith('.mat')]

    # file_path = os.path.join(data_path, 'mea_mask.mat')
    # data = h5py.File(file_path)
    # if 'mea_mask' in data:
    #     mea_mask = data['mea_mask']
    # mea_mask = torch.tensor(np.array(mea_mask)).float()

    file_path = os.path.join(data_path, 'excitation.mat')
    data = h5py.File(file_path)
    if 'excitation' in data:
        excitation = data['excitation']
    excitation = torch.tensor(np.array(excitation)).float()

    gt = []
    for file_name in gt_files:
        file_path = os.path.join(gt_data_path, file_name)
        data = h5py.File(file_path)
        if 'ground_truth' in data:
            gt.append(data['ground_truth'])

    gt = torch.tensor(np.array(gt)).float()

    measurements = []
    for file_name in measure_files:
        file_path = os.path.join(measure_data_path, file_name)
        data = h5py.File(file_path)
        if 'measurements' in data:
            measurements.append(data['measurements'])

    # measurements = torch.tensor(np.log(np.array(measurements)+1)).float()
    measurements = torch.tensor(np.array(measurements)).float()

    # # Calculate the Frobenius norm of the measurements
    # measurements_norm = torch.norm(measurements)
    #
    # # Determine the desired noise level (5% of the measurements' norm)
    # desired_noise_norm = 0.5 * measurements_norm
    #
    # # Generate random noise
    # random_noise = torch.randn(measurements.size())
    #
    # # Scale the noise to have the desired Frobenius norm
    # noise_norm = torch.norm(random_noise)
    # scaled_noise = (desired_noise_norm / noise_norm) * random_noise
    #
    # # Add the scaled noise to the measurements
    # measurements = measurements + scaled_noise


    # mea_mask_expanded = np.expand_dims(mea_mask, axis=0)
    excitation = excitation.unsqueeze(0).permute(0, 3, 2, 1).to(device)

    gt = np.transpose(gt, [0, 3, 2, 1]).to(device)
    measurements = np.transpose(measurements, [0, 3, 2, 1]).to(device)

    # applying mask
    threshold = 0.00
    mea_mask = excitation - threshold * excitation.max()
    mea_mask[mea_mask > 0] = 1
    mea_mask[mea_mask < 0] = 0
    measurements = measurements * mea_mask

    #random
    # 生成一个 0 到 1499 的随机排列的索引（因为 Python 是从 0 开始索引的）
    rand_indices = np.random.permutation(gt.shape[0])

    # 使用这个索引来重新排列每个数组的第一个维度
    gt_shuffled = gt[rand_indices, :, :, :]
    measurements_shuffled = measurements[rand_indices, :, :, :]

    if mode == 'train':
        training_images_count = round(gt.shape[0] * 0.9)
        gt_training = gt_shuffled[0:training_images_count].clone().detach().float().to(device)
        measurements_training = measurements_shuffled[0:training_images_count].clone().detach().float().to(device)
        gt_validation = gt_shuffled[training_images_count:].clone().detach().float().to(device)
        measurements_validation = measurements_shuffled[training_images_count:].clone().detach().float().to(device)
        return gt_training, measurements_training, gt_validation, measurements_validation, excitation
    elif mode == 'eval':
        # Generating a dummy tensor
        torch.manual_seed(0)  # For reproducibility

        # Adding random white noise to each [100, 55, 55] data slice
        noise_level = 0.0 # Adjust this as needed for the desired noise level

        # Generate noise tensor
        noisy_measurements = measurements.clone().to(device)
        # 为每个 [100, 55, 55] 切片添加噪声
        for i in range(measurements.shape[0]):
            norm = measurements[i].norm()
            random_gen = torch.randn(measurements.shape[1:]).to(device)
            noise = noise_level * norm * random_gen / random_gen.norm()
            noisy_measurements[i] += noise

        return gt, noisy_measurements, excitation
    else:
        raise ValueError('mode should be train or test')

    # if mode == 'train':
    #     important_indices_fn = h5py.File(os.path.join(data_path, 'important_indices.mat'))
    #     if 'important_indices' in important_indices_fn:
    #         important_indices = important_indices_fn['important_indices']
    #         important_indices = np.array(important_indices).squeeze().astype(int)-1

    # data = scipy_io.loadmat(data_path)
    # noisy = data['lab_d']
    # orig = data['lab_n']
    # noisy = np.transpose(noisy, [3, 2, 0, 1])
    # orig =
    # np.transpose(orig, [3, 2, 0, 1])
    #
    # training_images_count = round(noisy.shape[0]*0.95)
    # if mode == 'train':
    #     noisy = torch.tensor(noisy[0:training_images_count]).float().to(device)
    #     orig = torch.tensor(orig[0:training_images_count]).float().to(device)
    # elif mode == 'eval':
    #     noisy = torch.tensor(noisy[training_images_count:]).float().to(device)
    #     orig = torch.tensor(orig[training_images_count:]).float().to(device)
    # else:
    #     raise ValueError('mode should be train or test')

    # if mode == 'train':
    #     return gt, measurements, important_indices
    # else:
    #     return gt, measurements


def load_checkpoint(model, optimizer, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        raise ValueError('checkpoint dir does not exist')

    checkpoint_list = os.listdir(checkpoint_dir)
    if len(checkpoint_list) > 0:

        checkpoint_list.sort(key=lambda x: int(re.findall(r"epoch-(\d+).pkl", x)[0]))

        last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
        print('load checkpoint: %s' % last_checkpoint_path)

        # 原：model_ckpt = torch.load(last_checkpoint_path)
        model_ckpt = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)

        model.load_state_dict(model_ckpt['state_dict'])

        if optimizer:
            optimizer.load_state_dict(model_ckpt['optimizer'])

        epoch = model_ckpt['epoch']
        return model, optimizer, epoch


def cmap_convert(image_tensor):
    image = image_tensor.detach().cpu().clone().numpy().squeeze()
    # subtract the minimum and divide by the maximum, is it reasonable for medical image processing
    image = image - image.min()
    image = image / image.max()
    cmap_viridis = plt.get_cmap('viridis')
    image = cmap_viridis(image)
    image = Image.fromarray((image * 255).astype(np.uint8)).convert('L')
    return image


def rsnr(rec, oracle):
    "regressed SNR"
    sumP = sum(oracle.reshape(-1))
    sumI = sum(rec.reshape(-1))
    sumIP = sum(oracle.reshape(-1) * rec.reshape(-1) )
    sumI2 = sum(rec.reshape(-1)**2)
    A = np.matrix([[sumI2, sumI], [sumI, oracle.size]])
    b = np.matrix([[sumIP], [sumP]])
    c = np.linalg.inv(A)*b #(A)\b
    rec = c[0, 0]*rec+c[1, 0]
    err = sum((oracle.reshape(-1)-rec.reshape(-1))**2)
    SNR = 10.0*np.log10(sum(oracle.reshape(-1)**2)/err)

    if np.isnan(SNR):
        SNR = 0.0
    return SNR
