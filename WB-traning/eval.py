from model_attention import InitialCONVNet
from utils import load_checkpoint, load_data, cmap_convert, rsnr
import os
import torch
import math
import torchvision
import argparse
import numpy as np
import scipy.io as sio

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def eval(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initial_conv_net = UNet(in_channel=100, out_channel=15,
    #                         inner_channel=128,
    #                         attn_res=(14, 56), image_size=55).to(device)
    initial_conv_net = InitialCONVNet().to(device)

    if not (os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0):
        print('load checkpoint unsuccessfully')
        return

    initial_conv_net, _, _ = load_checkpoint(initial_conv_net, optimizer=None, checkpoint_dir=config.checkpoint_dir)
    initial_conv_net.eval()

    # print number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, initial_conv_net.parameters())
    num_params = sum([param.numel() for param in model_parameters])
    print(f"Number of model parameters: {num_params}")

    print('load testing data')
    gt, measurements, excitation = load_data(config.data_path, device, mode='eval')

    if not os.path.exists(config.eval_result_dir):
        os.mkdir(config.eval_result_dir)
    # store mse
    mse_loss = np.zeros(measurements.shape[0])

    for i in range(math.ceil(measurements.shape[0]/config.batch_size)):
        i_start = i
        i_end = min(i+config.batch_size, measurements.shape[0])
        measurements_batch = measurements[i_start:i_end]
        gt_batch = gt[i_start:i_end]

        y_pred = initial_conv_net(measurements_batch, excitation)
        for j in range(y_pred.shape[0]): # batch_size
            measurements_image_path = os.path.join(config.eval_result_dir, '{:04d}-measurements.mat'.format(i*config.batch_size+j+1))
            pred_image_path = os.path.join(config.eval_result_dir, '{:04d}-pred.mat'.format(i*config.batch_size+j+1))
            gt_image_path = os.path.join(config.eval_result_dir, '{:04d}-gt.mat'.format(i*config.batch_size+j+1))

            measurements_image = measurements_batch[j].detach().cpu().numpy()
            sio.savemat(measurements_image_path, {'measurements': measurements_image})
            print('save image:', measurements_image_path)

            pred_image = y_pred[j].detach().cpu().numpy()
            sio.savemat(pred_image_path, {'predict': pred_image})
            print('save image:', pred_image_path)

            gt_image = gt_batch[j].detach().cpu().numpy()
            sio.savemat(gt_image_path, {'gt': gt_image})
            print('save image:', gt_image_path)

            mse_loss[i * config.batch_size + j] = torch.nn.MSELoss()(y_pred[j], gt_batch[j]).item()
            print('mse:', mse_loss[i * config.batch_size + j])

            # SNR = rsnr(np.array(pred_image), np.array(gt_image))
            # print('%d-pred.jpg SNR:%f' % (i * config.batch_size + j + 1, SNR))

            # if config.cmap_convert:
            #     measurements_image = cmap_convert(measurements_batch[j].squeeze())
            #     measurements_image.save(measurements_image_path)
            #     print('save image:', measurements_image_path)
            #
            #     pred_image = cmap_convert(y_pred[j].squeeze())
            #     pred_image.save(pred_image_path)
            #     print('save image:', pred_image_path)
            #
            #     gt_image = cmap_convert(gt_batch[j].squeeze())
            #     gt_image.save(gt_image_path)
            #     print('save image:', gt_image_path)
            #
            #     SNR = rsnr(np.array(pred_image), np.array(gt_image))
            #     print('%d-pred.jpg SNR:%f' % (i * config.batch_size + j + 1, SNR))
            #
            # else:
            #     torchvision.utils.save_image(measurements_batch[j].squeeze(), measurements_image_path)
            #     print('save image:', measurements_image_path)
            #     torchvision.utils.save_image(y_pred[j].squeeze(), pred_image_path)
            #     print('save image:', pred_image_path)
            #     torchvision.utils.save_image(gt_batch[j].squeeze(), gt_image_path)
            #     print('save image:', gt_image_path)
            #
            #     pred_image = y_pred[j].clone().detach().cpu().squeeze()
            #     gt_image = gt_batch[j].clone().detach().cpu().squeeze()
            #     SNR = rsnr(np.array(pred_image), np.array(gt_image))
            #     print('%d-pred.jpg SNR:%f' % (i * config.batch_size + j + 1, SNR))

    print('Mean of MSE:', mse_loss.mean())
    mse_path = os.path.join(config.eval_result_dir, 'mse_loss.mat')
    sio.savemat(mse_path, {'gt': mse_loss})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='./log_545414_rec4cir1/9_rDnorm_angle_0/checkpoint/40/')
    parser.add_argument('--data_path', type=str, default='./data/simulation_FMT/13_phantom_simu/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_result_dir', type=str, default='./data/simulation_FMT/13_phantom_simu/9_rDnorm_angle_0/40/')
    # parser.add_argument('--cmap_convert', type=bool, default=True)
    config = parser.parse_args()
    print(config)
    eval(config)
