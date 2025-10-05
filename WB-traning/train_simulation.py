from model_attention import InitialCONVNet
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import argparse
import random
import os
from datetime import datetime
from torchviz import make_dot
import scipy.io as sio
from utils import load_data, load_checkpoint, cmap_convert, load_weighting_matrix
from torchsummary import summary
import matplotlib.pyplot as plt

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def data_argument(measurements, gt):

    # flip horizontal
    for i in range(measurements.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            measurements[i] = measurements[i].flip(2)
            gt[i] = gt[i].flip(2)

    # flip vertical
    for i in range(measurements.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            measurements[i] = measurements[i].flip(1)
            gt[i] = gt[i].flip(1)
    return measurements, gt

class L2RegularizedMSELoss(torch.nn.Module):
    def __init__(self, l2_lambda=1e-5):
        super(L2RegularizedMSELoss, self).__init__()
        self.l2_lambda = l2_lambda

    def forward(self, y_pred, y_true, excitation, model):
        # y_pred_normalized = y_pred / torch.max(y_pred.reshape(-1))
        # y_true_normalized = y_true / torch.max(y_true.reshape(-1))

        mse_loss = torch.nn.MSELoss()(y_pred, y_true)
        # mse_loss = torch.nn.MSELoss()(y_pred/np.max(y_pred.reshape(-1)), y_true/np.max(y_true.reshape(-1)))  # 计算均方误差损失
        l2_reg = torch.tensor(0.0).float().to(mse_loss.device)  # 初始化L2正则项

        # 计算L2正则项
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)

        # 添加L2正则项到损失中
        loss = mse_loss + self.l2_lambda * l2_reg

        return loss, mse_loss


class AngleLoss(nn.Module):
    def __init__(self, l2_lambda: float = 0.0):
        super().__init__()
        self.l2_lambda = float(l2_lambda)

    def forward(self, y_pred, y_true, excitation=None, model=None, weight_Matrix=None, mea_mask_array=None,measurements_batch=None):
        # Flatten per-sample
        B = y_pred.shape[0]
        y_pred_flat = y_pred.contiguous().view(B, -1)  # (B, N)
        y_true_flat = y_true.contiguous().view(B, -1)  # (B, N)

        # Sign-sensitive angle loss: pushes cos -> 1
        cos = F.cosine_similarity(y_pred_flat, y_true_flat, dim=1, eps=1e-12)
        angle_loss = (1.0 - cos).mean()

        # Optional squared-L2 regularization on model params
        reg = y_pred.new_tensor(0.0)
        if model is not None and self.l2_lambda > 0.0:
            reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
            reg = self.l2_lambda * reg

        loss = angle_loss + reg
        return loss, angle_loss

class AngleLoss_reg(nn.Module):
    """
    total = angle_loss + resid_lambda * residual_loss
      - angle_loss: 1 - cos(y_pred, y_true)  (方向对齐，角度越小越好)
      - residual_loss: 只在 mask==1 的测点上计算 ||A @ y_pred - b||
    """

    def __init__(self, resid_lambda: float = 0.0, squared: bool = True, normalize: bool = True,
                 thr: float = 2e-3, tau: float = 1e-3, eps: float = 1e-12):
        super().__init__()
        self.resid_lambda = float(resid_lambda)
        self.squared = bool(squared)
        self.normalize = bool(normalize)
        self.thr = float(thr)
        self.tau = float(tau)
        self.eps = float(eps)


    def forward(self, y_pred, y_true, A, b, mask=None, eps: float = 1e-12):
        device = y_pred.device
        B = y_pred.shape[0]

        yp = y_pred.contiguous().view(B, -1)                   # (B, n)
        yt = y_true.contiguous().view(B, -1)                   # (B, n)
        b = b.contiguous().view(B, -1) #(B,m)
        # n = yp.shape[1]

        # ----- Build D from y_true: L = dxj^{-1/2}, where dxj = 2*sqrt(yt^2 + eps)
        dxj = 2.0 * torch.sqrt(yt.pow(2) + self.tau)  # (B, n), strictly > 0
        if self.thr > 0.0:
            dxj = torch.where(dxj < self.thr, torch.as_tensor(self.tau, device=device), dxj)
        # L = dxj^{-1/2}
        L = torch.pow(dxj, -0.5)  # (B, n)

        # ----- D-angle: cosine on (L * x)
        yp_w = yp * L
        yt_w = yt * L

        # ---- angle loss term：1 - cos
        cos = F.cosine_similarity(yp_w, yt_w, dim=1, eps=self.eps)
        angle_loss = (1.0 - cos).mean()

        # ---- forward：Ax = A @ yp^T  (对每个样本)
        if not torch.is_tensor(A):
            raise TypeError("A must be a torch.Tensor with shape (m, n).")
        if A.dim() != 2:
            raise ValueError(f"A must have shape (m, n), got {A.shape}.")
        Ax = yp @ A.t()                           # (B, m)

        # ---- 处理 b 形状
        # b = torch.as_tensor(b, device=device)
        # if b.dim() == 1:                          # (m,) -> (B, m)
        #     b = b.unsqueeze(0).expand(B, -1)
        # elif b.dim() == 2 and b.shape[0] != B:
        #     raise ValueError(f"b must be (B, m) or (m,), got {b.shape} with B={B}.")

        # ---- 只对 b 做抽取（等价 MATLAB: b = b(mask)）
        mask = torch.as_tensor(mask, device=device)
        mask = (mask != 0).view(-1)  # -> (m,) bool
        idx = torch.where(mask)[0].long()  # -> (m_sel,)
        b_sel = b.index_select(1, idx)

        # if mask is not None:
        #     mask = torch.as_tensor(mask, device=device).bool()   # (m,)
        #     if mask.dim() != 1 or mask.shape[0] != Ax.shape[1]:
        #         raise ValueError(f"mask must be shape (m,), got {mask.shape}, m={Ax.shape[1]}")
        #     # MATLAB 等价：tmp_b = b(mask); b = tmp_b;
        #     b_sel  = b[:, mask]               # (B, m_sel)
        #     # 为了与 b_sel 匹配维度，这里“按位索引取 Ax 对应列”用于计算残差
        #     # 注意：这不是“修改/抽取 Ax”，而是仅在残差计算时取相同位置
        #     Ax_sel = Ax             # (B, m_sel)
        # else:
        #     b_sel  = b                         # (B, m)
        #     Ax_sel = Ax                        # (B, m)

        # ---- 残差
        r = Ax - b_sel                     # (B, m_sel 或 m)
        if self.squared:
            resid_per = r.pow(2).sum(dim=1)    # ||r||_2^2
        else:
            resid_per = r.norm(dim=1)          # ||r||_2

        if self.normalize:
            m_eff = r.shape[1]
            if self.squared:
                resid_per = resid_per / max(1, m_eff)
            else:
                resid_per = resid_per / (max(1, m_eff) ** 0.5)

        residual_loss = resid_per.mean()
        total = angle_loss + self.resid_lambda * residual_loss
        return total, angle_loss, residual_loss

class DnormLoss(torch.nn.Module):
    def __init__(self, l2_lambda=1e-5):
        super(DnormLoss, self).__init__()
        self.l2_lambda = l2_lambda

    def forward(self, y_pred, y_true, excitation, model, eps: float = 1e-12):
        # y_pred_normalized = y_pred / torch.max(y_pred.reshape(-1))
        # y_true_normalized = y_true / torch.max(y_true.reshape(-1))

        B = y_pred.shape[0]
        y_pred_flat = y_pred.contiguous().view(B, -1)  # (B, N)
        y_true_flat = y_true.contiguous().view(B, -1)  # (B, N)
        y_pred_unit = F.normalize(y_pred_flat, p=2, dim=1, eps=eps)
        # dot product for each batch: (B,)
        alpha = torch.sum(y_true_flat * y_pred_unit, dim=1, keepdim=True)  # (B,1)

        # scaled prediction
        y_hat = alpha * y_pred_unit  # (B, N)

        mse_loss = torch.nn.MSELoss()(y_hat, y_true_flat)
        # mse_loss = torch.nn.MSELoss()(y_pred/np.max(y_pred.reshape(-1)), y_true/np.max(y_true.reshape(-1)))  # 计算均方误差损失
        l2_reg = torch.tensor(0.0).float().to(mse_loss.device)  # 初始化L2正则项
        # threshold = model.thres.threshold
        #
        # excitation = excitation/excitation.max()
        # mask = excitation < threshold
        # residual_ex = excitation * mask.float()
        # l2_reg += torch.norm(residual_ex, p=2)

        # 计算L2正则项
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)

        # 添加L2正则项到损失中
        loss = mse_loss + self.l2_lambda * l2_reg

        return loss, mse_loss

class L2RegularizedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, l2_lambda=1e-5):
        super(L2RegularizedCrossEntropyLoss, self).__init__()
        self.l2_lambda = l2_lambda

    def forward(self, y_pred, y_true, model):
        # y_pred_normalized = y_pred / torch.max(y_pred.reshape(-1))
        # y_true_normalized = y_true / torch.max(y_true.reshape(-1))

        CE_loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        # mse_loss = torch.nn.MSELoss()(y_pred/np.max(y_pred.reshape(-1)), y_true/np.max(y_true.reshape(-1)))  # 计算均方误差损失
        l2_reg = torch.tensor(0.0)  # 初始化L2正则项

        # 计算L2正则项
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)

        # 添加L2正则项到损失中
        loss = CE_loss + self.l2_lambda * l2_reg

        return loss, CE_loss



def main(config):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('load training data')
    gt, measurements, gt_validation, measurements_validation, excitation = load_data(config.data_path, device=device, mode='train')
    # weight_Matrix, mea_mask_array = load_weighting_matrix(config.wm_data_path, device=device)
    gt_test, measurements_test, excitation = load_data(config.test_data_path, device, mode='eval')

    epoch = config.epoch
    batch_size = config.batch_size
    grad_max = config.grad_max
    learning_rate = config.learning_rate

    # initial_conv_net = UNet(in_channel=100, out_channel=15,
    #                         inner_channel= 128,
    #                         attn_res=(7,), image_size=55).to(device)
    rand_thres = random.random()
    initial_conv_net = InitialCONVNet().to(device)

    model_parameters = filter(lambda p: p.requires_grad, initial_conv_net.parameters())
    num_params = sum([param.numel() for param in model_parameters])
    print(f"Number of model parameters: {num_params}")
    criterion = AngleLoss(l2_lambda=config.l2_lambda)
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(initial_conv_net.parameters(), lr=learning_rate[0], momentum=config.momentum,
                                weight_decay=1e-8)

    epoch_start = 0

    # load check_point
    # if os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0:
    #     initial_conv_net, optimizer, epoch_start = load_checkpoint(initial_conv_net, optimizer, config.checkpoint_dir)

    initial_conv_net.train()
    log_path = os.path.join(config.log_dir, 'log_%s.txt' % (datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    loss_training_epoch = np.zeros(epoch)
    loss_validation_epoch = np.zeros(epoch)
    loss_test_epoch = np.zeros(epoch)
    angle_training_epoch = np.zeros(epoch)
    angle_validation_epoch = np.zeros(epoch)
    angle_test_epoch = np.zeros(epoch)

    # loss_training_epoch = np.zeros(epoch)
    # mse_training_epoch = np.zeros(epoch)
    # mse_validation_epoch = np.zeros(epoch)
    # mse_test_epoch = np.zeros(epoch)
    # thres_epoch = np.zeros(epoch)
    print('start training...')

    # old_model = initial_conv_net
    # initial_conv_net.thres.threshold = nn.Parameter(torch.tensor([0.5]).to(device))
    #
    # # Save the state of the old optimizer
    # old_state = optimizer.state_dict()
    #
    # # Reinitialize the optimizer
    # optimizer = torch.optim.SGD(initial_conv_net.parameters(), lr=1e-5, momentum=config.momentum,
    #                             weight_decay=1e-8)
    #
    # # Create a mapping from the old parameters to the new parameters
    # param_mapping = {old_param: new_param for old_param, new_param in
    #                  zip(old_model.parameters(), initial_conv_net.parameters())}
    #
    # # Now transfer the state
    # new_state = optimizer.state_dict()
    # for old_param, value in old_state['state'].items():
    #     new_param = param_mapping.get(old_param)
    #     if new_param is not None:
    #         new_state['state'][id(new_param)] = value
    #
    # # Load the state back into the new optimizer
    # optimizer.load_state_dict(new_state)




    for e in range(epoch_start, epoch):

        # each epoch
        for i in range(math.ceil(measurements.shape[0]/batch_size)):
            i_start = i*batch_size
            i_end = min(i_start+batch_size, measurements.shape[0])

            measurements_batch = measurements[i_start:i_end].clone()
            gt_batch = gt[i_start:i_end].clone()

            # data argument
            measurements_batch, gt_batch = data_argument(measurements_batch, gt_batch)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward Propagation
            y_pred = initial_conv_net(measurements_batch, excitation)
            # mask_image = mask.detach().cpu().numpy()
            # mask_path = os.path.join(config.sample_dir, 'epoch-%d-iter-%d.mat' % (e + 1, i + 1))
            # sio.savemat(mask_path, {'mask': mask_image})

            #
            # summary(initial_conv_net, input_size=(100, 55, 55))

            # save sample images
            # if (i+1) % config.sample_step == 0:
            #     if not os.path.exists(config.sample_dir):
            #         os.mkdir(config.sample_dir)
            #     sample_img_path = os.path.join(config.sample_dir, 'epoch-%d-iteration-%d.mat' % (e + 1, i + 1))
            #     sample_img = y_pred[0].detach().numpy()
            #     sio.savemat(sample_img_path, {'initial': sample_img})
            #     # sample_img.save(sample_img_path)
            #     print('save image:', sample_img_path)


            # Compute and print loss

            loss, angle_loss = criterion(y_pred, gt_batch)
            # initial_conv_net.thres.threshold.retain_grad()
            loss.backward()
            # print(initial_conv_net.thres.threshold.grad,initial_conv_net.thres.threshold.item() )


            # graph = make_dot(y_pred, params=dict(list(initial_conv_net.named_parameters())))
            # graph.render('computation_graph', view=True)
            # clip gradient
            torch.nn.utils.clip_grad_value_(initial_conv_net.parameters(), clip_value=grad_max)

            # Update the parameters
            optimizer.step()

        # shuffle data
        ind = np.random.permutation(measurements.shape[0])
        measurements = measurements[ind]
        gt = gt[ind]

        # adjust learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate[min(e+1, len(learning_rate)-1)]

        # save check_point
        if (e+1) % config.checkpoint_save_step == 0 or (e+1) == config.epoch:
            if not os.path.exists(config.checkpoint_dir):
                os.mkdir(config.checkpoint_dir)
            check_point_path = os.path.join(config.checkpoint_dir, 'epoch-%d.pkl' % (e+1))
            torch.save({'epoch': e+1, 'state_dict': initial_conv_net.state_dict(), 'optimizer': optimizer.state_dict()},
                       check_point_path)
            print('save checkpoint %s', check_point_path)

        # calculate mse_loss in training dataset
        angle_loss_training = np.zeros(math.ceil(measurements.shape[0]))
        loss_training = np.zeros(math.ceil(measurements.shape[0]))
        for i in range(math.ceil(measurements.shape[0] / batch_size)):
            i_start = i * batch_size
            i_end = min(i_start + config.batch_size, measurements.shape[0])
            measurements_batch = measurements[i_start:i_end]
            gt_batch = gt[i_start:i_end]
            y_pred = initial_conv_net(measurements_batch, excitation)
            loss_training[i], angle_loss_training[i] = criterion(y_pred, gt_batch)
            # loss_training[i], mse_training[i] = criterion(y_pred, gt_batch,excitation, initial_conv_net)
            # for j in range(i_start, i_end):
            #     loss_training[j], mse_training[j] = criterion(y_pred[j - i_start], gt_batch[j - i_start],
            #                                                       excitation, initial_conv_net)

        angle_training_epoch[e] = np.mean(angle_loss_training)
        loss_training_epoch[e] = np.mean(loss_training)
        # thres_epoch[e] = 1.0/(torch.exp(-initial_conv_net.thres.threshold)+torch.tensor(1.)).item()
        # thres_epoch[e] = initial_conv_net.thres.threshold.detach().clone().cpu().item()
        # print('mse_training: ', mse_training_epoch[e])
        # print('loss_training: ', loss_training_epoch[e])

        # calculate mse_loss in validation dataset
        loss_validation = np.zeros(math.ceil(measurements_validation.shape[0]))
        angle_loss_validation = np.zeros(math.ceil(measurements_validation.shape[0]))
        for i in range(math.ceil(measurements_validation.shape[0] / batch_size)):
            i_start = i*batch_size
            i_end = min(i_start + config.batch_size, measurements_validation.shape[0])
            measurements_batch = measurements_validation[i_start:i_end]
            gt_batch = gt_validation[i_start:i_end]
            y_pred = initial_conv_net(measurements_batch, excitation)
            loss_validation[i], angle_loss_validation[i]= criterion(y_pred, gt_batch)
            # for j in range(i_start,i_end):
            #     loss_validation, mse_validation[j] = criterion(y_pred[j-i_start], gt_batch[j-i_start],excitation, initial_conv_net)

        angle_validation_epoch[e] = np.mean(angle_loss_validation)
        loss_validation_epoch[e] = np.mean(loss_validation)

        loss_test = np.zeros(math.ceil(measurements_test.shape[0]))
        angle_loss_test = np.zeros(math.ceil(measurements_test.shape[0]))
        for i in range(math.ceil(measurements_test.shape[0] / batch_size)):
            i_start = i * batch_size
            i_end = min(i_start + config.batch_size, measurements_test.shape[0])
            measurements_batch = measurements_test[i_start:i_end]
            gt_batch = gt_test[i_start:i_end]
            y_pred = initial_conv_net(measurements_batch, excitation)
            loss_test[i], angle_loss_test[i] = criterion(y_pred, gt_batch)
            # loss_test, mse_test[i] = criterion(y_pred, gt_batch, excitation, initial_conv_net)
            # for j in range(i_start, i_end):
            #     loss_test, mse_test[j] = criterion(y_pred[j - i_start], gt_batch[j - i_start],
            #                                                   excitation, initial_conv_net)

            # loss_test, mse_test[i] = criterion(y_pred, gt_batch, excitation, initial_conv_net)

        angle_test_epoch[e] = np.mean(angle_loss_test)
        loss_test_epoch[e] = np.mean(loss_test)
        # print('mse_validation: ', mse_validation_epoch[e])

        print('angle_training (epoch-%d) : %f\n' % (e + 1, angle_training_epoch[e]))
        print('loss_training (epoch-%d) : %f\n' % (e + 1, loss_training_epoch[e]))
        print('angle_validation (epoch-%d) : %f\n' % (e + 1, angle_validation_epoch[e]))
        print('loss_validation (epoch-%d) : %f\n' % (e + 1, loss_validation_epoch[e]))
        print('angle_test (epoch-%d) : %f\n' % (e + 1, angle_test_epoch[e]))
        print('loss_test (epoch-%d) : %f\n' % (e + 1, loss_test_epoch[e]))
        # print('threshold (epoch-%d) : %f\n' % (e + 1, thres_epoch[e]))


        # 打开文件以写入模式，如果文件不存在会自动创建
        with open(log_path, 'a') as file:
            # 格式化loss并写入文件
            file.write('angle_training (epoch-%d) : %f\n' % (e + 1, angle_training_epoch[e]))
            file.write('loss_training (epoch-%d) : %f\n' % (e + 1, loss_training_epoch[e]))
            file.write('angle_validation (epoch-%d) : %f\n' % (e + 1, angle_validation_epoch[e]))
            file.write('loss_validation (epoch-%d) : %f\n' % (e + 1, loss_validation_epoch[e]))
            file.write('angle_test (epoch-%d) : %f\n' % (e + 1, angle_test_epoch[e]))
            file.write('loss_test (epoch-%d) : %f\n' % (e + 1, loss_test_epoch[e]))


    angle_training_path = os.path.join(config.log_dir,
                                     'angle_training.npy')
    np.save(angle_training_path, angle_training_epoch)
    loss_training_path = os.path.join(config.log_dir,
                                     'loss_training.npy')
    np.save(loss_training_path, loss_training_epoch)
    loss_validation_path = os.path.join(config.log_dir,
                                         'loss_validation.npy')
    np.save(loss_validation_path, loss_validation_epoch)
    loss_test_path = os.path.join(config.log_dir,
                                   'loss_test.npy')
    np.save(loss_test_path, loss_test_epoch)
    angle_validation_path = os.path.join(config.log_dir,
                                       'angle_validation.npy')
    np.save(angle_validation_path, angle_validation_epoch)
    angle_test_path = os.path.join(config.log_dir,
                                       'angle_test.npy')
    np.save(angle_test_path, angle_test_epoch)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--learning_rate', type=tuple, default=np.logspace(-2, -3, 50))
    parser.add_argument('--grad_max', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--l2_lambda', type=float, default=0)
    parser.add_argument('--data_path', type=str, default='D:/ZJR/11_initial_guess/InitialGuess_pytorch-master/data/simulation_FMT/545414_rec4cir1_ro/')
    parser.add_argument('--test_data_path', type=str, default='D:/ZJR/11_initial_guess/InitialGuess_pytorch-master/data/simulation_FMT/test_545414_circle/')
    parser.add_argument('--wm_data_path', type=str, default='D:/ZJR/11_initial_guess/InitialGuess_pytorch-master/data/simulation_FMT/')
    parser.add_argument('--sample_step', type=int, default=50)
    parser.add_argument('--sample_dir', type=str, default='./samples/')
    parser.add_argument('--checkpoint_save_step', type=int, default=40)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    parser.add_argument('--log_dir', type=str, default='./log/')

    config = parser.parse_args()
    print(config)
    main(config)

    # read mse
    angle_training_path = os.path.join(config.log_dir,
                                     'angle_training.npy')
    angle_training_epoch = np.load(angle_training_path)
    loss_training_path = os.path.join(config.log_dir,
                                     'loss_training.npy')
    loss_training_epoch = np.load(loss_training_path)
    angle_validation_path = os.path.join(config.log_dir,
                                       'angle_validation.npy')
    angle_validation_epoch = np.load(angle_validation_path)

    angle_test_path = os.path.join(config.log_dir,
                                  'angle_test.npy')
    angle_test_epoch = np.load(angle_test_path)

    # threshold_path = os.path.join(config.log_dir,
    #                                    'threshold.npy')
    # threshold_epoch = np.load(threshold_path)

    # plot mse curve with epoches

    # 绘制曲线

    plt.plot(np.log(angle_training_epoch), label='Training')
    plt.plot(np.log(angle_validation_epoch), label='Validation')
    plt.plot(np.log(angle_test_epoch), label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Log(MSE)')
    plt.legend()
    # plt.title('曲线图')
    plt.show()

    # plt.plot(threshold_epoch, label='Threshold')
    # plt.xlabel('Epoch')
    # plt.ylabel('Threshold')
    # plt.legend()
    # plt.show()
