dataFile = 'G:/DOT_cnn/lib/data/real_2022_6/real_2dTower'
data = io.loadmat(dataFile+'/raw_mea.mat')
bar_mea = data['raw_mea']*1.4

# bar_mea_max = bar_mea.max(1)
# bar_mea_min = bar_mea.min(1)

# bar_mea = (bar_mea-bar_mea_min)/(bar_mea_max-bar_mea_min)

bar_mea = np.log(np.expand_dims(bar_mea,0))

bar_mea = mea_filter(bar_mea)
bar_mea = Z_ScoreNormalization(bar_mea)
bar_mea = torch.Tensor(bar_mea)
bar_mea = bar_mea.to(torch.float32)
bar_mea = bar_mea.to(DEVICE)

bar_output, _ = model(bar_mea)
bar_output = bar_output.to(torch.device("cpu"))

# if not os.path.exists(dataFile+ '/result'):
# os.makedirs(resultFile+ '/result')

io.savemat(resultFile+'/mua_recon1.mat', {'mua_recon':bar_output.detach().numpy()})
show3Dmap(bar_output.detach().numpy()[0], 'result')
print(resultFile)

def Z_ScoreNormalization(x, mode='test'):

    global mu_train
    global sigma_train

    if mode == 'train':
        mu_train = x.mean(0)
        sigma_train = x.std(0)
        mu = mu_train
        sigma = sigma_train
    elif mode =='test':
        mu = mu_train
        sigma = sigma_train
    x_norm = (x-mu)/sigma
    return x_norm