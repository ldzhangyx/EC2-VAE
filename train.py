import torch
import numpy as np
import os
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils import data
import sklearn.utils
from ec2vae import *
from utils import *
from utils_old import *

# data: (2, 2154)
dataset = np.load('/scratch/dw1920/music_data/Nottingham/data.npy', allow_pickle = True)
melody = dataset[0]
chord = dataset[1]

# loss function with 2 reconstruction loss and two KL-divergence loss
def loss_function(recon,
                  recon_rhythm,
                  target_tensor,
                  rhythm_target,
                  distribution_1,
                  distribution_2,
                  step,
                  beta=1):
    CE1 = F.cross_entropy(
        recon.view(-1, recon.size(-1)),
        target_tensor,
        reduction='elementwise_mean')
    CE2 = F.cross_entropy(
        recon_rhythm.view(-1, recon_rhythm.size(-1)),
        rhythm_target,
        reduction='elementwise_mean')
    normal1 = Normal(
        torch.zeros(distribution_1.mean.size()).cuda(),
        torch.ones(distribution_1.stddev.size()).cuda())
    normal2 = Normal(
        torch.zeros(distribution_2.mean.size()).cuda(),
        torch.ones(distribution_2.stddev.size()).cuda())
    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()
    
    return CE1 + CE2 + beta * (KLD1 + KLD2)

# rhythm feature, the [sum of 128-dims, -2, -1] on the input 130-dim-data vector
def tensor_prepare(batch, c):
    n_batch, n_size, n_dim = batch.shape
    encode_tensor = batch
    rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)
    rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)
    rhythm_target = torch.from_numpy(rhythm_target).float()
    rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
    target_tensor = encode_tensor.view(-1, encode_tensor.size(-1)).max(-1)[1]
    return encode_tensor, target_tensor, rhythm_target, c

# the training function
def train(data, condition, step):
    batch = data
    c = condition
    num = 0
    encode_tensor, target_tensor, rhythm_target, c = tensor_prepare(batch, c)
    encode_tensor = encode_tensor.cuda()
    target_tensor = target_tensor.cuda()
    rhythm_target = rhythm_target.cuda()
    c = c.cuda()
    optimizer.zero_grad()
    recon, recon_rhythm, d1m, d1s, d2m, d2s = model(encode_tensor, c)
    distribution_1, distribution_2 = Normal(d1m, d1s), Normal(d2m, d2s)
    loss = loss_function(
        recon,
        recon_rhythm,
        target_tensor,
        rhythm_target,
        distribution_1,
        distribution_2,
        step,
        beta=1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    step += 1
    if step % 10 == 0:
        print("batch {}'s loss: {:.5f}".format(step, loss.item()))
    if step % 50 == 0:
        scheduler.step()
    return step

# scheduler to decay the learning rate
class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]
    
    
# datset class for dataloader
class Dataset(data.Dataset):
    def __init__(self, data, condition):
        all_data = list()
        all_condition = list()
        for i, j in zip(data, condition):
            if i.shape[0] == j.shape[0]:
                all_data.append(i)
                all_condition.append(j)
            else:
                if i.shape[0] < j.shape[0]:
                    all_data.append(i)
                    all_condition.append(j[:i.shape[0]])
                else:
                    all_data.append(i[:j.shape[0]])
                    all_condition.append(j)
                
        length = np.concatenate(all_data).shape[0]//(32)*32
        self.data = np.concatenate(all_data)[:length].reshape(-1, 32, 130)
        self.condition = np.concatenate(all_condition)[:length].reshape(-1, 32, 12)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = self.data[index]
        y = self.condition[index]

        return X, y
    
# main function
model = VAE(130, 2048, 3, 12, 128, 128, 32)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = MinExponentialLR(optimizer, gamma=0.95, minimum=1e-5)
step = 0
print('Using: ',
      torch.cuda.get_device_name(torch.cuda.current_device()))
model.cuda()

# model training
model.train()
for epoch in range(1, 101):
    print('Epoch: {}'.format(epoch))
    train_data = Dataset(melody, chord)
    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)
    for i, j in train_loader:
        i = i.float()
        j = j.float()
        step = train(i, j, step)
    
torch.save(model.cpu().state_dict(), 'params_128.pt')
