import os
import torch
import sparseconvnet as scn
import numpy as np
from models import Autoencoder, Encoder, Decoder
from module.utils.muon_track_dataset import MuonPose, MuonPoseLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.manual_seed(42)

batch_size = 10
trainloader = MuonPoseLoader("/home/jack/classes/thesis/autohas/LArCentroids/train/", batch_size=batch_size, device='cuda:0', return_energy=True)
valloader = MuonPoseLoader("/home/jack/classes/thesis/autohas/LArCentroids/val/", batch_size=batch_size, device='cuda:0', return_energy=True)
now = datetime.now().strftime("%b-%d-%y_%H:%M:%S")
logger = SummaryWriter(log_dir=f"autoencoder_run/{now}/")

device = 'cuda:0'

dimension = 3
reps = 3
# [(output_features, input_features), ...]
# encoder_layers = [(1, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
# decoder_layers = [(64, 32), (64, 16), (32, 8), (16, 4), (8, 2), (3, 1, False)]
'''
encoder_layers = [(1, 4), (4, 8), (8, 16)]
decoder_layers = [(16, 8), (16, 4), (8, 2), (3, 1, False)]

encoder = Encoder(3, 2, encoder_layers, leakiness=0, input_layer=torch.LongTensor([128,128,128]), device=device)
secondary = Encoder(3, 2, encoder_layers, leakiness=0, input_layer=torch.LongTensor([128,128,128]), device=device)
decoder = Decoder(3, decoder_layers, unet=True, is_submanifold=True, use_sparsify=False, device=device)

parameters = [encoder.parameters(), decoder.parameters()]
parameters = [{'params': p} for p in parameters]
'''
unet = scn.UNet(dimension, reps, [1, 4, 16, 64, 256], residual_blocks=True, downsample=[2, 2])
model = scn.Sequential().add(scn.InputLayer(3, [128,128,128])).add(unet)
model.to(device)
parameters = model.parameters()

# loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.9, 0.1]).to(device))
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(parameters)

densifier = scn.Sequential().add(
    scn.InputLayer(3, torch.LongTensor([128,128,128]))
).add(
    scn.SparseToDense(3, 1)
)


for epoch in range(20):
    print(f'Epoch {epoch}')
#    encoder.train(), decoder.train()
    model.train()
    total_loss = 0
    for i in range(len(trainloader)):
        light, energy, target = trainloader[i]

        '''
        # pileup: combines all batch instances into one
        # this actually probably shouldn't work because scn.BatchNorm
        # breaks with n=1 batches
        energy[0][:, -1] = 0
        target[0][:, -1] = 0
        target[1] /= batch_size
        '''

        target = densifier[0](target)
        target = [target.get_spatial_locations(), target.features]

#        blocks = encoder(energy, return_blocks=True)
#        prediction = decoder(blocks[::-1])

        prediction = model(energy)

        # softmax predictions batchwise
        for batch_idx in range(batch_size):
            slice_ = prediction.get_spatial_locations()[:, -1] == batch_idx
            prediction.features[slice_] = torch.nn.Softmax(dim=0)(prediction.features[slice_])

        '''
        # for CE loss
        dense_prediction = densifier[1](prediction).argmax(dim=1)

        topk = target[1].topk(2, 0).indices
        target[1] = torch.zeros_like(target[1])
        target[1][topk] = 1
        target = target[1].reshape(-1).long()
        '''

        loss = loss_fn(prediction.features, target[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f"\tIter {100*i/len(trainloader):.2f}%: Loss {loss.item()}", end='\r')
    total_loss /= len(trainloader)
    print(f"\tAverage training loss: {total_loss}")

#    encoder.eval(), decoder.eval()
    model.eval()
    total_loss = 0
    for i in range(len(valloader)):
        light, energy, target = valloader[i]

        '''
        # pileup
        energy[0][:, -1] = 0
        target[0][:, -1] = 0
        target[1] /= batch_size
        '''

        target = densifier[0](target)
        target = [target.get_spatial_locations(), target.features]

#        blocks = encoder(energy, return_blocks=True)
#        prediction = decoder(blocks[::-1])
        prediction = model(energy)

        # softmax predictions batchwise
        for batch_idx in range(batch_size):
            slice_ = prediction.get_spatial_locations()[:, -1] == batch_idx
            prediction.features[slice_] = torch.nn.Softmax(dim=0)(prediction.features[slice_])

        '''
        # for CE loss
        topk = target[1].topk(2, 0).indices
        target[1] = torch.zeros_like(target[1])
        target[1][topk] = 1
        target = target[1].reshape(-1).long()
        '''

        loss = loss_fn(prediction.features, target[1])
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"\tIter {100*i/len(valloader):.2f}%: Loss {loss.item()}", end='\r')
    total_loss /= len(valloader)
    print(f"\tAverage validation loss: {total_loss}")

import matplotlib.pyplot as plt
for i, (light, energy, target) in enumerate(trainloader):
    if i > 5:
        break
    
    target = densifier[0](target)
    target = [target.get_spatial_locations(), target.features]

#    blocks = encoder(energy, return_blocks=True)
#    prediction = decoder(blocks[::-1])
    prediction = model(energy)

    for batch_idx in range(5):
        slice_ = prediction.get_spatial_locations()[:, -1] == batch_idx
        prediction.features[slice_] = torch.nn.Softmax(dim=0)(prediction.features[slice_])

    dense_input = densifier(energy).squeeze().sum(dim=0).detach().cpu().numpy()
    dense_prediction = densifier[1](prediction).squeeze().sum(dim=0).detach().cpu().numpy()
    dense_target = densifier(target).squeeze().sum(dim=0).detach().cpu().numpy()

    fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(dense_input)
    fig.add_subplot(1, 3, 2)
    plt.imshow(dense_prediction)
    fig.add_subplot(1, 3, 3)
    plt.imshow(dense_target)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

for i, (light, energy, target) in enumerate(valloader):
    if i > 5:
        break
    
    target = densifier[0](target)
    target = [target.get_spatial_locations(), target.features]

#    blocks = encoder(energy, return_blocks=True)
#    prediction = decoder(blocks[::-1])
    prediciton = model(energy)

    for batch_idx in range(5):
        slice_ = prediction.get_spatial_locations()[:, -1] == batch_idx
        prediction.features[slice_] = torch.nn.Softmax(dim=0)(prediction.features[slice_])

    dense_input = densifier(energy).squeeze().sum(dim=0).detach().cpu().numpy()
    dense_prediction = densifier[1](prediction).squeeze().sum(dim=0).detach().cpu().numpy()
    dense_target = densifier(target).squeeze().sum(dim=0).detach().cpu().numpy()

    fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(dense_input)
    fig.add_subplot(1, 3, 2)
    plt.imshow(dense_prediction)
    fig.add_subplot(1, 3, 3)
    plt.imshow(dense_target)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

import code
code.interact(local=locals())