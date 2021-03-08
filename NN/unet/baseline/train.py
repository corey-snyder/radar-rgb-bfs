import sys
import os
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import warnings

from torch import optim
from torch.utils.data import DataLoader
from unet_model import UNet
from datasetloader import CDnetDataset
from tqdm import tqdm
from f_measure import compute_f_measure
from skimage.io import imsave

warnings.simplefilter('ignore', category=UserWarning)

DEVICE = 'cuda:1'
SPLIT = 0.7
def train_net(net, epochs, batch_size, lr, category, video):
    cdnet_path = '/mnt/data0-nfs/shared-datasets/CDnet2014'
    path_checkpoint = 'checkpoints/'
    
    train_dataset = CDnetDataset(cdnet_path, category, video, mode='train')
    test_dataset = CDnetDataset(cdnet_path, category, video, mode='test')
    roi = train_dataset.spatial_roi
    roi = roi.to(DEVICE)
    w_bg = train_dataset.bg_weight
    w_fg = train_dataset.fg_weight

    n_train_samples = len(train_dataset)
    indices = np.arange(n_train_samples)
    train_indices = indices[:int(n_train_samples*SPLIT)]
    val_indices = indices[int(n_train_samples*SPLIT):]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=2) 
    val_loader = DataLoader(train_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print('''
    Starting training for "{}: {}":
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Testing size: {}
    '''.format(category, video, epochs, batch_size, lr, len(train_indices), len(val_indices), len(test_dataset)))

    w = 0
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=w)

    criterion = nn.BCELoss()
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        if (epoch+1) % 15 == 0:
            lr = lr/10
            optimizer = optim.SGD(net.parameters(),
                                  lr=lr,
                                  momentum=0.9,
                                  weight_decay=w)
        epoch_loss = 0
        for i, (orig_tensors, gt_tensors, orig_files, gt_files) in enumerate(tqdm(train_loader)):
            orig_tensors = orig_tensors.to(DEVICE)
            gt_tensors = gt_tensors.to(DEVICE)
            fg_preds = net(orig_tensors)
            fg_preds_flat = fg_preds.view(-1)
            gt_flat = gt_tensors.view(-1)

            loss_weights = compute_loss_weights(gt_tensors, roi, w_bg, w_fg).view(-1)
            criterion = nn.BCELoss(weight=loss_weights)
            loss = criterion(fg_preds_flat, gt_flat)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_f_measure = eval_net(net, train_loader, roi, w_bg, w_fg)
        val_loss, val_f_measure = eval_net(net, val_loader, roi, w_bg, w_fg)
        print('Training Loss: {:.4e}'.format(train_loss))
        print('Training F-measure: {:.5f}'.format(train_f_measure))

        print('Validation Loss: {:.4e}'.format(val_loss))
        print('Validation F-measure: {:.5f}'.format(val_f_measure))
        #if epoch == epochs-1:
        #    torch.save(net.state_dict(), path_checkpoint + 'model-{}-{}.pth'.format(category, video))
        #    print('Model saved!')
    
    print('Saving soft predictons...')
    test_net(net, train_loader, cdnet_path, category, video) 
    test_net(net, val_loader, cdnet_path, category, video)
    test_net(net, test_loader, cdnet_path, category, video)
    print('Writing to results file...')
    with open('results.txt', 'a') as f:
        f.write('{}-{}: Training F-Measure = {:5f}, Validation F-Measure = {:5f}\n'.format(category, video, train_f_measure, val_f_measure))

def compute_loss_weights(gt_tensors, roi, w_bg, w_fg):
    #make n_bg*w_bg = n_fg*w_fg
    #make w_bg + w_fg = 1
    #mute pixels outside ROI
    #n_bg = torch.sum(gt_tensors[:, :, roi == 1] == 0).item()
    #n_fg = torch.sum(gt_tensors[:, :, roi == 1] == 1).item()
    gt_byte = gt_tensors.byte()
    #if n_fg:
    #    w_bg = (n_fg+n_bg)/(2*n_bg)
    #    w_fg = (n_fg+n_bg)/(2*n_fg)
    #else:
    #    w_bg = 1
    #    w_fg = 1
    weights = torch.zeros_like(gt_tensors) 
    weights[:, :, roi == 0] = 0
    for n in range(weights.size(0)):
        weights[n, :, (roi == 1) * (gt_byte[n, 0] == 0)] = w_bg
        weights[n, :, (roi == 1) * (gt_byte[n, 0] == 1)] = w_fg
    return weights

def eval_net(net, dataloader, roi, w_bg, w_fg):
    epoch_loss = 0
    epoch_f_measure = 0
    net.eval()
    with torch.no_grad():
        for orig_tensors, gt_tensors, in_files, gt_files in tqdm(dataloader):
            orig_tensors = orig_tensors.to(DEVICE)
            gt_tensors = gt_tensors.to(DEVICE)
            n_pixels = gt_tensors.size(2)*gt_tensors.size(3)
            
            fg_preds = net(orig_tensors)
            fg_preds_flat = fg_preds.view(-1)
            gt_flat = gt_tensors.view(-1)

            loss_weights = compute_loss_weights(gt_tensors, roi, w_bg, w_fg).view(-1)
            criterion = nn.BCELoss(weight=loss_weights)
            loss = criterion(fg_preds_flat, gt_flat)
            f_measure = compute_f_measure(fg_preds, gt_tensors, roi)
            epoch_loss += loss.item()
            epoch_f_measure += f_measure.item()
            
    return epoch_loss/len(dataloader), epoch_f_measure/len(dataloader)

def test_net(net, dataloader, cdnet_path, category, video):
    with torch.no_grad():
        for orig_tensors, _, in_files, _ in tqdm(dataloader):
            orig_tensors = orig_tensors.to(DEVICE)
            fg_preds = net(orig_tensors)
            for i in range(len(in_files)):
                curr_pred = fg_preds[i].cpu().squeeze(0).numpy()
                #curr_pred[curr_pred >= 0.5] = 255
                #curr_pred[curr_pred < 0.5] = 0
                #curr_pred = curr_pred.astype(np.uint8)
                curr_pred = (curr_pred*255).astype(np.uint8)
                pred_name = in_files[i].replace('in', 'bin').replace('.jpg', '.png')
                category_path = os.path.join(cdnet_path, 'soft_results', category)
                if not os.path.exists(category_path):
                    os.mkdir(category_path)
                video_path = os.path.join(category_path, video)
                if not os.path.exists(video_path):
                    os.mkdir(video_path)
                #save_path = os.path.join(cdnet_path, 'results', category, video, pred_name)
                save_path = os.path.join(video_path, pred_name)
                imsave(save_path, curr_pred)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=str, help='category in CDnet14')
    parser.add_argument('-v', '--video', type=str, help='video within category of CDnet14')
    args = parser.parse_args()
    category = args.category
    video = args.video
    net = UNet(n_channels=3, n_classes=1)
    net.to(DEVICE)
    n_epochs = 30
    batch_size = 2
    lr = 1e-2
    train_net(net, n_epochs, batch_size, lr, category, video)

