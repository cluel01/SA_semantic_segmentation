import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import os 

from torch.utils.tensorboard import SummaryWriter

from pytorch_segmentation.losses.dice import DiceLoss,dice_loss

from .evaluate import evaluate
from .utils.plotting import plot_predictions


def train(model, train_dl, valid_dl, loss_fn, optimizer, epochs,device,model_path,tensorboard_path=None,scheduler =  None,
         scheduler_warmup=10,early_stopping = None,metric="iou",deeplab=False,nimgs=2,figsize=(6,4),seed=42):
    torch.manual_seed(seed)
    
    run_name = os.path.basename(model_path).split(".")[0]
    print(f"INFO: Start run for {run_name}!")

    if tensorboard_path is not None:
        writer = SummaryWriter(os.path.join(tensorboard_path,run_name))

    start = time.time()
    model = model.to(device)

    train_loss, valid_loss = [], []

    best_valid_score = -np.inf
    early_stopping_counter = 0
    #best_valid_loss = np.inf
    best_model_wghts = None
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        if early_stopping is not None:
            if early_stopping == early_stopping_counter:
                print(f"INFO: Early stopping after {epoch} epochs")
                break

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl

            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_score = pd.DataFrame([{"acc":0.0,"iou":0.0,"dice":0.0}])

            step = 0

            # iterate over data
            for batch in dataloader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)

                    if deeplab:
                        outputs = outputs["out"]

                    loss = loss_fn(outputs, y) #+ dice_loss(outputs,y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        if deeplab:
                            outputs = outputs["out"]
                        loss = loss_fn(outputs, y) #+ dice_loss(outputs,y)

                # stats - whatever is the phase
                with torch.no_grad():
                    #score = evaluate(outputs.cpu(), y.cpu())
                    score = evaluate(outputs, y) #on GPU

                running_score  += score*len(x)
                running_loss += loss.item()*len(x) 

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_score = (running_score / len(dataloader.dataset)).iloc[0].to_dict()

            print('{} Loss: {:.4f} {}: {}'.format(phase, epoch_loss,metric, epoch_score[metric]))

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

            if tensorboard_path is not None:
                writer.add_scalar('loss/'+phase,epoch_loss,epoch)
                for k,v in epoch_score.items():
                    writer.add_scalar(k+"/"+phase,v,epoch)

                if phase == "train":
                    writer.add_scalar("Learning rate",optimizer.param_groups[0]["lr"],epoch)
                    
                    if (epoch % 5 == 0) or (epochs-1 == epoch ):
                        fig = plot_predictions(model,x,y,seed=seed,deeplab=deeplab,nimgs=nimgs,figsize=figsize)
                        writer.add_figure("Segmentation masks/train",fig,epoch)

                if phase == "valid":
                    if (epoch % 5 == 0) or (epochs-1 == epoch ):
                        fig = plot_predictions(model,x,y,seed=seed,deeplab=deeplab,nimgs=nimgs,figsize=figsize)
                        writer.add_figure("Segmentation masks/valid",fig,epoch)

            if phase == "valid":
                if epoch_score[metric] > best_valid_score:
                    best_valid_score = epoch_score[metric]
                    best_model_wghts = model.state_dict().copy()
                    torch.save(model.state_dict(), model_path)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if scheduler:
                    if epoch >= scheduler_warmup:
                        scheduler.step()#(epoch_score[metric])
                

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    model.load_state_dict(best_model_wghts)
    #torch.save(model.state_dict(), model_path)
    
    if tensorboard_path is not None:
        writer.close()
    return train_loss, valid_loss    



