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
import copy
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
import json
from pytorch_segmentation.losses.dice import DiceLoss,dice_loss

from .evaluate import evaluate
from .utils.plotting import  plot_predictions_subset


def train(model, train_dl, valid_dl, loss_fn, optimizer, epochs,device,model_path,tensorboard_path=None,scheduler =  None,
         scheduler_warmup=10,early_stopping = None,metric="iou",deeplab=False,nimgs=2,figsize=(6,4),seed=42,cfg=None):
    torch.manual_seed(seed)
    
    run_name = os.path.basename(model_path).split(".")[0]
    print(f"INFO: Start run for {run_name}!")

    if cfg is not None:
        #convert non-Python builtin vars to string
        for k,v in cfg.items():
            if v.__class__.__module__ != "builtins":
                cfg[k] = str(v)

        with open(model_path + ".config", 'w') as fp:
            json.dump(cfg, fp)

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

            # iterate over data
            for batch in dataloader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)

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
                    score,_ = evaluate(outputs, y) #on GPU

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
                        fig = plot_predictions_subset(model,x,y,seed=seed,deeplab=deeplab,nimgs=nimgs,figsize=figsize)
                        writer.add_figure("Segmentation masks/train",fig,epoch)

                if phase == "valid":
                    if (epoch % 5 == 0) or (epochs-1 == epoch ):
                        fig = plot_predictions_subset(model,x,y,seed=seed,deeplab=deeplab,nimgs=nimgs,figsize=figsize)
                        writer.add_figure("Segmentation masks/valid",fig,epoch)

            if phase == "valid":
                if epoch_score[metric] > best_valid_score:
                    best_valid_score = epoch_score[metric]

                    if isinstance(model, nn.DataParallel):
                        best_model_wghts = model.module.state_dict().copy()
                    else:
                        best_model_wghts = model.state_dict().copy()

                    torch.save(best_model_wghts, model_path)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if scheduler:
                    if epoch >= scheduler_warmup:
                        scheduler.step()#(epoch_score[metric])
                

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_wghts)
    else:
        model.load_state_dict(best_model_wghts)
    #torch.save(model.state_dict(), model_path)
    
    if tensorboard_path is not None:
        writer.close()
    return train_loss, valid_loss    


def train_classification(model, train_dl,valid_dl,criterion, optimizer, scheduler, device,model_path,tensorboard_path=None, num_epochs=25,seed=42):
    torch.manual_seed(seed)
    since = time.time()

    run_name = os.path.basename(model_path).split(".")[0]
    if tensorboard_path is not None:
        writer = SummaryWriter(os.path.join(tensorboard_path,run_name))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0
    es_counter = 0

    dataloaders = {"train": train_dl,"val":valid_dl}
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_f1 = 0.

            # Iterate over data.
            for data in dataloaders[phase]:
                #print(np.unique(labels,return_counts=True)[1])
                
                inputs = data["x"].to(device)
                labels = data["y"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze(1)
                    #_, preds = torch.max(outputs, 1)
                    logits = nn.Sigmoid()(outputs)
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                
                preds = (logits > 0.5).long()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data) 
                running_f1 += f1_score(labels.data.cpu(),preds.cpu(), average='weighted') #/ len(inputs)
                
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() /  len(dataloaders[phase].dataset) 
            epoch_f1 = running_f1 / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f} F1:{:.4f}'.format(
                phase, epoch_loss, epoch_acc,epoch_f1))
            
            writer.add_scalar('loss/'+phase, epoch_loss, epoch)
            writer.add_scalar('acc/'+phase, epoch_acc, epoch)
            writer.add_scalar('f1/'+phase, epoch_f1, epoch)

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_score:
                best_score = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),model_path)
                es_counter = 0
            elif (phase == "val") and (epoch_f1 <= best_score):
                es_counter += 1
                print(best_score)
                
            if es_counter >= 50:
                print("Early stoppping")
                model.load_state_dict(best_model_wts)
                return model

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val score: {:4f}'.format(best_score))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

