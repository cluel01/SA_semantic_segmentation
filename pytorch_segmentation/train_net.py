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

from sklearn.metrics import f1_score,accuracy_score
from torch.utils.tensorboard import SummaryWriter

from .evaluate import evaluate
from .utils.plotting import plot_predictions


def train(model, train_dl, valid_dl, loss_fn, optimizer, epochs,device,model_path,tensorboard_path=None,scheduler =  None,
         scheduler_warmup=10,metric="iou",deeplab=False,nimgs=2,seed=42):
    torch.manual_seed(seed)
    
    run_name = os.path.basename(model_path).split(".")[0]

    if tensorboard_path is not None:
        writer = SummaryWriter(os.path.join(tensorboard_path,run_name))

    start = time.time()
    model = model.to(device)

    train_loss, valid_loss = [], []

    best_valid_score = -np.inf
    #best_valid_loss = np.inf
    best_model_wghts = None
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

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

                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        if deeplab:
                            outputs = outputs["out"]
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                with torch.no_grad():
                    score = evaluate(outputs, y)
                
                running_score  += score*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_score = (running_score / len(dataloader.dataset)).iloc[0].to_dict()

            print('{} Loss: {:.4f} {}: {}'.format(phase, epoch_loss,metric, epoch_score[metric]))

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

            if tensorboard_path is not None:
                writer.add_scalar('loss/'+phase,epoch_loss,epoch)
                for k,v in epoch_score.items():
                    writer.add_scalar(k+"/"+phase,v,epoch)

                if phase == "train":
                    if scheduler is not None:
                        writer.add_scalar("Learning rate",scheduler.get_last_lr()[0],epoch)
                    else:
                        writer.add_scalar("Learning rate",optimizer.param_groups[0]["lr"],epoch)
                    
                    if (epoch % 5 == 0) or (epochs-1 == epoch ):
                        fig = plot_predictions(model,x,y,seed=seed,deeplab=deeplab,nimgs=nimgs)
                        writer.add_figure("Segmentation masks/train",fig,epoch)

                if phase == "valid":
                    if (epoch % 5 == 0) or (epochs-1 == epoch ):
                        fig = plot_predictions(model,x,y,seed=seed,deeplab=deeplab,nimgs=nimgs)
                        writer.add_figure("Segmentation masks/valid",fig,epoch)

            if phase == "valid":
                if epoch_score[metric] > best_valid_score:
                    best_valid_score = epoch_score[metric]
                    best_model_wghts = model.state_dict().copy()
            else:
                if scheduler:
                    if epoch >= scheduler_warmup:
                        scheduler.step()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    model.load_state_dict(best_model_wghts)
    torch.save(model.state_dict(), model_path)
    
    if tensorboard_path is not None:
        writer.close()
    return train_loss, valid_loss    








    # fig, axs = plt.subplots(ncols=4,nrows=nimgs, squeeze=False)
    # fig.patch.set_facecolor('white')
    # for n_row,i in enumerate(idxs):
    #     scores = evaluate(out[i].unsqueeze(0),labels[i].unsqueeze(0),reduction=False)
    #     mask_tens = out[i].cpu().byte()
    #     img_tens = (inputs[i]*255).cpu().byte()
    #     true_mask_tens = labels[i].byte()
    #     seg_mask_tens = draw_segmentation_masks(img_tens,mask_tens.bool(),alpha=0.6)
    #     tens = {"Img":img_tens,"GT":true_mask_tens,"Pred":mask_tens,"Mask":seg_mask_tens}

    #     for n_col,tup in enumerate(tens.items()):
    #         k,t = tup
    #         img = F_vis.to_pil_image(t)
    #         axs[n_row, n_col].imshow(np.asarray(img))
    #         axs[n_row, n_col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #         axs[n_row, n_col].set_title(k)
    # return axs








# def train_net(net,datasets,
#               device,save_dir,
#               epochs: int = 5,
#               batch_size: int = 1,
#               learning_rate: float = 1e-5,
#               save_checkpoint: bool = True,
#               amp: bool = False,
#               seed = 42):
#     torch.manual_seed(seed)
#     # 1. Create dataset
#     train_set,val_set = datasets
#     n_train,n_val = (len(train_set),len(val_set))

#     # 3. Create data loaders
#     loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=False)
#     train_loader = DataLoader(train_set, shuffle=True, **loader_args)
#     val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {learning_rate}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_checkpoint}
#         Device:          {device.type}
#         Mixed Precision: {amp}
#     ''')

#     # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
#     optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
#     criterion = nn.CrossEntropyLoss()
#     global_step = 0

#     # 5. Begin training
#     for epoch in range(epochs):
#         net.train()
#         epoch_loss = 0
#         with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
#             for batch in train_loader:
#                 images = batch["x"]
#                 true_masks = batch["y"]

#                 assert images.shape[1] == net.n_channels, \
#                     f'Network has been defined with {net.n_channels} input channels, ' \
#                     f'but loaded images have {images.shape[1]} channels. Please check that ' \
#                     'the images are loaded correctly.'

#                 images = images.to(device=device, dtype=torch.float32)
#                 true_masks = true_masks.to(device=device, dtype=torch.long)

#                 with torch.cuda.amp.autocast(enabled=amp):
#                     masks_pred = net(images)
#                     loss = criterion(masks_pred, true_masks) \
#                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
#                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
#                                        multiclass=True)

#                 optimizer.zero_grad(set_to_none=True)
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()

#                 pbar.update(images.shape[0])
#                 global_step += 1
#                 epoch_loss += loss.item()
#                 pbar.set_postfix(**{'loss (batch)': loss.item()})

#                 # Evaluation round
#                 division_step = (n_train // (10 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                         for tag, value in net.named_parameters():
#                             tag = tag.replace('/', '.')

#                         val_score = evaluate(net, val_loader, device)
#                         scheduler.step(val_score)

#                         logging.info('Validation Dice score: {}'.format(val_score))
                        

#         if save_checkpoint:
#             Path(save_dir).mkdir(parents=True, exist_ok=True)
#             torch.save(net.state_dict(), str(save_dir + '/checkpoint_epoch{}.pth'.format(epoch + 1)))
#             logging.info(f'Checkpoint {epoch + 1} saved!')