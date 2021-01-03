import os 
import argparse
import time 
from tqdm.auto import tqdm
import shutil 
import numpy as np  
import sys

import torch.nn as nn 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

sys.path.append(os.path.basename(__file__)+"/lib")

from lib.model import EfficientDet
from lib.dataset import CustomDataset, Resizer, Normalizer, Augmenter, collater


def parse_args():

    parser = argparse.ArgumentParser(description="EfficientDet: Scalable and Efficient Object Detection training module")
    
    # General Parameters
    parser.add_argument("--name", type=str, default="exp_0", help="Name of experiment")

    # Model parameters
    parser.add_argument("--model_coeff", type=int, default=0, required=True, help="Efficientdet model coeff (b0, b1, ....)")
    parser.add_argument("--image_size", type=int, default=512, help="The common height and width for all images")
    parser.add_argument("--ckpt", type=str, help="path to checkpoint from where to resume training ")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial Learning rate for training")
    parser.add_argument("--gpu", type=bool, default=True, required=True, help="True if training is to use GPU. False if not.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha parameter for focal loss")
    parser.add_argument("--gamma", type=float, default=1.5, help="Gamma parameter for focal loss")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to run training for")
    parser.add_argument("--es_min_delta", type=float, default=0.0, help="Early Stopping's Parameter: minimum change in loss to qualify as improvement")
    parser.add_argument("--es_patience", type=int, default=0, help="Early stopping's parameter: Number of epochs with no improvement in loss to stop training. 0 to disable")

    # Logging parameters
    parser.add_argument("--log_path", type=str, default="tensorboard/", help="Path to store tensorboard logs")
    parser.add_argument("--save_path", type=str, default="trained/", help="path to folder where to save trained model")
    parser.add_argument("--best_epoch", type=int, default=0)
    parser.add_argument("--best_loss", type=float, default=1e5)

    # Train Dataset parameters
    
    # Format of Dataset: 
    # - Root Directory
    #       - Annotations (COCO Format)
    #           - train_instance.json
    #           - test_instance.json
    #           - val_instance.json
    #       - train
    #           - img1
    #           - img2 
    #           .
    #           .
    #           - imgn 
    #       - test
    #           - img1
    #           - img2
    #           .
    #           .
    #           - imgn
    #       - val
    #           - img1
    #           - img2
    #           .
    #           .
    #           - imgn

    parser.add_argument("--root_dir", type=str, required=True, help="Path to root dataset directory")
    parser.add_argument("--coco_dir", type=str, default="./", required=True)
    parser.add_argument("--img_dir", type=str, required=True, help="Name of the folder containing the imgs in the root dir")
    parser.add_argument("--set_dir", type=str, required=True, help="name of set (train/test/val) being used for this")
    parser.add_argument("--num_threads", type=int, default=2, help="Number of threads to utilize for loading data")

    # Validation parameters
    parser.add_argument("--val", type=bool, default=False, help="Perform validation boolean")
    parser.add_argument("--val_interval", type=int, default=5, help="Epochs interval after which to run validation")
    parser.add_argument("--val_dir", type=str, help="Path to Validation set root directory")
    parser.add_argument("--val_imgs", type=str, help="Path to Val set imgs")
    parser.add_argument("--val_coco", type=str)
    parser.add_argument("--val_set", type=str, help="Path to set dir")

    args = parser.parse_args()

    return args

def Train(args):

    if args.gpu and not torch.cuda.is_available():
        raise ValueError(f"--gpu is {args.gpu} but cuda not found")

    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"
    
    # setting the trainloader
    trainset = CustomDataset(
        root_dir = args.root_dir + "/" + args.coco_dir, 
        img_dir = args.img_dir,
        set_name = args.set_dir,
        transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
    )
    trainloader = DataLoader(
        trainset, 
        batch_size = args.batch_size, 
        shuffle = False, 
        drop_last = False, 
        collate_fn = collater, 
        num_workers = args.num_threads 
    )
    
    # If validation is enabled, set the val loader
    if args.val:
        
        valset = CustomDataset(    
            root_dir = args.val_dir + "/" + args.val_coco, 
            img_dir = args.val_imgs,
            set_name = args.val_set,
            transform = transforms.Compose([Normalizer(), Resizer()])
        )

        valloader = DataLoader(
            valset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            drop_last=False, 
            collate_fn=collater, 
            num_workers=args.num_threads
        )    
    
    # setting the device and other model params
    
    num_classes = trainset.num_classes()
    efficientdet = EfficientDet(
        model_coeff = args.model_coeff, 
        num_classes=num_classes,
        focal_alpha = args.alpha, 
        focal_gamma = args.gamma, 
        device = device
    )

    # loading pretrained models (if passed)
    try:
        efficientdet.load_state_dict(torch.load(args.ckpt))
        print("checkpoint loaded successfully!")
    except Exception as e:
        print("ERROR: Model Loading failed: ", e)


    efficientdet = efficientdet.to(device)
    efficientdet.train()

    # Setting the optimizer and scheduler 
    optimizer = torch.optim.Adam(efficientdet.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # set up logging and model save directories
    args.log_path = args.log_path + "/" + "EfficientDet" + "/" + args.name
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)

    # setting up the tensorboard writer
    writer = SummaryWriter(args.log_path)

    len_trainloader = len(trainloader)

    if args.val:

        for epoch in range(args.epochs):

            efficientdet.train()

            epoch_loss = []
            epoch_progress = tqdm(trainloader)
            for idx, data  in enumerate(epoch_progress):
                try:
                    
                    # zero grading the optimizer
                    optimizer.zero_grad()

                    # forward pass
                    
                    img_batch = data['img'].to(device).float()
                    annot_batch = data['annot'].to(device)

                    cls_loss, reg_loss = efficientdet([img_batch, annot_batch])

                    # Optimization block 

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    total_loss = cls_loss + reg_loss
                    if total_loss == 0:
                        continue

                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(efficientdet.parameters(), 0.1) 

                    optimizer.step()

                    epoch_loss.append(float(total_loss))
                    total_mean_loss = np.mean(epoch_loss)

                    epoch_progress.set_description(
                        "Epoch: {}/{}, Batch id: {}/{}, Classification Loss: {:.5f}, Regression Loss: {:.5f}, Batch Loss: {:.5f}, Total Loss: {:.5f}".format(
                            epoch+1, args.epochs, idx, len_trainloader, cls_loss, reg_loss, total_loss, total_mean_loss
                        )
                    )

                    writer.add_scalar('Train/Total_Loss', total_mean_loss, epoch * len_trainloader + idx)                   
                    writer.add_scalar('Train/Regression_Loss', reg_loss, epoch * len_trainloader + idx)
                    writer.add_scalar('Train/Classification_loss (Focal Loss)', cls_loss, epoch * len_trainloader + idx)
                
                except Exception as e:
                    print(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            if epoch % args.val_interval == 0:

                efficientdet.eval()
                loss_reg_ls = []
                loss_cls_ls = []
                
                for idx, data in enumerate(valloader):

                    img_batch = data['img'].to(device).float()
                    annot_batch = data['annot'].to(device)

                    with torch.no_grad():
                        
                        cls_loss, reg_loss = efficientdet([img_batch, annot_batch])

                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss_cls_ls.append(float(cls_loss))
                        loss_reg_ls.append(float(reg_loss))
                
                cls_loss = np.mean(loss_cls_ls)
                reg_loss = np.mean(loss_reg_ls)
                loss = cls_loss + reg_loss

                print(
                    'Epoch: {}/{}, Classification Loss: {:1.5f}, Regression Loss: {:1.5f}, Total Loss: {:1.5f}'.format(
                        epoch+1, args.epochs, cls_loss, reg_loss, np.mean(loss)
                    )
                )



                writer.add_scalar('Val/Total_Loss', loss, epoch)
                writer.add_scalar('Val/Regression_Loss', reg_loss, epoch)
                writer.add_scalar('Val/Classification_Loss', cls_loss, epoch)

                if loss + args.es_min_delta < args.best_loss:

                    args.best_loss = loss
                    args.best_epoch = epoch 
                    torch.save(efficientdet, os.path.join(args.save_path, "efficientdet_best.pth"))

                    dummy = torch.rand(1, 3, 512, 512)
                    dummy = dummy.to(device)
            
                if isinstance(efficientdet, nn.DataParallel):
                
                    efficientdet.backbone_net.model.set_swish(memory_efficient=False)

                    try:
                        torch.onnx.export(
                            efficientdet.module, dummy, os.path.join(args.save_path, "efficientdet_best.onnx"), 
                            verbose=False, opset_version=11
                            )   
                    except:
                        print("Failed ONNX export")

                else:

                    efficientdet.backbone_net.model.set_swish(memory_efficient=False)
                    torch.onnx.export(
                        efficientdet, dummy, os.path.join(args.save_path, "efficientdet_best.onnx"), 
                        verbose=False, opset_version=11
                    )
                    efficientdet.backbone_net.model.set_swish(memory_efficient=True)
                
                if epoch - args.best_epoch > args.es_patience > 0:
                    print(f"Stopped training at epoch: {epoch}, Lowerst loss: {loss}")
                    break
                    
    else:

        for epoch in range(args.epochs):

            efficientdet.train()

            epoch_loss = []
            epoch_progress = tqdm(trainloader)
            for idx, data  in enumerate(epoch_progress):
                try:
                    
                    # zero grading the optimizer
                    optimizer.zero_grad()

                    # forward pass
                    
                    img_batch = data['img'].to(device).float()
                    annot_batch = data['annot'].to(device)

                    cls_loss, reg_loss = efficientdet([img_batch, annot_batch])

                    # Optimization block 

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    total_loss = cls_loss + reg_loss
                    if total_loss == 0:
                        continue

                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(efficientdet.parameters(), 0.1) 

                    optimizer.step()

                    epoch_loss.append(float(total_loss))
                    total_mean_loss = np.mean(epoch_loss)

                    epoch_progress.set_description(
                        "Epoch: {}/{}, Batch id: {}/{}, Classification Loss: {:.5f}, Regression Loss: {:.5f}, Batch Loss: {:.5f}, Total Loss: {:.5f}".format(
                            epoch+1, args.epochs, idx, len_trainloader, cls_loss, reg_loss, total_loss, total_mean_loss
                        )
                    )

                    writer.add_scalar('Train/Total_Loss', total_mean_loss, epoch * len_trainloader + idx)                   
                    writer.add_scalar('Train/Regression_Loss', reg_loss, epoch * len_trainloader + idx)
                    writer.add_scalar('Train/Classification_loss (Focal Loss)', cls_loss, epoch * len_trainloader + idx)
                
                except Exception as e:
                    print(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            torch.save(efficientdet, os.path.join(args.save_path, "efficientdet_best.pth"))

            dummy = torch.rand(1, 3, 512, 512)
            dummy = dummy.to(device)
            if isinstance(efficientdet, nn.DataParallel):
                
                efficientdet.backbone_net.model.set_swish(memory_efficient=False)

                try:
                    torch.onnx.export(
                        efficientdet.module, dummy, os.path.join(args.save_path, "efficientdet_best.onnx"), 
                        verbose=False, opset_version=11
                        )   
                except:
                    print("Failed ONNX export")

            else:

                efficientdet.backbone_net.model.set_swish(memory_efficient=False)
                torch.onnx.export(
                    efficientdet, dummy, os.path.join(args.save_path, "efficientdet_best.onnx"), 
                    verbose=False, opset_version=11
                )
                efficientdet.backbone_net.model.set_swish(memory_efficient=True)


    writer.close()


    
if __name__ == "__main__":
    opts = parse_args()
    Train(opts)