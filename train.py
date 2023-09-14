import torch
import os
import pandas as pd
from Data import FPUS
from PIL import Image
import numpy as np
from Unet import UnetLoc
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from loss import MultiBoxLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import *
#import matplotlib
import time
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training(arg):
    print("start training")
    runtime = AverageMeter()
    runtimestart = time.time()

    #args
    batch = arg.batch
    num_epochs = arg.epoch
    random_seed = arg.seed
    lr = arg.lr
    shuffle = arg.shuffle
    pin_memory = arg.pin_memory
    num_workers = arg.num_workers
    pt_step = arg.pt_step
    
    global device

    #path
    datasetPath = arg.data_path
    annoPath = os.path.join(datasetPath,'boxes/annotations.csv')
    imagePath = os.path.join(datasetPath,'four_poses')

    checkPoint = arg.pretrain_path

    save_path = arg.save_path

    #processing
    torch.manual_seed(random_seed)

    #tensorboard
    global writer
    writer = SummaryWriter(log_dir=os.path.join(save_path,'run'))
    
    #logging
    mylogger = logging.getLogger("train")
    mylogger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()#console print
    mylogger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(arg.save_path,"train.log"))
    formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    mylogger.addHandler(file_handler)
    #models

    if checkPoint != None:
        checkPoint = torch.load(checkPoint)

        model = checkPoint['model']
        start_epoch = checkPoint['epoch'] + 1
        num_epochs += start_epoch - 1
        optimizer = checkPoint['optimizer']
    else:
        model = UnetLoc(5).to(device)
        start_epoch = 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.parameters() ,lr=1e-3, momentum=0.9, weight_decay=5e-4)
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)

    model = model.to(device)

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, neg_pos_ratio=4).to(device)

    df = pd.read_csv(annoPath, index_col=0)
    filtered = df.dropna().groupby(['id','type','name','winSize']).agg({'label':list,'xtl':list,'ytl':list,'xbr':list,'ybr':list}).sort_values(by=['type','id'],ascending=True).reset_index()
    image_paths = [os.path.join(imagePath,t,n) for t, n in list(zip(filtered['type'],filtered['name']))]
    target_labels = filtered[['label','xtl','ytl','xbr','ybr']].to_dict('records')

    dataset = FPUS(image_paths,target_labels)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    indices = torch.randperm(total_size)

    #학습용 데이터셋
    train_indices = indices[:train_size]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=train_sampler, collate_fn=dataset.collate_fn, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 검증용 데이터셋
    val_indices = indices[train_size:train_size + val_size]
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=val_sampler, collate_fn=dataset.collate_fn, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 테스트용 데이터셋
    test_indices = indices[train_size + val_size:]
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=test_sampler, collate_fn=dataset.collate_fn, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 데이터셋 분할 확인
    print("전체 데이터 개수:", len(dataset))
    print("학습 데이터 개수:", len(train_loader))
    print("검증 데이터 개수:", len(val_loader))
    print("테스트 데이터 개수:", len(test_loader))

    if not os.path.exists(os.path.join(save_path,'img')):
        os.mkdir(os.path.join(save_path,'img'))

    for epoch in range(start_epoch,num_epochs+1):
        
        running_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss
        start = time.time()

        # Training
        with tqdm(train_loader,desc=f"Epoch {epoch} - Training", leave=False) as train_pbar:
            for i, (images, boxes, labels) in enumerate(train_pbar):
                data_time.update(time.time() - start)

                images = images.to(device)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]

                predicted_locs, predicted_scores  = model(images)

                loss = criterion(predicted_locs, predicted_scores, boxes, labels)

                writer.add_scalar('totalLoss/train',loss,train_size*epoch+i)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), images.size(0))
                batch_time.update(time.time() - start)
                #print(loss)
                running_loss += loss.item()
                start = time.time()
                
                with torch.no_grad():
                    if i == 0:
                        
                        annoted_image = []
                        for i,image in enumerate(images):
                            annoted_image.append(list(np.array(visualize(image,boxes[i],labels[i])).transpose((2,0,1))))
                        for image in images:
                            annoted_image.append(list(np.array(detect(model, image, min_score=0.2, max_overlap=0.5, top_k=200)).transpose((2,0,1))))

                        img_grid = torchvision.utils.make_grid(torch.tensor(np.array(annoted_image)),nrow=batch)
                        #writer.add_images('image/Train',img_grid,epoch)
                        img_grid = Image.fromarray(img_grid.permute(1, 2, 0).cpu().numpy())
                        img_grid.save(os.path.join(save_path, 'img', f"train_Epoch_{epoch:06d}_predict.png"))
                        
        print('Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(epoch, i+1, len(train_loader),
                                                                                batch_time=batch_time,
                                                                                data_time=data_time))
        epoch_train_loss = running_loss / (train_size//batch+1)


        # Validation
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False) as val_pbar:
                for i, (images, boxes, labels) in enumerate(val_pbar):

                    images = images.to(device)
                    boxes = [b.to(device) for b in boxes]
                    labels = [l.to(device) for l in labels]
                    
                    predicted_locs, predicted_scores  = model(images)
                    
                    loss = criterion(predicted_locs, predicted_scores, boxes, labels)

                    writer.add_scalar('totalLoss/val',loss,val_size*epoch+i)
                    val_loss += loss.item()
                    acc = torch.where(loss<0.5, torch.tensor(1.,device=device),torch.tensor(0.,device=device))
                    val_acc += acc.mean().item()

        epoch_val_loss = val_loss / (val_size//batch+1)
        epoch_val_acc = val_acc / (val_size//batch+1)
        scheduler.step(epoch_val_loss)


        with torch.no_grad():
            with tqdm(test_loader, desc=f"Epoch {epoch} - Validation", leave=False) as test_pbar:
                for i, (images, boxes, labels) in enumerate(test_pbar):

                    images = images.to(device)
                    boxes = [b.to(device) for b in boxes]
                    labels = [l.to(device) for l in labels]
                    
                    predicted_locs, predicted_scores  = model(images)
                    
                    loss = criterion(predicted_locs, predicted_scores, boxes, labels)

                    writer.add_scalar('totalLoss/test',loss,test_size*epoch+i)
                    test_loss += loss.item()
                    acc = torch.where(loss<0.5, torch.tensor(1.,device=device),torch.tensor(0.,device=device))
                    test_acc += acc.mean().item()
                    
        epoch_test_loss = test_loss / (test_size//batch+1)
        epoch_test_acc = test_acc / (test_size//batch+1)
        

         # LR Scheduler
        mylogger.info(f"Epoch: {epoch} ==>train_loss: {epoch_train_loss} ==>val_loss: {epoch_val_loss} ==>val_accuracy: {epoch_val_acc} ==>test_loss: {epoch_test_loss} ==>test_accuracy: {epoch_test_acc} ==>LR: {optimizer.param_groups[0]['lr']}\n")
        
        if epoch % pt_step == 0:
            save_checkpoint(save_path,model,epoch,optimizer)

    runtime.update(time.time() - runtimestart)
    mylogger.info(f"Batch: {batch} ==>Runtime: {runtime.val // 3600:02}:{(runtime.val % 3600) // 60:02}:{runtime.val % 60:02}")