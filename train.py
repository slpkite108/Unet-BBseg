import torch
import os
import pandas as pd
from Data import FPUS
from PIL import Image
import numpy as np
from Unet import UnetEdit
#from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from loss import intersection_over_union
from torch.utils.tensorboard import SummaryWriter


def training(arg):
    print("start training")

    #args
    batch = arg.batch
    num_epochs = arg.epoch
    random_seed = arg.seed
    shuffle = arg.shuffle
    pin_memory = arg.pin_memory
    num_workers = arg.num_workers
    pt_step = arg.pt_step
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #path
    datasetPath = arg.data_path
    annoPath = os.path.join(datasetPath,'boxes/annotations.csv')
    imagePath = os.path.join(datasetPath,'four_poses')

    pretrain_path = arg.pretrain_path
    use_pt = True if pretrain_path != None else False

    save_path = arg.save_path

    #processing
    torch.manual_seed(random_seed)
    writer = SummaryWriter()

    model = UnetEdit(batch,6,600,600).to(device)
    #model = Unet().to(device)
    #transform = transforms.Compose([transforms.ToTensor(),])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)
    if use_pt:
        model.load_state_dict(torch.load(pretrain_path))

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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=train_sampler, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 검증용 데이터셋
    val_indices = indices[train_size:train_size + val_size]
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=val_sampler, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 테스트용 데이터셋
    test_indices = indices[train_size + val_size:]
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=test_sampler, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 데이터셋 분할 확인
    print("전체 데이터 개수:", len(dataset))
    print("학습 데이터 개수:", len(train_loader))
    print("검증 데이터 개수:", len(val_loader))
    print("테스트 데이터 개수:", len(test_loader))

    for epoch in range(num_epochs):
        
        running_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # Training
        for i, data in enumerate(train_loader):
            #image, truth = data['image'].type(torch.float),data['bbconv']
            image = data['image'].type(torch.float).to('cuda')
            truth = data['bbcoord'].type(torch.float).to('cuda')
            #truth = truth/255
            predictions = model(image)
            
            loss = 1-intersection_over_union(predictions, truth)
            loss = loss.sum()
            writer.add_scalar('Loss/train',loss,epoch*len(train_loader))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                #image, truth = data['image'].type(torch.float),data['bbconv']
                image = data['image'].type(torch.float).to('cuda')
                truth = data['bbcoord'].type(torch.float).to('cuda')
                #truth = truth/255
                predictions = model(image)
                predictions = torch.round(predictions) # Rounding the predictions for classification
                loss = 1-intersection_over_union(predictions, truth)
                loss = loss.sum()
                val_loss += loss.item()
                #val_acc += dice_sim(predictions, truth)*100

        epoch_train_loss = running_loss / (train_size//batch+1)
        epoch_val_loss = val_loss / (val_size//batch+1)
        epoch_val_acc = val_acc / (val_size//batch+1)
        scheduler.step(epoch_val_loss) # LR Scheduler
        print(f"==>train_loss: {epoch_train_loss} ==>val_loss: {epoch_val_loss} ==>val_accuracy: {epoch_val_acc}")
        
        if epoch % pt_step == 0:
            checkpoint_path = os.path.join(save_path,'pt',f'model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at epoch {epoch}")
    