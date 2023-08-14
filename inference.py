import torch
import os
import pandas as pd
from Data import FPUS
from utils import draw_filter, save_image
from PIL import Image
import numpy as np
from Unet import UnetEdit
#from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from loss import intersection_over_union
from torch.utils.tensorboard import SummaryWriter
import cv2


def inference(arg):
    print("start training")

    #args
    batch = 1
    num_epochs = 0
    random_seed = arg.seed
    shuffle = arg.shuffle
    pin_memory = arg.pin_memory
    num_workers = arg.num_workers
    
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
    #train_indices = indices[:train_size]
    #train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    #train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=train_sampler, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 검증용 데이터셋
    #val_indices = indices[train_size:train_size + val_size]
    #val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    #val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=val_sampler, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 테스트용 데이터셋
    test_indices = indices[train_size + val_size:]
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=test_sampler, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 데이터셋 분할 확인
    #print("전체 데이터 개수:", len(dataset))
    #print("학습 데이터 개수:", len(train_loader))
    #print("검증 데이터 개수:", len(val_loader))
    print("테스트 데이터 개수:", len(test_loader))

    if not os.path.exists(os.path.join(save_path,'img')):
        os.mkdir(os.path.join(save_path,'img'))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i > 100:
                break

            #image, truth = data['image'].type(torch.float),data['bbconv']
            image = data['image'].type(torch.float).to('cuda')
            truth = data['bbcoord'].type(torch.float).to('cuda')
            #truth = truth/255
            predictions = model(image)
            print(predictions)
            predictions = predictions*600
            predictions = torch.round(predictions) # Rounding the predictions for classification
            
            imageFolder = os.path.join(save_path,'img',str(data['index'].item()))
            if not os.path.exists(imageFolder):
                os.mkdir(imageFolder)
            

            print(predictions)
            exit()
            image = np.array(image.squeeze())
            predictions = np.array(predictions.squeeze())

            
            for i in range(6):
                color = list(np.random.random(size=3) * 256)
                cv2.rectangle(image, (predictions[i][1], predictions[i][0]), (predictions[i][3], predictions[i][2]), color, 4)

            save_image(image,imageFolder,'origin')


