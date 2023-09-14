import torch
import os
import pandas as pd
from Data import FPUS
from PIL import Image
import numpy as np
from utils import *
import torchvision
from tqdm import tqdm
import time
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(arg):
    print("start training")

    #args
    batch = 4
    random_seed = arg.seed
    shuffle = arg.shuffle
    pin_memory = arg.pin_memory
    num_workers = arg.num_workers

    #path
    datasetPath = arg.data_path
    annoPath = os.path.join(datasetPath,'boxes/annotations.csv')
    imagePath = os.path.join(datasetPath,'four_poses')

    pretrain_path = arg.pretrain_path
    checkPoint = torch.load(pretrain_path)
    save_path = arg.save_path

    #processing
    torch.manual_seed(random_seed)

    model = checkPoint['model'].to(device)
    model.eval()

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
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=test_sampler, shuffle=shuffle, collate_fn=dataset.collate_fn, pin_memory=pin_memory, num_workers=num_workers)

    #logging
    mylogger = logging.getLogger("inference")
    mylogger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()#console print
    stream_handler.setLevel(logging.INFO)
    mylogger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(arg.save_path,"inference.log"))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    mylogger.addHandler(file_handler)

    if not os.path.exists(os.path.join(save_path,'inference')):
        os.mkdir(os.path.join(save_path,'inference'))

    save_path = os.path.join(save_path,'inference')

    if not os.path.exists(os.path.join(save_path,'img')):
        os.mkdir(os.path.join(save_path,'img'))

    subset_size = 200
    inference_indices = test_indices[:subset_size]
    inference_sampler = torch.utils.data.SubsetRandomSampler(inference_indices)
    inference_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=inference_sampler, collate_fn=dataset.collate_fn, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    # 데이터셋 분할 확인
    mylogger.info(f"추측 데이터 개수: {len(inference_loader)}")

    with torch.no_grad():
        with tqdm(inference_loader,desc=f"Inference", leave=False) as inference_pbar:
            whole_time = AverageMeter()
            predict_time = AverageMeter()

            for i, (images, boxes, labels) in enumerate(inference_pbar):
                start = time.time()
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]

                annoted_image = []
                for idx,image in enumerate(images):
                    annoted_image.append(list(np.array(visualize(image,boxes[idx],labels[idx])).transpose((2,0,1))))
                for image in images:
                    pTime = time.time()
                    annoted_image.append(list(np.array(detect(model,image, min_score=0.2, max_overlap=0.5, top_k=200)).transpose((2,0,1))))
                    predict_time.update(time.time() - pTime)
                    mylogger.debug(f"batch: {batch}, batch time: {whole_time.val}, predict time: {predict_time.val}")

                img_grid = torchvision.utils.make_grid(torch.tensor(np.array(annoted_image)),nrow=batch)
                img_grid = Image.fromarray(img_grid.permute(1, 2, 0).cpu().numpy())
                img_grid.save(os.path.join(save_path, 'img', f"inference_{i:06d}_predict.png"))

                whole_time.update(time.time() - start)

            mylogger.info(f"batch: {batch}, mean batch time: {whole_time.avg}, mean predict time: {predict_time.avg}")



