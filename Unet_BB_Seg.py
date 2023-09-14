from train import training
from inference import inference
from easydict import EasyDict
import click
import os
from datetime import datetime
import socket
import logging

@click.command()
#Training
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--epoch', help='Total epoch size', metavar='INT', type=click.IntRange(min=1), default=200, show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0)
@click.option('--pin_memory', help='Use pin memory', metavar='BOOL',type=bool, default=False)
@click.option('--shuffle', help='Use shuffle', metavar='BOOL', type=bool, default=False)
@click.option('--num_workers', help='Number of workers', metavar='INT', type=click.IntRange(min=1), default=1)
@click.option('--pt_step', help='How often to save pretrained data', metavar='INT', default=50, show_default=True)
@click.option('--pretrain_path', help='start from pretrained data', metavar='[PATH|URL]', type=str)
@click.option('--data_path', help='dataset path', metavar='[PATH|URL]', type=str, required=True)
@click.option('--save_path', help='save path', metavar='[PATH|URL]', type=str, required=True)
@click.option('--inference', help='Enable inference mode', metavar='BOOL', type=bool, default=False)
@click.option('--lr', help='set learning late default:0.01', metavar='FLOAT', type=float, default=1e-3)
@click.option('--name', help='set custom running folder name', metavar='STRING', type=str, default=None)

def main(**kwargs):
    arg = EasyDict(kwargs)

    mylogger = logging.getLogger("config")
    mylogger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()#console print
    mylogger.addHandler(stream_handler)

    arg.save_path = os.path.join(arg.save_path,(f"{len(os.listdir(arg.save_path)):06d}_{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}" if arg.name==None else arg.name))
    if not os.path.exists(arg.save_path):
        os.mkdir(arg.save_path)
    else:
        mylogger.warning('There is SavePath already')
        exit()

    file_handler = logging.FileHandler(os.path.join(arg.save_path,"config.log"))
    formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    mylogger.addHandler(file_handler)

    for key, value in arg.items():
        mylogger.info('%s: %s', key, value)

    if arg.inference:#inference 진행
        inference(arg)
    else:#traning 진행
        training(arg)

if __name__ == "__main__":
    main()