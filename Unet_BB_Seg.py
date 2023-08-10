from train import training
from easydict import EasyDict
import click

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
def main(**kwargs):
    training(arg = EasyDict(kwargs))

if __name__ == "__main__":
    main()