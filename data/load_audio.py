import argparse
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import audio_processor as module_ap
from data.datasets import SOL_ordinario


def main(args):
    """Load and save audio waveforms from wav files
    
    Example:
        python load_audio.py -p path/to/wav/files -s path/to/save/dir/ -t ReadAudio Zscore
    """
    sol_ds = SOL_ordinario(args.path_to_data, transform=args.transform)

    args.path_to_save = Path(args.path_to_save)

    for idx in range(len(sol_ds)):
        f_name = sol_ds.path_to_audio[idx].stem
        x, _, _ = sol_ds[idx]
        if args.is_tensor:
            f_name = f_name + '.pth'
            torch.save(x, args.path_to_save / f_name)
        else:
            f_name = f_name + '.npy'
            np.save(args.path_to_save / f_name, x)
                 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_data', required=True)
    parser.add_argument('-s', '--path_to_save', required=True)
    # parser.add_argument('-r', '--resample', help="resampling rate", type=int)
    parser.add_argument('-t', '--transform', nargs='+', help="input names of classes for transformation, sep. by spaces", required=True)
    args = parser.parse_args()

    if 'ToTensor' not in args.transform:
        args.is_tensor = False
    else:
        args.is_tensor = True

    list_transform = [getattr(module_ap, t)() for t in args.transform]
    args.transform = transforms.Compose(list_transform)

    main(args)
