"""Convert checkpoint from torch/MicroSoft"""
import os
import argparse
import mindspore as ms
import torch
from mindformers.tools.utils import str2bool


def convert_pretrained_weight(pth_file="swin_base_patch4_window7_224.pth", ms_ckpt_path="swin_base_p4w7.ckpt"):
    """
    convert swin_base_p4w7 weights from pytorch to mindspore
    pytorch required.
    """
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    for k, v in state_dict['model'].items():
        if 'decoder.0' in k:
            k = k.replace('decoder.0', 'decoder')
        if 'patch_embed.' in k:
            k = k.replace('proj', 'projection')
        if 'relative_position' in k:
            k = k.replace('relative_position', 'relative_position_bias.relative_position')
        if 'norm' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if '.mlp' in k:
            if '.fc1' in k:
                k = k.replace('.fc1', '.mapping')
                if "weight" in k:
                    v = v.transpose(-1, 0)
            if '.fc2' in k:
                k = k.replace('.fc2', '.projection')
                if "weight" in k:
                    v = v.transpose(-1, 0)
        if '.qkv' not in k:
            ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})
        else:
            data = ms.Tensor(v.numpy())
            length = len(data)
            ms_ckpt.append({'name': k.replace('.qkv', '.q'), 'data': data[:length // 3]})
            ms_ckpt.append({'name': k.replace('.qkv', '.k'), 'data': data[length // 3:length // 3 * 2]})
            ms_ckpt.append({'name': k.replace('.qkv', '.v'), 'data': data[length // 3 * 2:]})

    if not os.path.exists(ms_ckpt_path):
        try:
            ms.save_checkpoint(ms_ckpt, ms_ckpt_path)
        except:
            raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.')


def convert_finetuned_weight(pth_file="swin_base_patch4_window7_224.pth", ms_ckpt_path="swin_base_p4w7.ckpt"):
    """
    convert swin_base_p4w7 weights from pytorch to mindspore
    pytorch required.
    """
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    for k, v in state_dict['model'].items():
        if 'head' in k:
            ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})
            continue
        if 'patch_embed.' in k:
            k = k.replace('proj', 'projection')
        if 'relative_position' in k:
            k = k.replace('relative_position', 'relative_position_bias.relative_position')
        if 'norm' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if '.mlp' in k:
            if '.fc1' in k:
                k = k.replace('.fc1', '.mapping')
                if "weight" in k:
                    v = v.transpose(-1, 0)
            if '.fc2' in k:
                k = k.replace('.fc2', '.projection')
                if "weight" in k:
                    v = v.transpose(-1, 0)
        if '.qkv' not in k:
            ms_ckpt.append({'name': 'encoder.' + k, 'data': ms.Tensor(v.numpy())})
        else:
            data = ms.Tensor(v.numpy())
            length = len(data)
            ms_ckpt.append({'name': 'encoder.' + k.replace('.qkv', '.q'), 'data': data[:length // 3]})
            ms_ckpt.append(
                {'name': 'encoder.' + k.replace('.qkv', '.k'), 'data': data[length // 3:length // 3 * 2]})
            ms_ckpt.append({'name': 'encoder.' + k.replace('.qkv', '.v'), 'data': data[length // 3 * 2:]})

    if not os.path.exists(ms_ckpt_path):
        try:
            ms.save_checkpoint(ms_ckpt, ms_ckpt_path)
        except:
            raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="swin weight convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="swin_base_patch4_window7_224.pth",
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default="swin_base_p4w7.ckpt",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--is_pretrain",
                        type=str2bool,
                        default=False,
                        help="whether converting pretrained weights.")
    opt = parser.parse_args()

    if opt.is_pretrain:
        convert_pretrained_weight(opt.torch_path, opt.mindspore_path)
    else:
        convert_finetuned_weight(opt.torch_path, opt.mindspore_path)
