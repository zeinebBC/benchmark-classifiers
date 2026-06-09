import torchvision.transforms.functional as fT
import torch.nn.functional as F
import random
import numpy as np
import nibabel as nib
import torch
import blosc2
from pathlib import Path
def pad_to_shape(vol, target_shape, constant_values=0):
    pad_width = []
    current_shape = vol.shape[-3:]
    for i in range(3):
        total_pad = max(0, target_shape[i] - current_shape[i])
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    if isinstance(vol, torch.Tensor):
        pad_tuple = tuple([elem for tup in reversed(pad_width) for elem in tup])
        return F.pad(vol, pad_tuple, mode="constant", value=constant_values), pad_width
    else:
        leading_dims = len(vol.shape) - 3
        pad_width_full = [(0, 0)] * leading_dims + pad_width
        return np.pad(vol, pad_width_full, mode="constant", constant_values=constant_values), pad_width

def unpad_tensor(vol, pad_width):
    slices = []
    for before, after in pad_width:
        if after == 0:
            slices.append(slice(before, None))
        else:
            slices.append(slice(before, -after))
    return vol[..., slices[0], slices[1], slices[2]]

def unpad_batch_with_channel(batch_vols, pad_widths):
    return torch.stack([unpad_tensor(vol, pad_w) for vol, pad_w in zip(batch_vols, pad_widths)])

def pad_batch_with_channel(vols, target_shape, constant_values=0):
    padded_vols = []
    pad_widths = []
    for vol in vols:
        padded_vol, pad_width = pad_to_shape(vol, target_shape, constant_values)
        padded_vols.append(padded_vol)
        pad_widths.append(pad_width)
    return torch.stack(padded_vols), pad_widths

def load_b2nd(path):
   
    schunk = blosc2.open(path, mode="r")
    arr = schunk[:][0]  
    return arr

def load_volume(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        return np.load(path)["image"]
    elif path.suffix in [".nii", ".gz"]:
        return nib.load(str(path)).get_fdata()
    elif path.suffix == ".b2nd":
        return np.transpose(load_b2nd(path),(2,1,0))
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
   
            


def transform_image_and_label(image, label):
    # Horizontal flip
    if random.random() < 0.4:
        image = fT.hflip(image)
        label = fT.hflip(label)

    # Vertical flip
    if random.random() < 0.4:
        image = fT.vflip(image)
        label = fT.vflip(label)

    # Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-10, 10)
        image = fT.rotate(image, angle)
        label = fT.rotate(label, angle, interpolation=F.InterpolationMode.NEAREST)

    return image, label







