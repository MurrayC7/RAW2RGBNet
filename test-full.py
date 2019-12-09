import argparse
import os
from importlib import import_module
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import utils
import torchvision.transforms.functional as F
import numpy as np

parser = argparse.ArgumentParser(description="Test Script")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="name of model for this training"
)
parser.add_argument("--checkpoint", type=str, required=True, help="path to load model checkpoint")
parser.add_argument("--output", type=str, required=True, help="path to save output images")
parser.add_argument("--data", type=str, required=True, help="path to load data images")

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.output):
    os.makedirs(opt.output)

model = import_module('models.' + opt.model.lower()).make_model(opt)
model.load_state_dict(torch.load(opt.checkpoint)['state_dict_model'])
model = model.cuda()
model = model.eval()

images = utils.load_all_image(opt.data)
images.sort()


def infer(im):
    w, h = im.size
    pad_w = 4 - w % 4
    pad_h = 4 - h % 4
    to_tensor = transforms.ToTensor()

    im_pad = transforms.Pad(padding=(pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2), padding_mode='reflect')(im)
    im_augs = [
        to_tensor(im_pad),
        to_tensor(F.hflip(im_pad)),
        to_tensor(F.vflip(im_pad)),
        to_tensor(F.hflip(F.vflip(im_pad)))
    ]
    output_augs = []
    for im_pad in im_augs:
        im_pad_th = im_pad.unsqueeze(0).cuda()
        with torch.no_grad():
            torch.cuda.empty_cache()
            output = model(im_pad_th)
        output_augs.append(np.transpose(output.squeeze(0).cpu().numpy(), (1, 2, 0)))
    output_augs = [
        output_augs[0],
        np.fliplr(output_augs[1]),
        np.flipud(output_augs[2]),
        np.fliplr(np.flipud(output_augs[3]))
    ]
    output = np.mean(output_augs, axis=0) * 255.
    output = output[pad_h // 2:-(pad_h - pad_h // 2), pad_w // 2:-(pad_w - pad_w // 2), :]
    output = output.round()
    output[output >= 255] = 255
    output[output <= 0] = 0
    output = Image.fromarray(output.astype(np.uint8), mode='RGB')
    return output


for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    img = Image.open(im_path)
    output = infer(img)
    assert output.size == img.size
    output.save(os.path.join(opt.output, filename))
