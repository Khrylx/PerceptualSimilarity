import argparse
import models
import torch
import os
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cfg', type=str, default='pretrain')
parser.add_argument('--ninput', type=int, default=106)
parser.add_argument('--nsample', type=int, default=50)
parser.add_argument('--gpu_index', type=int, default=0)

opt = parser.parse_args()

device = torch.device('cuda', index=opt.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex', use_gpu=opt.gpu_index >= 0 and torch.cuda.is_available(), gpu_ids=[opt.gpu_index])

results_dir = os.path.expanduser('~/results/diverse_gan/bicyclegan/%s/test/images' % opt.cfg)
print(results_dir)

t_LPIPS = 0.0
for i in range(opt.ninput):
    imgs = []
    for j in range(opt.nsample):
        img_name = 'input_%03d_random_sample%02d.png' % (i, j + 1)
        img_path = os.path.join(results_dir, img_name)
        # print(img_path)
        img = util.im2tensor(util.load_image(img_path)).to(device)
        imgs.append(img)

    LPIPS = 0.0
    n_pairs = 0
    for p in range(1, len(imgs)):
        for q in range(p):
            LPIPS += model.forward(imgs[q], imgs[p])
            n_pairs += 1
    LPIPS /= n_pairs
    t_LPIPS += LPIPS
    print('%d %.4f' % (i, LPIPS.item()))

t_LPIPS /= opt.ninput
print('Total LPIPS %.4f' % t_LPIPS)
