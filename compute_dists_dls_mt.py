import argparse
import models
import torch
import os
import multiprocessing
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cfg', type=str, default='dls_06')
parser.add_argument('--ninput', type=int, default=106)
parser.add_argument('--nsample', type=int, default=5)
parser.add_argument('--num_thread', type=int, default=8)

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex', use_gpu=False)

results_dir = os.path.expanduser('~/results/diverse_gan/bicyclegan/%s/test/images' % opt.cfg)
print(results_dir)


def worker(qq, i, models):
    imgs = []
    for j in range(opt.nsample):
        img_name = 'input_%03d_random_sample%02d.png' % (i, j + 1)
        img_path = os.path.join(results_dir, img_name)
        # print(img_path)
        img = util.im2tensor(util.load_image(img_path))
        imgs.append(img)

    LPIPS = 0.0
    n_pairs = 0
    for p in range(1, len(imgs)):
        for q in range(p):
            LPIPS += models.forward(imgs[q], imgs[p])
            n_pairs += 1
    LPIPS /= n_pairs
    qq.put([LPIPS])


queue = multiprocessing.Queue()
workers = []
for i in range(opt.num_thread):
    workers.append(multiprocessing.Process(target=worker, args=(queue, i, model)))
for worker in workers:
    worker.start()


queue.get()

#LPIPS = queue.get_nowait()
#print(LPIPS)

t_LPIPS = 0.0


t_LPIPS /= opt.ninput
print('Total LPIPS %.4f' % t_LPIPS)
