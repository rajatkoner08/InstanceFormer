# ------------------------------------------------------------------------
# InstanceFormer code for visualizing reference points and sampling locations.
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os
import matplotlib
import random

root = 'root_path'
'''
During inference --analysis flag has to be enabled to save the referanve points.
'''
def draw_inter_ref(c, i, im, instance, ref, layer=-1):
    # if i == 0:
    #     plt.scatter(x=ref[instance, i, layer, ..., 0, :,0] * im.shape[1],
    #     y=ref[instance,layer, i, ..., 0, :,1] * im.shape[0], c=c, s=100)
    # else:
    # inverse linear transformation for double sigmoid in referance propagation module
    plt.scatter(x=(np.clip(((ref[instance, i, layer,..., 0, :, 0] - .5) / (.731 - .5)) * im.shape[1], 0, im.shape[1])),
                y=(np.clip(((ref[instance, i, layer,..., 0, :,1] - .5) / (.731 - .5)) * im.shape[0], 0, im.shape[0])), c=c,
                s=100)


def draw_init_ref(c, i, im, init_ref, instance):
    #if i == 0:
    plt.scatter(x=init_ref[instance, i, 0] * im.shape[1], y=init_ref[instance, i, 1] * im.shape[0], c=c, s=200)
    # else:
    #     plt.scatter(x=(np.clip(((init_ref[instance, i, 0] - .5) / (.731 - .5)) * im.shape[1], 0, im.shape[1])),
    #             y=(np.clip(((init_ref[instance, i, 0] - .5) / (.731 - .5)) * im.shape[0], 0, im.shape[0])), c=c,
    #             s=200)


def draw_ref(video, model, video_path):
    ref = np.load(
        f'{root}/instanceformer_output/analysis/{model}/analysis/{video}_ref_analysis.npy', mmap_mode='r')
    init_ref = np.load(
        f'{root}/instanceformer_output/analysis/{model}/analysis/{video}_init_ref_analysis.npy', mmap_mode='r')
    list_of_files = sorted(filter(os.path.isfile, glob.glob(video_path + '*')))
    Path(f'{root}/instanceformer_output/analysis/{video}').mkdir(parents=True, exist_ok=True)
    dpi = matplotlib.rcParams['figure.dpi']
    for i, file in enumerate(list_of_files):
        im = plt.imread(file)
        height, width, depth = im.shape

        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        plt.imshow(im)

        # for p in range(1):
        layer = -1
        color = ['b','r','w','g','k','y','c','m']*2
        for iq in range(12):
            draw_inter_ref(color[iq%len(color)], i, im, iq, ref, layer)
        plt.axis('off')
        # plt.show()
        fig.savefig(f'{root}/instanceformer_output/analysis/{video}/{i}.png', dpi=fig.dpi,  bbox_inches='tight',pad_inches = 0)
        # print()

        # plt.rcParams['font.size'] = 25
        # plt.text(width - 122, height - 33, f'{i + 1}/{len(list_of_files)}', style='normal', bbox={
        #     'facecolor': 'white', 'alpha': 1, 'pad': 5})

video = 'eb9681e03a'#'3e35d85e4b'#'0b97736357'#'b1a8a404ad'#'5c3d2d3155'#'4035d3275c'#'0a49f5265b'# '00f88c4f0a'
model = 'r50_pretrain_ref_prop_class_yvis21'
# video_path = f'{root}/data/youtubeVOS/valid/JPEGImages/{video}/'

#ovis
video = '3d04522a'#'2a02f752'#'505ed57c'# '3d04522a'
model = 'r50_joint_yvis_pretrain_ref_wdcls_nf4_lr2e4_OVIS_memory_drop4_10_--supcon'
video_path = f'{root}/data/OVIS/valid/{video}/'

#yvis21
# video_path = f'{root}/data/youtubeVOS/2021/valid/JPEGImages/{video}/'
draw_ref(video, model, video_path)


