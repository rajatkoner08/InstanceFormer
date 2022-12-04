import glob
from pathlib import Path
import os
from detectron2.utils.colormap import random_color
import json
import matplotlib.pyplot as plt
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from inference import CLASSES_OVIS, CLASSES_YVIS_21_22, CLASSES_YVIS_19
import pycocotools.mask as mask_util
from visualize import visualize

root = 'your_root'

def draw_mask(video, video_path, inference_path, dataset='yvis', instances=[0]):
    list_of_files = sorted(filter(os.path.isfile, glob.glob(video_path + '*')))
    f = open(inference_path)
    inference = json.load(f)
    Path(f'{root}/instanceformer_output/{dataset}/{video}').mkdir(parents=True, exist_ok=True)
    visualization_masks = [ins['segmentations'] for ins in inference if ins['video_name']==video]
    visualization_masks = list(map(list, zip(*visualization_masks)))
    labels = [ins['category_id'] for ins in inference if ins['video_name']==video]
    conf_scores = [ins['score'] for ins in inference if ins['video_name']==video]
    # assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(len(labels))]
    assigned_colors = ['tab:blue', 'orange','w','g','k','y','c','m']*2
    # im_shape = tuple([ins for ins in inference if ins['video_name']==video][0]['segmentations'][0]['size'])
    for i in range(len(list_of_files)):
        visualization_masks[i] = [mask_util.decode(v_mask) for v_mask in visualization_masks[i]]
    for i, file in enumerate(list_of_files):
        im = plt.imread(file)
        im = np.array(im)
        visualizer = Visualizer(im, metadata=None, scale=1.0)
        non_zero_mask = [j for j, vm in enumerate(visualization_masks[i]) if vm.sum() > 2]
        vis_mask = [visualization_masks[i][j] for j in non_zero_mask]
        vis_label = [
            CLASSES_OVIS[labels[j] - 1] if 'ovis' in dataset else CLASSES_YVIS_21_22[labels[j] - 1] if 'yvis21' in dataset
            else CLASSES_YVIS_19[labels[j] - 1]  + (f':{conf_scores[j] * 100:.1f}%') for j in non_zero_mask]
        # vis_color = [assigned_colors[i] for i in non_zero_mask]
        if len(non_zero_mask)< len(instances):
            temp_instances = range(len(non_zero_mask))
        else:
            temp_instances = instances
        vis = visualizer.overlay_instances(labels=[vis_label[i] for i in temp_instances], masks=[vis_mask[i] for i in temp_instances],
                                           assigned_colors=assigned_colors[:len(temp_instances)],alpha=.75)
        vis.save(f'{root}/instanceformer_output/{dataset}/{video}/mask2_{i}.png')

# YTVIS-19
# video = 'b1a8a404ad'#'3e35d85e4b'#'0b97736357'#'5c3d2d3155'#'4035d3275c'#'0a49f5265b'# '00f88c4f0a'
# inference_path = f'{root}/instanceformer_output/results/swin_pretrain_ref_prop_class_2/results.json'
# video_path = f'{root}/youtubeVOS/valid/JPEGImages/{video}/'

#ovis
# video = 'd41a62d4'#'1ef6cb7b'#'1b664206'#'f326bfb7'#'af48b2f9'#'f326bfb7'#'90d7f538'#'505ed57c'# '3d04522a'
# inference_path = f'{root}/instanceformer_output/results/r50_joint_yvis_pretrain_ref_wdcls_nf3_lr2e4_OVIS_memory_drop4_10_supcon/results.json'
# video_path = f'{root}/OVIS/valid/{video}/'

#YTVIS-22
video = 'f7255a57d0'#'1ecc34b1bf'#'0fc3e9dbcc'#'3e35d85e4b'
inference_path = f'{root}/instanceformer_output/results/r50_pretrain_ref_prop_class_yvis21/ckp13/results.json'
video_path = f'{root}/data/youtubeVOS/2021/valid/JPEGImages/{video}/'

draw_mask(video, video_path, inference_path, dataset='yvis21', instances=[0,1])

