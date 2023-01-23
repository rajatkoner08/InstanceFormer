# ------------------------------------------------------------------------
# InstanceFormer Inference.
# ------------------------------------------------------------------------

import argparse
import random
import torch
import glob
import util.misc as utils
from models import build_model
import torchvision.transforms as T
import os
from PIL import Image
import torch.nn.functional as F
import json
import pycocotools.mask as mask_util
from pathlib import Path
import zipfile
from arg_parse import get_args_parser
from tqdm import tqdm
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from util.server_process import upload_file

CLASSES = ['person', 'giant_panda', 'lizard', 'parrot', 'skateboard', 'sedan', 'ape',
           'dog', 'snake', 'monkey', 'hand', 'rabbit', 'duck', 'cat', 'cow', 'fish',
           'train', 'horse', 'turtle', 'bear', 'motorbike', 'giraffe', 'leopard',
           'fox', 'deer', 'owl', 'surfboard', 'airplane', 'truck', 'zebra', 'tiger',
           'elephant', 'snowboard', 'boat', 'shark', 'mouse', 'frog', 'eagle', 'earless_seal',
           'tennis_racket']

CLASSES_YVIS_22 = ['airplane', 'bear', 'bird', 'boat', 'car', 'cat', 'cow', 'deer', 'dog', 'duck', 'earless_seal',
                   'elephant', 'fish', 'flying_disc', 'fox', 'frog', 'giant_panda', 'giraffe', 'horse', 'leopard',
                   'lizard', 'monkey', 'motorbike', 'mouse', 'parrot', 'person', 'rabbit', 'shark', 'skateboard',
                   'snake', 'snowboard', 'squirrel', 'surfboard', 'tennis_racket', 'tiger', 'train', 'truck',
                   'turtle', 'whale', 'zebra']

CLASSES_OVIS = ['Person', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe',
                'Poultry', 'Giant_panda', 'Lizard', 'Parrot', 'Monkey', 'Rabbit', 'Tiger', 'Fish', 'Turtle', 'Bicycle',
                'Motorcycle', 'Airplane', 'Boat', 'Vehical']

transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    save_vis = False
    fast_inference = True
    device = torch.device(args.device)
    args.train = False
    print(device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    save_dir = args.initial_save_dir if args.initial_save_dir else args.initial_output_dir

    visualization_path = os.path.join(save_dir, "visualization", args.exp_name)
    Path(visualization_path).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        if args.ckpt_name:
            ckpt_list = [os.path.join(args.initial_output_dir, "experiments", args.exp_name + '/' + args.ckpt_name)]
        else:
            ckpt_list = glob.glob(os.path.join(args.initial_output_dir, "experiments", args.exp_name + '/*.pth'))
            ckpt_list.sort(reverse=True)
            ckpt_list = ckpt_list[:-1]

        videos = json.load(open(args.ann_path, 'rb'))['videos']
        vis_num = len(videos)
        # now run the model for all the checkpoint
        for ckp_id, ckpt in enumerate(ckpt_list):
            ckpt_name = os.path.basename(ckpt).split('.')[-2]
            print("Processing checkpoint : ", ckpt_name)
            # check result dir if already processed
            if args.inference_comment:
                results_path = os.path.join(save_dir, "results", args.exp_name, args.inference_comment)
                analysis_path = os.path.join(args.initial_output_dir, "analysis", args.exp_name, args.inference_comment)
            else:
                results_path = os.path.join(save_dir, "results", args.exp_name)
                analysis_path = os.path.join(args.initial_output_dir, "analysis", args.exp_name)

            if os.path.exists(os.path.join(results_path, args.exp_name + "_" + ckpt_name + ".zip")):
                print("Results are already processed for this checkpoint")
                continue

            Path(analysis_path).mkdir(parents=True, exist_ok=True)
            state_dict = torch.load(ckpt)['model']
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            folder = args.ytvis_path

            result = []
            for i in tqdm(range(vis_num)):
                id_ = videos[i]['id']
                vid_len = videos[i]['length']
                file_names = videos[i]['file_names']
                video_name_len = 8 if 'ovis' in args.dataset_file else 10

                if save_vis: #save vis only for last checkpoint
                    video_path = f"{visualization_path}/{file_names[0][:video_name_len]}"
                    Path(video_path).mkdir(parents=True, exist_ok=True)

                img_set = []
                visualize_img_set = []

                vid_range = min(25, vid_len) if (args.debug and 'ovis' in args.dataset_file) else vid_len
                for k in range(vid_range):
                    im = Image.open(os.path.join(folder, file_names[k]))
                    if save_vis:
                        visualize_img_set.append(np.array(im))
                    img_set.append(transform(im).unsqueeze(0).cuda())

                img = torch.cat(img_set, 0)

                memory_pos = None
                model.num_frames = vid_range

                outputs = model.inference(img, img.shape[-1], img.shape[-2], memory_pos, fast_inference)


                logits = outputs['pred_logits'].squeeze(1)
                output_mask = outputs['pred_masks'].squeeze(2).permute(1, 0, 2, 3)

                scores = logits.sigmoid().cpu().detach().numpy()
                hit_dict = {}
                score_dict = {}
                num_proposals = 25 * args.increase_proposals if 'ovis' in args.dataset_file else 10 * args.increase_proposals
                topkv, indices10 = torch.topk(logits[:3, ...].sigmoid().cpu().detach().flatten(0), k=num_proposals)
                indices10 = indices10.tolist()
                num_classes = 27 if 'ovis' in args.dataset_file else 42


                for idx in indices10:
                    queryid = (idx % (logits.shape[1] * num_classes)) // num_classes
                    if args.weighted_category:
                        topk = torch.topk(logits[:, queryid].sigmoid().cpu().detach().flatten(0), k=(vid_range // 4))
                        topk_classes = topk.indices % num_classes
                        hit_dict[queryid] = [torch.mode(topk_classes).values]
                        score_dict[queryid] = np.mean(
                            [topk.values[i].item() for i, tc in enumerate(topk_classes) if tc == hit_dict[queryid][0]])
                    else:
                        hit_dict[queryid] = [idx % num_classes]

                if save_vis:
                    visualization_masks = [[] for _ in range(vid_range)]
                    labels = [[] for _ in range(vid_range)]
                    conf_scores = [[] for _ in range(vid_range)]

                for inst_id in hit_dict.keys():
                    masks = output_mask[inst_id]
                    pred_masks = F.interpolate(masks[:, None, :, :], (im.size[1], im.size[0]), mode="bilinear")
                    pred_masks = pred_masks.sigmoid().cpu().detach().numpy() > 0.5
                    if pred_masks.max() == 0:
                        print('skip')
                        continue
                    for class_id in hit_dict[inst_id]:
                        category_id = class_id
                        if args.weighted_category:
                            score = score_dict[inst_id]
                        else:
                            score = scores[:, inst_id, class_id][0]
                        instance = {'video_id': id_, 'video_name': file_names[0][:video_name_len],
                                    'score': float(score), 'category_id': int(category_id)}
                        segmentation = []
                        for n in range(vid_range):
                            if not args.weighted_category:
                                score = score[n]
                            if score < 0.001:
                                segmentation.append(None)
                            else:
                                mask = (pred_masks[n, 0]).astype(np.uint8)
                                if save_vis:
                                    visualization_masks[n].append(mask)
                                    labels[n].append(int(category_id))
                                    conf_scores[n].append(score)
                                rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
                                rle["counts"] = rle["counts"].decode("utf-8")
                                segmentation.append(rle)
                        instance['segmentations'] = segmentation
                        result.append(instance)

                if save_vis:
                    num_instances = len(visualization_masks[0])
                    assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
                    for k in range(vid_range):
                        visualizer = Visualizer(visualize_img_set[k], metadata=None, scale=1.0)
                        non_zero_mask = [j for j, vm in enumerate(visualization_masks[k]) if vm.sum() > 2]
                        vis_mask = [visualization_masks[k][i] for i in non_zero_mask]
                        vis_label = [
                            CLASSES_OVIS[labels[k][i] - 1] if 'ovis' in args.dataset_file else CLASSES[labels[k][i] - 1] + (
                                f':{conf_scores[k][i] * 100:.1f}%') for i in non_zero_mask]
                        vis_color = [assigned_colors[i] for i in non_zero_mask]
                        vis = visualizer.overlay_instances(labels=vis_label, masks=vis_mask, assigned_colors=vis_color)
                        vis.save(f"{visualization_path}/{file_names[k]}")

                if args.analysis:
                    analysis_indices = torch.tensor(list(hit_dict.keys()))
                    hs_analysis = outputs['hs_analysis'][:, analysis_indices, ...]
                    ref_analysis = outputs['ref_analysis'][:, :,analysis_indices, ...]
                    init_ref_analysis = outputs['init_ref_analysis'][:, analysis_indices, ...]
                    np.save(f"{analysis_path}/{videos[i]['file_names'][0].split('/')[0]}_hs_analysis.npy", hs_analysis.cpu().detach().numpy())
                    np.save(f"{analysis_path}/{videos[i]['file_names'][0].split('/')[0]}_ref_analysis.npy", ref_analysis.cpu().detach().numpy())
                    np.save(f"{analysis_path}/{videos[i]['file_names'][0].split('/')[0]}_init_ref_analysis.npy", init_ref_analysis.cpu().detach().numpy())

            torch.cuda.empty_cache()
            Path(results_path).mkdir(parents=True, exist_ok=True)
            json_path = f"{results_path}/results.json"
            if args.inference_comment:
                zip_path = f'{results_path}/{args.exp_name + "_" + args.inference_comment + "_" + ckpt_name}.zip'
            else:
                zip_path = f'{results_path}/{args.exp_name + "_" + ckpt_name}.zip'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f)
            with zipfile.ZipFile(zip_path, 'w',
                                 compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=9) as zf:
                zf.write(json_path, arcname='results.json')
            os.remove(json_path)
            if args.upload_file:
                upload_file(zip_path, args.competition)


if __name__ == '__main__':
    print("Initi main")
    parser = argparse.ArgumentParser(' inference script', parents=[get_args_parser()])
    print("Initi parse")
    args = parser.parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_devices))
    main(args)
