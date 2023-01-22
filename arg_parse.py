import argparse
from util.visualize import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--sgd', action='store_true')

    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=[4, 10], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--with_box_refine', default=True, action='store_true')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the model weights.")
    # Model parameters
    parser.add_argument('--pretrain_weights', type=str, default=None,
                            help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone

    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--rel_coord', default=True, action='store_true')


    # Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_out_stride', default=4, type=int)


    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients

    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)



    # dataset parameters
    parser.add_argument('--ytvis_path', default='../ytvis', type=str)
    parser.add_argument('--img_path', default='../ytvis/val/JPEGImages/')
    parser.add_argument('--ann_path', default='../ytvis/annotations/instances_val_sub.json')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--dataset_file', default='YoutubeVIS')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--dataset_type', default='original')


    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    #parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_frames', default=1, type=int, help='number of frames')
    parser.add_argument('--eval_types', default='')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--cuda_visible_devices", nargs="*", type=int, default=None,
                        help="list of cuda visible devices")

    # visualization related parameters
    parser.add_argument('--save_visualization', default=False, action='store_true', help="save output images during inference")
    parser.add_argument('--exp_name', default='r50_joint', help="experiemnt will also be saved in a folder named as this")
    parser.add_argument('--ckpt_name', default=None, help="evaluate specific ckp")
    parser.add_argument('--inference_comment', default=None, help="identifier for different evaluation strategies")
    parser.add_argument('--online', default=True, help='for online segmentation', action='store_true')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    parser.add_argument('--debug', default=False, action='store_true', help="reduce image resolution and run few iterations")
    parser.add_argument('--upload_file', default=False, action='store_true', help="reduce image resolution and run few iterations")
    parser.add_argument('--competition', default='youtubeVIS19')
    parser.add_argument('--writer', action="store_true", help="for writing in tensorboard")
    parser.add_argument('--jointfinetune', action='store_true',
                            help="keep all weight when load joint training model")
    parser.add_argument('--skip_pretrain_detr_decoder', action='store_true',
                            help="skip the pretrained weights of detr transformer for online training")
    parser.add_argument('--skip_pretrain_detr_temporal_class', action='store_true',
                        help="skip the pretrained weights of detr transformer for online training")

    parser.add_argument('--increase_nf_epochs', type=int, nargs='+', help='increase number of frames at epoch') #[2, 6, 8, 10]
    parser.add_argument('--propagate_ref_points', default=False, action="store_true")
    parser.add_argument('--propagate_inter_ref', default=False, action='store_true')
    parser.add_argument('--propagate_ref_additive', default=False, action="store_true")
    parser.add_argument('--propagate_class', default=False, action='store_true')
    parser.add_argument('--sg_query', default=False, action='store_true', help='stop gradient while propagating query')
    parser.add_argument('--reinitialize_query', default=False, action='store_true', help="randomly initialize bottom-k queries")
    parser.add_argument('--initial_output_dir', default='../instanceformer_output')
    parser.add_argument('--memory_support', default=0, type=int, help='use memory support for multiple frame')
    parser.add_argument('--memory_token', default=0, type=int,help='use memory instance memory token')
    parser.add_argument('--weighted_category', default=False, action='store_true',help='weighted category id')
    parser.add_argument('--save2memory', default=False, action='store_true', help=' save memory to disk')
    parser.add_argument('--propagate_time_ref', default=False, action='store_true', help=' save memory to disk')
    parser.add_argument('--memory_as_query', default=False, action='store_true', help=' maintain a dynamic memeory of token on the go')
    parser.add_argument('--alternate_attn', default=False, action='store_true',help='alternate mem and self attn between frames')
    parser.add_argument('--attn_at', default=3, type=int, choices=[1,2,3], help='attend to memory at first/second/third')
    parser.add_argument('--dynamic_memory', default=False, action='store_true', help='memories are computed on the fly basis from previous frame')
    parser.add_argument('--last_ref', default=False, action="store_true", help='propagate ref point of the last layer as the init ref for next frame')
    parser.add_argument('--prop_box', default=False, action="store_true", help='propagate box centers of the last layer as the init ref for next frame')
    parser.add_argument('--remove_zero_mask', default=False, action="store_true", help='remove zero masks during training')
    parser.add_argument('--init_ref_after_memory', default=False, action="store_true", help='initialize memory after memory attn')
    parser.add_argument('--temp_midx_pos', default=False, action="store_true", help='temporal memory index pos ')
    parser.add_argument('--frame_wise_loss', default=False, action="store_true", help='backprop loss for each frame')
    parser.add_argument('--enc_ref_prop', default=False, action="store_true", help='temporaly propagate ref point for encoder')
    parser.add_argument('--dec_ref_prop', default=False, action="store_true", help='temporaly propagate ref point from decoder to encoder')
    parser.add_argument('--dec_ref_keep_ratio',  type=float, default=0.5,help='how much referance to keef for the dec2enc ref propagation')
    parser.add_argument('--increase_proposals', default=1, type=int, help='increase the number of proposal for inference')
    parser.add_argument('--mask_query', default=False, action="store_true",help='increase the number of proposal for inference')
    parser.add_argument('--long_memory', default=False, action="store_true",help='increase the number of memory for inference')
    parser.add_argument('--enc_aux_loss', default=False, action="store_true", help='as sparse detr')
    parser.add_argument('--supcon', default=False, action="store_true", help='supervised contrastive loss')
    parser.add_argument('--analysis', default=False, action="store_true", help='analysis while inference')
    parser.add_argument('--upload_file', default=False, action='store_true', help='automatically upload results to the server')
    parser.add_argument('--propagate_inter_ref_2', default=False, action='store_true')
    parser.add_argument('--sim_match', default=False, action='store_true')

    return parser
