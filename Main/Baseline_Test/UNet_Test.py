from Data_Augmentation import get_default_augmentation, default_3D_augmentation_params
from Loss.Dice_loss import SoftDiceLoss, SoftDiceLossSquared, DC_and_CE_loss
from Baseline.UNet import Generic_UNet
from Data_Reader_CADA import get_labeled_data
from Data_Generator import DataGenerator3D
from Data_Augmentation import get_default_augmentation
from Data_Preprocessing.Data_Utils import split_data
from Models.Model_Utils import softmax_helper, InitWeights_He
from Utils_Train.Utils_Train import poly_lr, generate_img_patches, generate_d_map_patch2Img
from Utils.Metrics import dice
from torch.optim import lr_scheduler
from Utils_Train.Utils_Train import validation, print_log, save_checkpoint, pad_img_to_fit_network
from Data_Utils import save_nii
from batchgenerators.augmentations.utils import random_crop_3D_image_batched, pad_nd_image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import os
import argparse
import random

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for UNet training")
    parser.add_argument('--fold', type=int, default=0,
                        help='cross validation fold No.', )
    parser.add_argument('--k_fold', type=int, default=5,
                        help='cross validation: number of folds.', )
    parser.add_argument('--k_fold_shuffle', type=int, default=1,
                        help='whether shuffle data list before split dataset.', )
    parser.add_argument('--full_training', type=int, default=1,
                        help='whether to use all samples to train', )

    parser.add_argument('--vendor', type=str, default='A',
                        help='where the dataset comes from', )
    parser.add_argument('--patch_size_x', type=int, default=128,
                        help='training patch size x', )
    parser.add_argument('--patch_size_y', type=int, default=128,
                        help='training patch size y', )
    parser.add_argument('--patch_size_z', type=int, default=128,
                        help='training patch size z', )
    parser.add_argument('--batch_size', type=int, default=2,
                        help='training batch size', )
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of targets', )
    parser.add_argument('--input_channel', type=int, default=1,
                        help='number of channels of input data', )
    parser.add_argument('--base_num_features', type=int, default=16,
                        help='number of features in the first stage of UNet', )
    parser.add_argument('--max_filters', type=int, default=512,
                        help='max number of features in UNet', )
    parser.add_argument('--num_pool', type=int, default=5,
                        help='number of pool ops in UNet', )

    parser.add_argument('--initial_lr', type=float, default=1e-4,
                        help='initial learning rate', )
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='lr * lr_gamma every lr_step_size', )
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='lr * lr_gamma every lr_step_size', )
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='used in optimizer', )
    parser.add_argument('--batches_of_epoch', type=int, default=250,
                        help='iterations in an epoch', )
    parser.add_argument('--epoches', type=int, default=20, #100,
                        help='training epoches in total', )

    parser.add_argument('--nonlin', type=int, default=2,
                        help='1:ReLU, 2: LReLU', )
    parser.add_argument('--norm_op', type=int, default=1,
                        help='1:InstanceNorm, 2: BatchNorm', )

    parser.add_argument('--log_path', type=str, default='logs/Baseline_0717_w1/fold{}lr{}',
                        help='log path', )
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/Baseline_0717_w1/fold{}lr{}',
                        help='checkpoint path', )
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint-{}.pth',
                        help='checkpoint name', )
    parser.add_argument('--summary_writer', type=str, default='logs/Baseline_UNet/fold{}lr{}',
                        help='checkpoint name', )
    parser.add_argument('--model_path', type=str, default='../../checkpoints/Baseline_0719_w1/fold0deleted_and_addData/model_best1.pth',
                        help='checkpoint name', )                   
    return parser.parse_args()


np.random.seed(123)
torch.manual_seed(1234)
if torch.cuda.is_available():
    print('torch.cuda.is_available()')
    torch.cuda.manual_seed_all(123456)
cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


args = get_arguments()

# cross validation
FOLD = args.fold
K_FOLD = args.k_fold
K_FOLD_SHUFFLE = True if args.k_fold_shuffle == 1 else False
FULL_TRAINING = True if args.full_training == 1 else False

VENDOR = args.vendor
PATCH_SIZE = (args.patch_size_x, args.patch_size_y, args.patch_size_z)
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
INPUT_CHANNELS = args.input_channel
BASE_NUM_FEATURES = args.base_num_features
MAX_FILTERS = args.max_filters
NUM_POOL = args.num_pool

INITIAL_LR = args.initial_lr
LR_STEP_SIZE = args.lr_step_size
LR_GAMMA = args.lr_gamma
WEIGHT_DECAY = args.weight_decay
BATCHES_OF_EPOCH = args.batches_of_epoch
EPOCHES = args.epoches

LOG_PATH = args.log_path.format(args.fold,args.initial_lr)
CHECKPOINT_PATH = args.checkpoint_path.format(args.fold,args.initial_lr)
CHECKPOINT_NAME = args.checkpoint_name
SUMMARY_WRITER = args.summary_writer.format(args.fold,args.initial_lr)

NONLIN = args.nonlin
NORM_OP = args.norm_op

conv_op = nn.Conv3d
dropout_op = nn.Dropout3d

if NORM_OP == 1:
    norm_op = nn.InstanceNorm3d
elif NORM_OP == 2:
    norm_op = nn.BatchNorm3d
else:
    raise Exception('Norm_OP Invalid!')

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}

if NONLIN == 1:
    net_nonlin = nn.ReLU
elif NONLIN == 2:
    net_nonlin = nn.LeakyReLU
else:
    raise Exception('nonlin invalid!')

net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
final_nonlin = lambda x: x

# initialize log

# read data
train_dataset, test_dataset = get_labeled_data(norm=True, one_hot=True)
# val_dataset = get_labeled_data(norm=True, one_hot=True)
# n-fold or full
if FULL_TRAINING:
    train_dataset = dict(list(train_dataset.items()) + list(test_dataset.items()))
    val_dataset = train_dataset
    test_dataset = train_dataset
else:
    splits = split_data(list(train_dataset.keys()), K=K_FOLD, shuffle=K_FOLD_SHUFFLE)
    train_dataset_n_fold = {k:v for k,v in train_dataset.items() if k in splits[FOLD]['train']}
    val_dataset_n_fold = {k:v for k,v in train_dataset.items() if k in splits[FOLD]['val']}
    train_dataset = train_dataset_n_fold
    val_dataset = val_dataset_n_fold

# train_loader = DataGenerator3D(train_dataset, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
# train_loader.generate_train_batch()

# train_gen, _ = get_default_augmentation(train_loader, None, PATCH_SIZE, params=default_3D_augmentation_params)

# define model
model = Generic_UNet(INPUT_CHANNELS, BASE_NUM_FEATURES, NUM_CLASSES, NUM_POOL, num_conv_per_stage=2,
                     feat_map_mul_on_downscale=2, conv_op=conv_op,
                     norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                     dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                     nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs, deep_supervision=False, dropout_in_localization=False,
                     final_nonlin=final_nonlin, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                     conv_kernel_sizes=None,
                     upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=True)
print(model)
if args.model_path:
    print('load path: {}'.format(args.model_path))
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model1'])

if torch.cuda.is_available():
    model.cuda()

save_path = '{}.test_results'.format(args.model_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
dices = []
model.eval()
result_log = open(os.path.join(LOG_PATH, 'test_result.txt'),'w')


with torch.no_grad():
    for key in test_dataset.keys():
        img_ed = test_dataset[key]['img']     # (13, 216, 256)
        img_ed_gt = test_dataset[key]['gt']   # (13, 4, 216, 256)
        
        # input_data = np.expand_dims(img_ed, axis=0)
        # input_data = pad_nd_image(input_data, (224,256,256))
        # input_data = np.expand_dims(input_data, axis=0)
        # output = softmax_helper(model(torch.from_numpy(input_data).float().cuda())).detach().cpu().numpy()  
        # pred_map = output.squeeze().argmax(0)
        
        # if img_ed_gt.shape[1] == 220:
        #     pred_map = pred_map[2:-2, :, :]
        # elif img_ed_gt.shape[1] == 218:
        #     pred_map = pred_map[3:-3, :, :]
        # import pdb;pdb.set_trace()
        
        # img_es = dataset[pat]['ES']
        # img_es_gt = dataset[pat]['ES_GT']
        patch_size = (128, 128, 128)
        ita = 2
        # crop image
        img_ed = np.expand_dims(img_ed, axis=0)
        # if len(test_dataset[key]['center']) > 1:
        #     continue
        center_pt = test_dataset[key]['center'][0]#random.choice(test_dataset[key]['center'])
        lt_x = int(max(0, center_pt[0] - patch_size[0]/2))
        lt_y = int(max(0, center_pt[1] - patch_size[1]/2))
        lt_s = int(max(0, center_pt[2] - patch_size[2]/2))
        rb_x = int(min(img_ed.shape[2], lt_x + patch_size[0]))
        rb_y = int(min(img_ed.shape[3], lt_y + patch_size[1]))
        rb_s = int(min(img_ed.shape[1], lt_s + patch_size[2]))
        crop_img = np.zeros((1, 128, 128, 128))
        crop_img[:, :rb_s-lt_s, :rb_x-lt_x, :rb_y-lt_y] = img_ed[:, lt_s:rb_s, lt_x:rb_x, lt_y:rb_y]  # in case that crop length < 128
        
        crop_img_pad = pad_nd_image(crop_img, patch_size)
        
        crop_img_pad = crop_img
        input_data = np.expand_dims(crop_img_pad, axis=0)
        seg_output, mid_feat_output = model(torch.from_numpy(input_data).float().cuda())
        crop_output = softmax_helper(seg_output).detach().cpu().numpy()  
        patch_pred = crop_output.squeeze().argmax(0)
        pred_map = np.zeros(img_ed.shape[1:], dtype=np.int64)
        pred_map[lt_s:rb_s, lt_x:rb_x, lt_y:rb_y] = patch_pred[:rb_s-lt_s, :rb_x-lt_x, :rb_y-lt_y]

        # crop_data_list, ss_c, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z = generate_img_patches(img_ed, patch_size, ita)

        # pred_list = []
        # for crop_data in crop_data_list:
        #     input_data = np.expand_dims(crop_data, axis=0)
        #     padded_output_ed = softmax_helper(model(torch.from_numpy(input_data).float().cuda())).detach().cpu().numpy()  
        #     patch_pred = padded_output_ed.squeeze().argmax(0)
        #     pred_list.append(patch_pred)
        
        # pred_map = generate_d_map_patch2Img(pred_list, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z, patch_size, ita)
        
        dice_cof = dice(pred_map, img_ed_gt.argmax(0))
        pred_map = np.transpose(pred_map, (1, 2, 0))  
        # save_nii(os.path.join(save_path, '{}_{}.nii.gz'.format(key, dice_cof)), pred_map, test_dataset[key]['nii_'][0], test_dataset[key]['nii_'][1])
        print(dice_cof)
        dices.append(dice_cof)
    
    for i,(key,dice) in enumerate(zip(test_dataset.keys(),dices)):
        print("key",key,"dice",dice)
        # print()
        result_log.write("key:{},dice:{}\n".format(key,dice))
        result_log.flush()
    
print('mean dice: {}'.format(np.mean(dices)))


