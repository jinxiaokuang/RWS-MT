import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))  # 相对路径

from Data_Preprocessing.Data_Augmentation import get_default_augmentation, default_3D_augmentation_params
from Loss.Dice_loss import SoftDiceLoss, SoftDiceLossSquared, DC_and_CE_loss
from Models.Baseline.UNet import Generic_UNet
from Data_Preprocessing.Data_Reader_CADA import get_labeled_data
from Data_Preprocessing.Data_Generator import DataGenerator3D
from Data_Preprocessing.Data_Augmentation import get_default_augmentation
from Data_Preprocessing.Data_Utils import split_data
from Models.Model_Utils import softmax_helper, InitWeights_He
from Main.Utils_Train.Utils_Train import poly_lr, get_current_consistency_weight, update_ema_variables
from torch.optim import lr_scheduler
from Main.Utils_Train.Utils_Train import validation, print_log, save_checkpoint, pad_img_to_fit_network, infer_and_save, sigmoid_rampup
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
    parser.add_argument('--full_training', type=int, default=0,
                        help='whether to use all samples to train', )

    parser.add_argument('--vendor', type=str, default='A',
                        help='where the dataset comes from', )
    parser.add_argument('--patch_size_x', type=int, default=128,
                        help='training patch size x', )
    parser.add_argument('--patch_size_y', type=int, default=128,
                        help='training patch size y', )
    parser.add_argument('--patch_size_z', type=int, default=128,
                        help='training patch size z', )
    parser.add_argument('--batch_size', type=int, default=1,
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
    parser.add_argument('--batches_of_epoch', type=int, default=150,
                        help='iterations in an epoch', )
    parser.add_argument('--epoches', type=int, default=80,
                        help='training epoches in total', )

    parser.add_argument('--nonlin', type=int, default=2,
                        help='1:ReLU, 2: LReLU', )
    parser.add_argument('--norm_op', type=int, default=2,
                        help='1:InstanceNorm, 2: BatchNorm', )
    parser.add_argument('--log_path', type=str, default='logs/UA-MT_0719_w1/fold{}lr{}',
                        help='log path', )
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/UA-MT_0719_w1/fold{}lr{}',
                        help='checkpoint path', )
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint-{}.pth',
                        help='checkpoint name', )
    parser.add_argument('--summary_writer', type=str, default='logs/UA-MT_0717_w1/fold{}lr{}',
                        help='checkpoint name', )
    # parser.add_argument('--log_path', type=str, default='logs/Baseline_UNet/fold{}',
    #                     help='log path', )
    # parser.add_argument('--model_path', type=str, default='', help='model path')
    # parser.add_argument('--checkpoint_path', type=str, default='checkpoints/Baseline_UNet/fold{}',
    #                     help='checkpoint path', )
    parser.add_argument('--checkpoint_name1', type=str, default='checkpoint1-{}.pth',
                        help='checkpoint name', )
    parser.add_argument('--checkpoint_name2', type=str, default='checkpoint2-{}.pth',
                        help='checkpoint name', )
    # parser.add_argument('--summary_writer', type=str, default='logs/Baseline_UNet/fold{}',
    #                     help='checkpoint name', )
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
    parser.add_argument('--differ_data', type=int, default=0, help='input different dataset into the two mutual models')
    parser.add_argument('--update_ema_interval', type=int, default=50)
    
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
CHECKPOINT_NAME1 = args.checkpoint_name1
CHECKPOINT_NAME2 = args.checkpoint_name2
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
now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
log_path = LOG_PATH
if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
log = open(os.path.join(log_path, 'log_fold{}_{}.txt'.format(FOLD, now)), 'w')
print_log('save path : {}'.format(log_path), log)

# initialize summarywriter
writer = SummaryWriter(SUMMARY_WRITER)

# read data
train_dataset, test_dataset = get_labeled_data(norm=True, one_hot=True)
# val_dataset = get_labeled_data(norm=True, one_hot=True)
# n-fold or full
if FULL_TRAINING:
    train_dataset = dict(train_dataset.items() + test_dataset.items())
    val_dataset = train_dataset
    test_dataset = train_dataset
else:
    splits = split_data(list(train_dataset.keys()), K=K_FOLD, shuffle=K_FOLD_SHUFFLE)
    train_dataset_n_fold = {k:v for k,v in train_dataset.items() if k in splits[FOLD]['train']}
    val_dataset_n_fold = {k:v for k,v in train_dataset.items() if k in splits[FOLD]['val']}
    train_dataset = train_dataset_n_fold
    val_dataset = val_dataset_n_fold
    # random.seed(2333)
    # all_keys = list(train_dataset.keys())
    # random.shuffle(all_keys)
    # train_ratio = 0.5
    # train_keys = all_keys[:int(len(all_keys)*train_ratio)]
    # val_keys = all_keys[int(len(all_keys)*train_ratio):]
    # train_dataset_new = {k:v for k,v in train_dataset.items() if k in train_keys}
    # val_dataset_new = {k:v for k,v in train_dataset.items() if k in val_keys}
    # train_dataset = train_dataset_new
    # val_dataset = val_dataset_new
    # random.seed(2333)
    # all_keys = list(train_dataset.keys())
    # random.shuffle(all_keys)
    # train_ratio = 0.2
    # train_keys = all_keys[:int(len(all_keys)*train_ratio)]
    # val_keys = all_keys[int(len(all_keys)*train_ratio):]
    # train_dataset_new = {k:v for k,v in train_dataset.items() if k in train_keys}
    # val_dataset_new = {k:v for k,v in train_dataset.items() if k in val_keys}
    # train_dataset = train_dataset_new
    # val_dataset = val_dataset_new

print('train set: {}'.format(len(train_dataset.keys())))
print('val set: {}'.format(len(val_dataset.keys())))
print('test set: {}'.format(len(test_dataset.keys())))

# print('dasdasdf:',test_dataset.lens)


# train_dataset_1_keys = random.sample(train_dataset.keys(), int(len(train_dataset.keys())*0.9))
# train_dataset_2_keys = random.sample(train_dataset.keys(), int(len(train_dataset.keys())*0.9))
# train_dataset_1 = {k:v for k,v in train_dataset.items() if k in train_dataset_1_keys}
# train_dataset_2 = {k:v for k,v in train_dataset.items() if k in train_dataset_2_keys}
train_dataset_1 = train_dataset

src_train_loader1 = DataGenerator3D(train_dataset_1, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
src_train_gen1, _ = get_default_augmentation(src_train_loader1, None, PATCH_SIZE, params=default_3D_augmentation_params)

tgt_train_dataset = val_dataset
tgt_train_loader = DataGenerator3D(tgt_train_dataset, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
tgt_train_gen, _ = get_default_augmentation(tgt_train_loader, None, PATCH_SIZE, params=default_3D_augmentation_params)

# train_loader = DataGenerator3D(train_dataset, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
# # train_loader.generate_train_batch()
# train_gen, _ = get_default_augmentation(train_loader, None, PATCH_SIZE, params=default_3D_augmentation_params)

# define model
model1 = Generic_UNet(INPUT_CHANNELS, BASE_NUM_FEATURES, NUM_CLASSES, NUM_POOL, num_conv_per_stage=2,
                     feat_map_mul_on_downscale=2, conv_op=conv_op,
                     norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                     dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                     nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs, deep_supervision=False, dropout_in_localization=False,
                     final_nonlin=final_nonlin, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                     conv_kernel_sizes=None,
                     upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=True)
ema_model1 = Generic_UNet(INPUT_CHANNELS, BASE_NUM_FEATURES, NUM_CLASSES, NUM_POOL, num_conv_per_stage=2,
                     feat_map_mul_on_downscale=2, conv_op=conv_op,
                     norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                     dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                     nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs, deep_supervision=False, dropout_in_localization=False,
                     final_nonlin=final_nonlin, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                     conv_kernel_sizes=None,
                     upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=True)
# checkpoint1 = torch.load('/home/hci003/work/projects/cada-as/Code_007/checkpoints/Baseline_0706/fold0/model_best1.pth')
# model1.load_state_dict(checkpoint1['model1'])
# ema_model1.load_state_dict(checkpoint1['model1'])
for param in ema_model1.parameters():
    param.detach_()

# print(model)
if torch.cuda.is_available():
    model1.cuda()
    ema_model1.cuda()

# define loss func
criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-8)
consistency_criterion = criterion

# define optimizer
optimizer1 = torch.optim.Adam(model1.parameters(), INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
scheduler1 = lr_scheduler.StepLR(optimizer1, LR_STEP_SIZE, LR_GAMMA)

best_dice1 = 0.
# infer ema_model on labelled data
ema_mid_feat_dict, ema_pred_dict = infer_and_save(ema_model1, train_dataset_1)
# training
for epoch in range(EPOCHES):
    model1.train()
    ema_model1.train()
    for iter in range(BATCHES_OF_EPOCH):
        # loading data
        src_train_batch1 = next(src_train_gen1)
        src_train_img1 = src_train_batch1['data']
        src_train_label1 = src_train_batch1['target']

        if not isinstance(src_train_img1, torch.Tensor):
            src_train_img1 = torch.from_numpy(src_train_img1).float()
        if not isinstance(src_train_label1, torch.Tensor):
            src_train_label1 = torch.from_numpy(src_train_label1).float()

        tgt_train_batch = next(tgt_train_gen)
        tgt_train_img = tgt_train_batch['data']
        tgt_train_label = tgt_train_batch['target']

        if not isinstance(tgt_train_img, torch.Tensor):
            tgt_train_img = torch.from_numpy(tgt_train_img).float()
        if not isinstance(tgt_train_label, torch.Tensor):
            tgt_train_label = torch.from_numpy(tgt_train_label).float()

        if torch.cuda.is_available():
            src_train_img1 = src_train_img1.cuda(non_blocking=True)
            src_train_label1 = src_train_label1.cuda(non_blocking=True)
            tgt_train_img = tgt_train_img.cuda(non_blocking=True)
            tgt_train_label = tgt_train_label.cuda(non_blocking=True)
            
        inputs1 = torch.cat((src_train_img1, tgt_train_img), dim=0)
        noise = torch.clamp(torch.randn_like(tgt_train_img) * 0.1, -0.2, 0.2)
        outputs1, _ = model1(inputs1)
        ema_inputs = tgt_train_img + noise
        with torch.no_grad():
            ema_output1 = ema_model1(ema_inputs)
            print('eam_output.shape', ema_output1.shape)
        T = 8
        tgt_train_img_r = tgt_train_img.repeat(2, 1, 1, 1, 1)
        stride = tgt_train_img_r.shape[0] // 2
        preds = torch.zeros([stride * T, 2, 128, 128, 128])
        for i in range(T//2):
            ema_inputs = tgt_train_img_r + torch.clamp(torch.randn_like(tgt_train_img_r) * 0.1, -0.2, 0.2)
            with torch.no_grad():
                print("ema_input_1.shape", ema_model1(ema_inputs).shape)
                print("preds.shape",  preds[2 * stride * i:2 * stride * (i + 1)].shape)
                preds[2 * stride * i:2 * stride * (i + 1)] = ema_model1(ema_inputs)
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = preds.reshape(T, stride, 2, 128, 128, )
        preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
        uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)

        ## calculate the loss
        supervised_loss1 = torch.nn.functional.cross_entropy(outputs1[: BATCH_SIZE], src_train_label1)

        consistency_weight = get_current_consistency_weight(iter//150)
        consistency_dist = consistency_criterion(outputs1[BATCH_SIZE:], ema_output1) #(batch, 2, 112,112,80)
        threshold = (0.75+0.25*sigmoid_rampup(iter, 600))*np.log(2)
        mask = (uncertainty<threshold).float()
        consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
        consistency_loss1 = consistency_weight * consistency_dist
        loss1 = supervised_loss1 + consistency_loss1

        pseudo_label1 = torch.argmax(softmax_helper(ema_output1), dim=1).long()



        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        scheduler1.step(epoch)

        total_iters = epoch * BATCHES_OF_EPOCH + iter
        if total_iters % args.update_ema_interval == 0 and total_iters > 0: 
            print('update ema')
            update_ema_variables(model1, ema_model1, args.ema_decay, total_iters)
            # infer ema_model on labelled data
            ema_mid_feat_dict, ema_pred_dict = infer_and_save(ema_model1, train_dataset_1)
        
        # update_ema_variables(model2, ema_model2, args.ema_decay, epoch * BATCHES_OF_EPOCH + iter)

        writer.add_scalar("supervised_loss1/Dice", supervised_loss1.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.add_scalar("consistency_loss1", consistency_loss1.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        
        writer.flush()

        log_str = 'epoch: {}, iter: {}, lr: {}, sup_loss1: {}, con_loss1: {}, con_weight: {}'\
            .format(epoch, iter, scheduler1.get_lr()[0], supervised_loss1.data.item(),
                    consistency_loss1.data.item(), consistency_weight)
        print_log(log_str, log)
    
    # update_ema_variables(model1, ema_model1, args.ema_decay, epoch * BATCHES_OF_EPOCH + iter)

    # evaluation
    with torch.no_grad():
        model1.eval()
        mean_dice1, dices1 = validation(model1, test_dataset, NUM_POOL)
        log_str = 'val: epoch: {}, mean_dice1: {}, dices1: {}'\
            .format(epoch, mean_dice1, dices1)
        print_log(log_str, log)
        writer.add_scalar("val acc1/Dice", mean_dice1, epoch * BATCHES_OF_EPOCH)
        writer.flush()

        print_log("saving checkpoint...", log)
        is_best1 = False
        if mean_dice1 > best_dice1:
            best_dice1 = mean_dice1
            is_best1 = True
            log_str = 'best dice1: mean_dice1: {}'.format(mean_dice1)
            print_log(log_str, log)
        
        state1 = {
            'epoch1': epoch,
            'best_epoch1': best_dice1,
            'model1': model1.state_dict(),
            'ema_model1': ema_model1.state_dict(),
            'optimizer1': optimizer1.state_dict(),
        }
        # import pdb;pdb.set_trace()
        save_checkpoint(state1, is_best1, CHECKPOINT_PATH, CHECKPOINT_NAME1.format(epoch), 'UA-MT_model_best.pth')
        

mean_dice, dices = validation(model1, test_dataset, NUM_POOL)
print("final validation on test set: mean_dice: {}".format(mean_dice))
writer.close()
log.close()


