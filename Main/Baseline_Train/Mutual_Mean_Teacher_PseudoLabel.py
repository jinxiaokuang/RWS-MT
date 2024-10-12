from Data_Augmentation import get_default_augmentation, default_2D_augmentation_params
from Loss.Dice_loss import SoftDiceLoss, SoftDiceLossSquared, DC_and_CE_loss
from Loss.Consistency_loss import softmax_kl_loss, softmax_mse_loss
from Baseline.UNet import Generic_UNet
from Data_Reader import get_labeled_data, get_labeled_data_by_patient_list, get_data_description_by_vendor
from Data_Generator import DataGenerator2D
from Data_Augmentation import get_default_augmentation
from Data_Preprocessing.Data_Utils import split_data
from Models.Model_Utils import softmax_helper, InitWeights_He
from Utils_Train.Utils_Train import poly_lr, get_current_consistency_weight, sigmoid_rampup, update_ema_variables
from torch.optim import lr_scheduler
from Utils_Train.Utils_Train import validation, print_log, save_checkpoint_v2, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import os
import argparse


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for UNet training")
    parser.add_argument('--fold', type=int, default=0,
                        help='cross validation fold No.', )
    parser.add_argument('--k_fold', type=int, default=5,
                        help='cross validation: number of folds.', )
    parser.add_argument('--k_fold_shuffle', type=int, default=0,
                        help='whether shuffle data list before split dataset.', )
    parser.add_argument('--full_training', type=int, default=0,
                        help='whether to use all samples to train', )

    parser.add_argument('--vendor_src', type=str, default='A',
                        help='where the source dataset comes from', )
    parser.add_argument('--vendor_tgt', type=str, default='B',
                        help='where the target dataset comes from', )
    parser.add_argument('--patch_size_h', type=int, default=224,
                        help='training patch size height', )
    parser.add_argument('--patch_size_w', type=int, default=224,
                        help='training patch size width', )
    parser.add_argument('--batch_size', type=int, default=48,
                        help='training batch size', )
    parser.add_argument('--num_classes', type=int, default=4,
                        help='number of targets', )
    parser.add_argument('--input_channel', type=int, default=1,
                        help='number of channels of input data', )
    parser.add_argument('--base_num_features', type=int, default=32,
                        help='number of features in the first stage of UNet', )
    parser.add_argument('--max_filters', type=int, default=512,
                        help='max number of features in UNet', )
    parser.add_argument('--num_pool', type=int, default=5,
                        help='number of pool ops in UNet', )

    parser.add_argument('--initial_lr', type=float, default=1e-3,
                        help='initial learning rate', )
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='lr * lr_gamma every lr_step_size', )
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='lr * lr_gamma every lr_step_size', )
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='used in optimizer', )
    parser.add_argument('--batches_of_epoch', type=int, default=250,
                        help='iterations in an epoch', )
    parser.add_argument('--epoches', type=int, default=100,
                        help='training epoches in total', )

    parser.add_argument('--nonlin', type=int, default=2,
                        help='1:ReLU, 2: LReLU', )
    parser.add_argument('--norm_op', type=int, default=1,
                        help='1:InstanceNorm, 2: BatchNorm', )

    parser.add_argument('--log_path', type=str, default='../../logs/Baseline_UNet/{}/fold{}',
                        help='log path', )
    parser.add_argument('--checkpoint_path', type=str, default='../../checkpoints/Baseline_UNet/{}/fold{}',
                        help='checkpoint path', )
    parser.add_argument('--checkpoint_name1', type=str, default='checkpoint1-{}-{}.pth',
                        help='checkpoint name', )
    parser.add_argument('--checkpoint_name2', type=str, default='checkpoint2-{}-{}.pth',
                        help='checkpoint name', )
    parser.add_argument('--summary_writer', type=str, default='../../logs/Baseline_UNet/{}/fold{}',
                        help='checkpoint name', )

    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
    parser.add_argument('--differ_data', type=int, default=0, help='input different dataset into the two mutual models')
    parser.add_argument('--pretrained_model1', type=str, default="checkpoint.pth", help='pretrained model')
    parser.add_argument('--pretrained_model2', type=str, default="checkpoint.pth", help='pretrained model')
    return parser.parse_args()


np.random.seed(12345)
torch.manual_seed(12345)
if torch.cuda.is_available():
    print('torch.cuda.is_available()')
    torch.cuda.manual_seed_all(12345)
cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


args = get_arguments()

# cross validation
FOLD = args.fold
K_FOLD = args.k_fold
K_FOLD_SHUFFLE = True if args.k_fold_shuffle == 1 else False
FULL_TRAINING = True if args.full_training == 1 else False

VENDOR_SRC = args.vendor_src
VENDOR_TGT = args.vendor_tgt
PATCH_SIZE = (args.patch_size_h, args.patch_size_w)
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

LOG_PATH = args.log_path
CHECKPOINT_PATH = args.checkpoint_path
CHECKPOINT_NAME1 = args.checkpoint_name1
CHECKPOINT_NAME2 = args.checkpoint_name2
SUMMARY_WRITER = args.summary_writer

NONLIN = args.nonlin
NORM_OP = args.norm_op

DIFFER_DATA = args.differ_data

PRETRAINED_MODEL1 = args.pretrained_model1
PRETRAINED_MODEL2 = args.pretrained_model2

conv_op = nn.Conv2d
dropout_op = nn.Dropout2d

if NORM_OP == 1:
    norm_op = nn.InstanceNorm2d
elif NORM_OP == 2:
    norm_op = nn.BatchNorm2d
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

# split data
src_data_description = get_data_description_by_vendor(VENDOR_SRC)
src_train_pats = list(src_data_description.keys())
cnt_src_train_pats = len(src_train_pats)

tgt_data_description = get_data_description_by_vendor(VENDOR_TGT)
tgt_train_pats = list(tgt_data_description.keys())

if DIFFER_DATA == 0:
    src_train_pats1 = src_train_pats
    src_train_pats2 = src_train_pats
elif DIFFER_DATA == 1:
    src_train_pats1 = src_train_pats[:(cnt_src_train_pats // 2)]
    src_train_pats2 = src_train_pats[(cnt_src_train_pats // 2):]

# read data
src_train_dataset1 = get_labeled_data_by_patient_list(vendor=VENDOR_SRC, patient_list=src_train_pats1, norm=True, one_hot=True)
src_train_dataset2 = get_labeled_data_by_patient_list(vendor=VENDOR_SRC, patient_list=src_train_pats2, norm=True, one_hot=True)
tgt_train_dataset = get_labeled_data_by_patient_list(vendor=VENDOR_TGT, patient_list=tgt_train_pats, norm=True, one_hot=True)

src_train_loader1 = DataGenerator2D(src_train_dataset1, vendor=None, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES)
src_train_gen1, _ = get_default_augmentation(src_train_loader1, None, PATCH_SIZE, params=default_2D_augmentation_params)

src_train_loader2 = DataGenerator2D(src_train_dataset2, vendor=None, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES)
src_train_gen2, _ = get_default_augmentation(src_train_loader2, None, PATCH_SIZE, params=default_2D_augmentation_params)

tgt_train_loader = DataGenerator2D(tgt_train_dataset, vendor=None, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES)
tgt_train_gen, _ = get_default_augmentation(tgt_train_loader, None, PATCH_SIZE, params=default_2D_augmentation_params)


# define model
def create_model(ema=False, pretrained_model=None):
    model = Generic_UNet(INPUT_CHANNELS, BASE_NUM_FEATURES, NUM_CLASSES, NUM_POOL, num_conv_per_stage=2,
                         feat_map_mul_on_downscale=2, conv_op=conv_op,
                         norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                         dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                         nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs, deep_supervision=False, dropout_in_localization=False,
                         final_nonlin=final_nonlin, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                         conv_kernel_sizes=None,
                         upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=True)
    if pretrained_model:
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model'])
    if torch.cuda.is_available():
        model.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model1 = create_model(False, pretrained_model=None)
ema_model1 = create_model(True, pretrained_model=None)
model2 = create_model(False, pretrained_model=None)
ema_model2 = create_model(True, pretrained_model=None)

# define loss func
criterion = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, do_bg=False, smooth=1e-8)
if args.consistency_type == 'mse':
    consistency_criterion = softmax_mse_loss
elif args.consistency_type == 'kl':
    consistency_criterion = softmax_kl_loss
elif args.consistency_type == 'ce':
    consistency_criterion = torch.nn.CrossEntropyLoss()
elif args.consistency_type == 'dice':
    consistency_criterion = criterion
else:
    assert False, args.consistency_type

# define optimizer
optimizer1 = torch.optim.Adam(model1.parameters(), INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
scheduler1 = lr_scheduler.StepLR(optimizer1, LR_STEP_SIZE, LR_GAMMA)
optimizer2 = torch.optim.Adam(model2.parameters(), INITIAL_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
scheduler2 = lr_scheduler.StepLR(optimizer2, LR_STEP_SIZE, LR_GAMMA)

best_dice1 = 0.
best_dice2 = 0.
# training
for epoch in range(EPOCHES):
    model1.train()
    ema_model1.train()
    model2.train()
    ema_model2.train()
    for iter in range(BATCHES_OF_EPOCH):
        # loading data
        src_train_batch1 = next(src_train_gen1)
        src_train_img1 = src_train_batch1['data']
        src_train_label1 = src_train_batch1['target']

        if not isinstance(src_train_img1, torch.Tensor):
            src_train_img1 = torch.from_numpy(src_train_img1).float()
        if not isinstance(src_train_label1, torch.Tensor):
            src_train_label1 = torch.from_numpy(src_train_label1).float()

        src_train_batch2 = next(src_train_gen2)
        src_train_img2 = src_train_batch2['data']
        src_train_label2 = src_train_batch2['target']

        if not isinstance(src_train_img2, torch.Tensor):
            src_train_img2 = torch.from_numpy(src_train_img2).float()
        if not isinstance(src_train_label2, torch.Tensor):
            src_train_label2 = torch.from_numpy(src_train_label2).float()

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
            src_train_img2 = src_train_img2.cuda(non_blocking=True)
            src_train_label2 = src_train_label2.cuda(non_blocking=True)
            tgt_train_img = tgt_train_img.cuda(non_blocking=True)
            tgt_train_label = tgt_train_label.cuda(non_blocking=True)

        inputs1 = torch.cat((src_train_img1, tgt_train_img), dim=0)
        inputs2 = torch.cat((src_train_img2, tgt_train_img), dim=0)
        noise = torch.clamp(torch.randn_like(tgt_train_img) * 0.1, -0.2, 0.2)
        ema_inputs = tgt_train_img + noise
        outputs1 = model1(inputs1)
        outputs2 = model2(inputs2)
        with torch.no_grad():
            ema_output1 = ema_model1(ema_inputs)
            ema_output2 = ema_model2(ema_inputs)
        supervised_loss1 = criterion(outputs1[: BATCH_SIZE], src_train_label1)
        supervised_loss2 = criterion(outputs2[: BATCH_SIZE], src_train_label2)

        consistency_weight = get_current_consistency_weight(epoch, args)
        # consistency_dist1 = consistency_criterion(outputs1[BATCH_SIZE:], ema_output2)
        # consistency_loss1 = consistency_weight * consistency_dist1
        # consistency_dist2 = consistency_criterion(outputs2[BATCH_SIZE:], ema_output1)
        # consistency_loss2 = consistency_weight * consistency_dist2

        pseudo_label1 = torch.argmax(softmax_helper(ema_output2), dim=1).long()
        pseudo_label2 = torch.argmax(softmax_helper(ema_output1), dim=1).long()

        consistency_loss1 = consistency_weight * consistency_criterion(outputs1[BATCH_SIZE:], pseudo_label1)
        consistency_loss2 = consistency_weight * consistency_criterion(outputs2[BATCH_SIZE:], pseudo_label2)

        loss1 = supervised_loss1 + consistency_loss1
        loss2 = supervised_loss2 + consistency_loss2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()
        scheduler1.step(epoch)
        scheduler2.step(epoch)

        update_ema_variables(model1, ema_model1, args.ema_decay, epoch * BATCHES_OF_EPOCH + iter)
        update_ema_variables(model2, ema_model2, args.ema_decay, epoch * BATCHES_OF_EPOCH + iter)

        writer.add_scalar("supervised_loss1/Dice", supervised_loss1.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.add_scalar("consistency_loss1", consistency_loss1.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.add_scalar("supervised_loss2/Dice", supervised_loss2.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.add_scalar("consistency_loss2", consistency_loss2.data.item(), epoch * BATCHES_OF_EPOCH + iter)
        writer.flush()

        log_str = 'epoch: {}, iter: {}, lr: {}, sup_loss1: {}, con_loss1: {}, sup_loss2: {}, con_loss2: {}, con_weight: {}'\
            .format(epoch, iter, scheduler1.get_lr()[0], supervised_loss1.data.item(),
                    consistency_loss1.data.item(), supervised_loss2.data.item(),
                    consistency_loss2.data.item(), consistency_weight)
        print_log(log_str, log)
    # evaluation
    with torch.no_grad():
        ema_model1.eval()
        mean_dice1, dices1 = validation(ema_model1, tgt_train_dataset, NUM_POOL)
        ema_model2.eval()
        mean_dice2, dices2 = validation(ema_model2, tgt_train_dataset, NUM_POOL)
        log_str = 'val: epoch: {}, mean_dice1: {}, dices1: {}, mean_dice2: {}, dices2: {}'\
            .format(epoch, mean_dice1, dices1, mean_dice2, dices2)
        print_log(log_str, log)
        writer.add_scalar("val acc1/Dice", mean_dice1, epoch * BATCHES_OF_EPOCH)
        writer.add_scalar("val acc2/Dice", mean_dice2, epoch * BATCHES_OF_EPOCH)
        writer.flush()

        print_log("saving checkpoint...", log)
        is_best1 = False
        if mean_dice1 > best_dice1:
            best_dice1 = mean_dice1
            is_best1 = True
            log_str = 'best dice1: mean_dice1: {}'.format(mean_dice1)
            print_log(log_str, log)
        is_best2 = False
        if mean_dice2 > best_dice2:
            best_dice2 = mean_dice2
            is_best2 = True
            log_str = 'best dice2: mean_dice2: {}'.format(mean_dice2)
            print_log(log_str, log)

        state1 = {
            'epoch1': epoch,
            'best_epoch1': best_dice1,
            'model1': model1.state_dict(),
            'ema_model1': ema_model1.state_dict(),
            'optimizer1': optimizer1.state_dict(),
        }
        save_checkpoint(state1, is_best1, CHECKPOINT_PATH, CHECKPOINT_NAME1.format(epoch), 'model_best1.pth')
        state2 = {
            'epoch2': epoch,
            'best_epoch2': best_dice2,
            'model2': model2.state_dict(),
            'ema_model2': ema_model2.state_dict(),
            'optimizer2': optimizer2.state_dict(),
        }
        save_checkpoint(state2, is_best2, CHECKPOINT_PATH, CHECKPOINT_NAME2.format(epoch), 'model_best2.pth')

writer.close()
log.close()


