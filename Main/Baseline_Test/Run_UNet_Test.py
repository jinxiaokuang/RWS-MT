import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# nonlin=1:ReLU, 2:LeakyReLU
# norm_op=1:IN, 2:BN


# os.system('python3 -W ignore ./UNet_Test.py --fold=1 --vendor="A" --patch_size_h=224 --patch_size_w=224 --batch_size=48 '
#           '--initial_lr=1e-3 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=250 --epoches=100 '
#           '--log_path="../../logs/Baseline_UNet/fold{}" '
#           '--checkpoint_path="../../checkpoints/Baseline_UNet/fold{}" '
#           '--checkpoint_name="checkpoint-{}.pth" '
#           '--summary_writer="../../logs/Baseline_UNet/fold{}" ')


os.system('python3 -W ignore ./UNet_Test.py --fold=0 --vendor="A" --patch_size_h=224 --patch_size_w=224 --batch_size=48 '
          '--initial_lr=1e-3 --lr_step_size=15 --lr_gamma=0.1 --batches_of_epoch=250 --epoches=60 --nonlin=2 --norm_op=2 '
          '--full_training=0 '
          '--log_path="../../logs/Baseline_UNet/{}/IN/fold{}" '
          '--checkpoint_path="../../checkpoints/Baseline_UNet/{}/IN/fold{}" '
          '--checkpoint_name="checkpoint-{}-IN-{}.pth" '
          '--summary_writer="../../logs/Baseline_UNet/{}/IN/fold{}" ')


