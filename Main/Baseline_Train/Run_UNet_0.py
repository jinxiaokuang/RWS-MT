import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.system('./UNet_Train.py --fold=0 --batch_size=1 '
          '--initial_lr=1e-3 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --epoches=50 --nonlin=2 --norm_op=1 '
          '--log_path="logs/Baseline_0719_w1/fold{}lr{}" '
          '--checkpoint_path="checkpoints/Baseline_0719_w1/fold{}lr{}" '
          '--summary_writer="logs/Baseline_0719_w1/fold{}lr{}" '
          '--consistency=1')

# os.system('python Main/Baseline_Train/UNet_Train.py --fold=0 --batch_size=1 '
#     '--initial_lr=1e-5 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --epoches=50 --nonlin=2 --norm_op=1 '
#     '--log_path="logs/Baseline_0719_w1/fold{}lr{}" '
#     '--checkpoint_path="checkpoints/Baseline_0719_w1/fold{}lr{}" '
#     '--summary_writer="logs/Baseline_0719_w1/fold{}lr{}" '
#     '--consistency=1')
# os.system('python Main/Baseline_Train/UNet_Train.py --fold=0 --batch_size=1 '
#         '--initial_lr=1e-6 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --epoches=50 --nonlin=2 --norm_op=1 '
#         '--log_path="logs/Baseline_0719_w1/fold{}" '
#         '--checkpoint_path="checkpoints/Baseline_0719_w1/fold{}" '
#         '--summary_writer="logs/Baseline_0719_w1/fold{}" '
#         '--consistency=1')

# os.system('python Main/Baseline_Train/UNet_Train.py --fold=0 --batch_size=2 '
#           '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=250 --epoches=50 --nonlin=2 --norm_op=1 '
#           '--log_path="logs/Baseline_0625_IN/fold{}" '
#           '--checkpoint_path="checkpoints/Baseline_0625_IN/fold{}" '
#           '--checkpoint_name="checkpoint-{}.pth" '
#           '--summary_writer="logs/Baseline_0625_IN/fold{}" ')

# os.system('python Main/Baseline_Train/UNet_Train.py --fold=1 --batch_size=2 '
#           '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=250 --epoches=50 --nonlin=2 --norm_op=2 '
#           '--log_path="logs/Baseline_0608/fold{}" '
#           '--checkpoint_path="checkpoints/Baseline_0608/fold{}" '
#           '--checkpoint_name="checkpoint-{}.pth" '
#           '--summary_writer="logs/Baseline_0608/fold{}" ')

