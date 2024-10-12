import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.system('python ../../Main/Baseline_Train/UNet_Train.py --fold=0 --batch_size=1 '
          '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=400 --epoches=50 --nonlin=2 --norm_op=1 '
          '--log_path="../../logs/Baseline_0721_w1/fold{}deleted_and_addData_5fold" '
          '--checkpoint_path="../../checkpoints/Baseline_0721_w1/fold{}deleted_and_addData_5fold" '
          '--summary_writer="../../logs/Baseline_0721_w1/fold{}deleted_and_addData_5fold" '
          '--consistency=1')

# os.system('python Main/Baseline_Train/UNet_Train.py --fold=2 --batch_size=2 '
#           '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=250 --epoches=50 --nonlin=2 --norm_op=2 '
#           '--log_path="logs/Baseline_0608/fold{}" '
#           '--checkpoint_path="checkpoints/Baseline_0608/fold{}" '
#           '--checkpoint_name="checkpoint-{}.pth" '
#           '--summary_writer="logs/Baseline_0608/fold{}" ')

# os.system('python Main/Baseline_Train/UNet_Train.py --fold=3 --batch_size=2 '
#           '--initial_lr=1e-4 --lr_step_size=30 --lr_gamma=0.1 --batches_of_epoch=250 --epoches=50 --nonlin=2 --norm_op=2 '
#           '--log_path="logs/Baseline_0628/fold{}" '
#           '--checkpoint_path="checkpoints/Baseline_0628/fold{}" '
#           '--checkpoint_name="checkpoint-{}.pth" '
#           '--summary_writer="logs/Baseline_0628/fold{}" ')

