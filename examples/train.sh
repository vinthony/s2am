# CUDA_VISIBLE_DEVICES=1 python main.py\
#   --epochs 80\
#   --schedule 80\
#   --lr 1e-3\
#   -c psnr/cocov2/1e3_bs16_256\
#   --arch naivemmucross\
#   --machine multimaskedpixel\
#   --attention-loss-weight 1e9\
#   --input-size 256\
#   --limited-dataset 0\
#   --train-batch 16\
#   --test-batch 1\
#   --base-dir $HOME/Datasets/scoco/\
#   --data 2017

# CUDA_VISIBLE_DEVICES=1 python main.py\
#   --epochs 80\
#   --schedule 80\
#   --lr 1e-3\
#   -c psnr/cocov2/1e3_bs16_256\
#   --arch naivemmucross\
#   --machine mmaskedgan\
#   --input-size 256\
#   --train-batch 16\
#   --norm-type gan\
#   --limited-dataset 0\
#   --test-batch 1\
#   --base-dir $HOME/Datasets/scoco/\
#   --data 2017


# CUDA_VISIBLE_DEVICES=1 python main.py\
#   --epochs 60\
#   --schedule 60\
#   --lr 1e-3\
#   -c psnr/cocov2/1e3_bs16_256\
#   --arch rascv3\
#   --machine basic\
#   --input-size 256\
#   --train-batch 16\
#   --limited-dataset 0\
#   --test-batch 1\
#   --base-dir $HOME/Datasets/mix/\
#   --data mix

CUDA_VISIBLE_DEVICES=1 python main.py\
  --epochs 60\
  --schedule 60\
  --lr 1e-3\
  -c psnr/cocov4/1e3_bs4_256\
  --arch rascv2\
  --machine basic\
  --input-size 256\
  --limited-dataset 0\
  --train-batch 4\
  --test-batch 1\
  --base-dir $HOME/Datasets/mixed/\
  --data final