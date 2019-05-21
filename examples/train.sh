
set -ex
CUDA_VISIBLE_DEVICES=1 python main.py\
  --epochs 80\
  --schedule 80\
  --lr 1e-3\
  -c limited/cocov4/1e3_bs16_256\
  --arch rascv2\
  --machine basic\
  --input-size 256\
  --train-batch 16\
  --limited-dataset 10000\
  --test-batch 1\
  --base-dir /home/liuxuebo/scocov4/\
  --data 2017



