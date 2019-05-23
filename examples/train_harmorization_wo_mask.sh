CUDA_VISIBLE_DEVICES=1 python main.py\
  --epochs 60\
  --schedule 60\
  --lr 1e-3\
  -c psnr/cocov4/1e3_bs8_256\
  --arch naivemultimaskedurasc\
  --machine maskedganplus\
  --input-size 256\
  --limited-dataset 0\
  --train-batch 8\
  --test-batch 1\
  --base-dir $HOME/Datasets/scoco/\
  --data 2017
