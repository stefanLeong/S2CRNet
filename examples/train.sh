
source /home/user/mb95464/anaconda3/bin/activate py36

set -ex

CUDA_VISIBLE_DEVICES=0 python /home/user/mb95464/projects/splitNet/main.py  --epochs 400\
 --schedule 400\
 --lr 1e-3\
 -c eval/HFlickr/1e4_bs4_256_vgg\
 --arch split_with_s2am\
 --sltype vgg\
 --style-loss 0.025\
 --ssim-loss 0.15\
 --masked True\
 --machine split_with_s2am\
 --input-size 256\
 --train-batch 4\
 --test-batch 1\
 --base-dir /home/user/mb95464/projects/splitNet/datasets/\
 --data HFlickr
