
source /home/user/mb95464/anaconda3/bin/activate py36

set -ex

CUDA_VISIBLE_DEVICES=2 python /home/user/mb95464/projects/splitNet/main.py  --epochs 200\
 --schedule 200\
 --lr 1e-3\
 -c eval/10kgray/1e3_bs4_256_relative_ssim_vgg\
 --arch s2am\
 --sltype vggx\
 --style-loss 0.0\
 --ssim-loss 0.0\
 --masked True\
 --machine s2am\
 --input-size 256\
 --train-batch 4\
 --test-batch 1\
 --base-dir /home/user/mb95464/projects/splitNet/datasets/\
 --data HCOCO
