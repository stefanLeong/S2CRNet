source /home/user/mb95464/anaconda3/bin/activate py36

set -ex

CUDA_VISIBLE_DEVICES=0 python /home/user/mb95464/projects/splitNet/main.py  --epochs 400\
 --schedule 400\
 --lr 1e-3\
 -c eval/HFlickr/splitsingle/1e3_bs4_256_no_se_single\
 --arch split_single\
 --sltype vggx\
 --style-loss 0.0\
 --ssim-loss 0.0\
 --masked True\
 --machine split_single\
 --input-size 256\
 --train-batch 4\
 --test-batch 1\
 --base-dir /home/user/mb95464/projects/splitNet/datasets/\
 --data HFlickr
