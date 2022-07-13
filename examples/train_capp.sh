source /home/user/mb95464/anaconda3/bin/activate py36

set -ex

CUDA_VISIBLE_DEVICES=0 python /home/user/mb95464/projects/splitNet/main.py  --epochs 400\
 --schedule 400\
 --lr 1e-3\
 -c eval/HAdobe5k/splitwith_capp/1e3_bs4_256_no_se\
 --arch split_with_capp\
 --sltype vggx\
 --lambda_NCE 0.0\
 --ssim-loss 0.0\
 --masked True\
 --machine split_with_capp\
 --input-size 256\
 --train-batchu 4\
 --test-batch 1\
 --base-dir /home/user/mb95464/projects/splitNet/datasets/\
 --data HAdobe5k
