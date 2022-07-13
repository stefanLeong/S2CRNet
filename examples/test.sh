
source /home/user/mb95464/anaconda3/bin/activate py36
set -ex

CUDA_VISIBLE_DEVICES=2 python /home/user/mb95464/projects/splitNet/test.py \
  -c test/HCOCO_ssim\
  --resume /home/user/mb95464/eval/10kgray/1e3_bs4_256_relative_ssim_vgg_s2am_HCOCO_s2am/model_best.pth.tar\
  --arch s2am\
  --machine s2am\
  --input-size 256\
  --test-batch 64\
  --evaluate\
  --base-dir /home/user/mb95464/projects/splitNet/datasets/\
  --data HAdobe5k