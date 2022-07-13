source /home/ipprlab/anaconda3/bin/activate py36

set -ex


 CUDA_VISIBLE_DEVICES=0 python /home/ipprlab/projects/xharm/test.py \
 --resume /home/ipprlab/projects/xharm/DSCR.tar \
 --c test/vgg\
 --arch fusionv3s\
 --withLabel True\
 --sltype vggx\
 --lambda_NCE 0.0\
 --min_area 1000 \
 --L1_pixel_loss 0.0\
 --ssim-loss 0.0\
 --masked False\
 --machine ssppv2\
 --input-size 256\
 --hr_size 256 \
 --train-batch 8\
 --test-batch 1\
 --base-dir /home/ipprlab/datasets/\
 --data HCOCO\

