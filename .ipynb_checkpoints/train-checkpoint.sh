source /home/ipprlab/anaconda3/bin/activate py36

set -ex
CUDA_VISIBLE_DEVICES=0 python main.py  --epochs 50\
 --schedule 50\
 --lr 2e-4\
 -c eval/ablation/fusionv3s_wlabel_concat_1GAN_stack_64_cr\
 --arch fusionv3s_vgg\
 --withLabel True\
 --sltype vggx\
 --lambda_NCE 0.0\
 --min_area 1000 \
 --L1_pixel_loss 0.0\
 --ssim-loss 0.0\
 --masked True\
 --machine ssppv2\
 --input-size 256\
 --train-batch 8\
 --test-batch 1\
 --base-dir /home/ipprlab/datasets/\
 --data iHarmony\


