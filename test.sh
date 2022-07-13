python test.py \
 --resume S2CR.tar \
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
 --base-dir /datasets/ \
 --data HCOCO\

