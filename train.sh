python main.py  --epochs 50\
 --schedule 50\
 --lr 2e-4\
 -c eval/ \
 --arch fusion_re\
 --withLabel True\
 --sltype vggx\
 --lambda_NCE 0.0\
 --min_area 1000 \
 --L1_pixel_loss 0.0\
 --ssim-loss 0.0\
 --masked True\
 --machine sspp_re\
 --input-size 256\
 --train-batch 8\
 --test-batch 1\
 --base-dir /home/datasets/\
 --data iHarmony\


