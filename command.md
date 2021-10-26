#### train 示例代码 ##
python train.py --cuda \
--batchSize 2 \
--size 512 \
--decay_epoch 50 \
--n_epochs 100 \
--dataroot /home/lyc/code/HaoL/LDH/dataset/dataset/syn2city_scale_width/ \
--exp_name syn2city_100e_512_upsamp


#### test 示例代码 ##
python test_single.py \
--cuda \
--which_epoch 99 \
--direction A2B \
--res_dir exp/GTA52City/test_A2B \
--model_dir exp/GTA52City/output \
--dataroot /home/lyc/data/ll/dataset/gta2cityscapes_resize/



python train.py --dataroot /home/lyc/data/ll/dataset/gta2cityscapes_resize/ --cuda
python continue_train.py --dataroot /home/lyc/data/ll/dataset/SYN_day2night --epoch 99 --cuda
python test.py --dataroot /home/lyc/data/ll/dataset/gta2cityscapes/ --res_dir exp/G2C_100e_400_upsamp/test/C2G_100_epoch_256 --model_dir exp/G2C_100e_400_upsamp/output/ --which_epoch 90 --cuda
python test.py --dataroot /home/lyc/data/ll/dataset/SYN_day2night/ --res_dir D2N_100_epoch_256 --model_dir exp/S_D2N_100e_400_upsamp/output/ --which_epoch 99 --cuda
python test_single.py --dataroot /home/lyc/code/HaoL/data/GTA5/val/images_resize --res_dir exp/G2C_100e_400_upsamp/test/gta2city_90e_scale_width --model_dir exp/G2C_100e_400_upsamp/output/ --which_epoch 90 --cuda

python test.py --dataroot /home/lyc/code/HaoL/LDH/dataset/dataset/syn2city_scale_width/ --res_dir exp/syn2city_50e --model_dir exp/syn2city_100e_400_upsamp/output/ --which_epoch 50 --cuda

##  SYN_sunny2foggy
python train.py --dataroot /home/lyc/code/HaoL/LDH/dataset/dataset/SYN_sunny2foggy_resize/ --batchSize 2  --size 512 --cuda

/home/lyc/code/HaoL/LDH/dataset/dataset/SYN_sunny2foggy_resize/



##### GTA52City
# test
python test_single.py --dataroot /home/lyc/code/HaoL/data/GTA5/images --res_dir /home/lyc/data/HaoL/GTA5/rotate_OP_90e_1024-2048-size_trans_images --model_dir exp/G2C_100e_400_upsamp/output/ --which_epoch 90 --cuda --direction A2B
python test.py --dataroot /home/lyc/data/ll/dataset/gta2cityscapes/ --test_dir G2c_100_epoch_256 --model_dir exp/G2C_100e_400_upsamp/output/ --which_epoch 99 --cuda

python test.py --dataroot /home/lyc/data/ll/dataset/gta2cityscapes/ --test_dir G2c_100_epoch_1024_512 --model_dir exp/G2C_100epoch_256/output/ --which_epoch 99


python test.py --dataroot /home/lyc/code/HaoL/data/GAN/GTA5_to_cityscape --generator_A2B 'output/CycleGAN+��ת�Լල/netG_A2B_100.pth'  --res 'res/CycleGAN+��ת�Լල' --direction  'A2B' --cuda
python test.py --dataroot /home/lyc/code/HaoL/data/GAN/GTA5_to_cityscape --generator_A2B 'output/CycleGAN+��ת�Լල/netG_B2A_100.pth'  --res 'res/CycleGAN+��ת�Լල' --direction  'B2A' --cuda

## sny2city
python test_single.py --dataroot /home/lyc/code/HaoL/LDH/dataset/dataset/syn2city_scale_width/train/A/ --res_dir exp/syn2city_100e_400_upsamp/test/syn2city_55e_1024_608 --model_dir exp/syn2city_100e_400_upsamp/output/ --which_epoch 55 --cuda --direction A2B
python test_single.py --dataroot /home/haol/code/data/SYNTHIA/SPRING/RGB/ --res_dir /home/lyc/data/HaoL/image_transform/transform/syn2city/rotate_OP_55epoch_1024_2048/ --model_dir exp/syn2city_100e_400_upsamp/output/ --which_epoch 55 --cuda --direction A2B
## day2night
python test_single.py --dataroot /home/lyc/code/HaoL/LDH/dataset/dataset/SYN_day2night_scale_width/train/B/ --res_dir exp/S_D2N_100e_400_upsamp/test/day2night_99e_1024_608 --model_dir exp/S_D2N_100e_400_upsamp/output/ --which_epoch 99 --cuda --direction B2A
## sunny2foggy
python test_single.py --dataroot /home/lyc/code/HaoL/LDH/dataset/dataset/SYN_sunny2foggy_scale_width/train/B/ --res_dir exp/SYN_sun2fog_100e_512_upsamp/test/sunny2foggy_99e_1024_608 --model_dir exp/SYN_sun2fog_100e_512_upsamp/output/ --which_epoch 99 --cuda --direction B2A

##night2day
python test_single.py --dataroot /home/lyc/code/HaoL/LDH/dataset/dataset/syn_foggy/RGB/ --res_dir /home/lyc/data/HaoL/image_transform/transform/foggy2sunny/rotate_OP_55epoch_1280_760/ --model_dir exp/SYN_sun2fog_100e_512_upsamp/output/ --which_epoch 99 --cuda --direction B2A

## city_foggy2sunny
python test_single.py --dataroot /home/lyc/data/HaoL/GAN/Cityscape_day_foy/val/补充B/ --save_dir /home/lyc/data/HaoL/GAN/Cityscape_day_foy/val/B_2sunny --model_dir /home/lyc/code/HaoL/LDH/CycleGAN+旋转损失+多尺度鉴别器+OP/exp/city_sunny2foogy_resize1024_512_cropsize400_bs2_foggybata0.01/output/ --which_epoch 99 --cuda --direction B2A
## syn_night2day
python test_single.py --dataroot /home/lyc/code/HaoL/LDH/dataset/dataset/syn_night/RGB/ --save_dir /home/lyc/code/HaoL/LDH/CycleGAN+旋转损失+多尺度鉴别器+OP/exp/S_D2N_100e_400_upsamp/syn_night2day_99e --model_dir /home/lyc/code/HaoL/LDH/CycleGAN+旋转损失+多尺度鉴别器+OP/exp/S_D2N_100e_400_upsamp/output/ --which_epoch 99 --cuda --direction B2A


## 消融实验

# use_rot use_ms
python train.py --cuda --batchSize 1 --size 400 \
--decay_epoch 50 --n_epochs 100 --use_rot --use_ms \
--dataroot /home/haol/data/Dataset/风格迁移数据集/gta2cityscapes_1024_512 --exp_name gta2city_100e_400_upsamp_use_rot__use_ms

# use_op （h和w 反了）
python train.py --cuda --batchSize 1 --size 400 \
--decay_epoch 50 --n_epochs 100 --use_op \
--dataroot /home/haol/data/Dataset/风格迁移数据集/gta2cityscapes --exp_name gta2city_100e_512_upsamp_use_op

# use_op use_ms
python train.py --cuda --batchSize 1 --size 400 \
--decay_epoch 50 --n_epochs 100 --use_op --use_ms \
--dataroot /home/haol/data/Dataset/风格迁移数据集/gta2cityscapes --exp_name gta2city_100e_512_upsamp_use_op_use_ms