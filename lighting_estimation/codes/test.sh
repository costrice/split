INDOOR_AE=lighting_estimation/checkpoints/ae_indoor.pth
OUTDOOR_AE=lighting_estimation/checkpoints/ae_outdoor.pth

INDOOR_EST=lighting_estimation/checkpoints/indoor_SLE_trained_on_ae.pth
OUTDOOR_EST=lighting_estimation/checkpoints/outdoor_SLE_trained_on_ae.pth

INDOOR_PRED=lighting_estimation/checkpoints/pred_indoor.pth
OUTDOOR_PRED=lighting_estimation/checkpoints/pred_outdoor.pth

CUDA_IDS=2
num_test=11000
####################### real data ###############################

echo proposed outdoor GAN

CUDA_VISIBLE_DEVICES=$CUDA_IDS python test.py --expname infer_outdoor --batch-size 1 --gpu-ids 0 --net_type predictor --num-epoch 30 --use_aug 0 --real_data 1 --load_ae_dir $OUTDOOR_AE --predictor_type gan --vis_size 256 --real_test_mode 2 --saving_prefix _real_tex --syn_real_v2 --load_est_dir $OUTDOOR_EST --use_tex --load_pred_dir $OUTDOOR_PRED  --new_outdoor


echo proposed indoor GAN
CUDA_VISIBLE_DEVICES=$CUDA_IDS python test.py --expname infer_indoor --batch-size 1 --gpu-ids 0 --net_type gan --num-epoch 30 --use_aug 0 --real_data 1 --load_ae_dir $INDOOR_AE --predictor_type ffgan --vis_size 256 --real_test_mode 1 --saving_prefix _real_tex --load_est_dir $INDOOR_EST --use_tex --syn_real_v2  --load_pred_dir $INDOOR_PRED
