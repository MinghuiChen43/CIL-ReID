# settings: augmix + soft erasing + pre part remix
python train.py \
--config_file configs/Market/resnet_base.yml \
MODEL.DEVICE_ID "('0')" \
OUTPUT_DIR "('./logs/market')" INPUT.AUGMIX "(True)" \
INPUT.ERASING_TYPE "('soft')" \
INPUT.RE_PROB "(0.5)" \
INPUT.MIXING_COEFF "([0.5, 1.0])"
