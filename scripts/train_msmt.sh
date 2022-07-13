# settings: augmix + soft erasing + pre part remix
python train.py \
--config_file configs/MSMT17/resnet_base.yml \
MODEL.DEVICE_ID "('2')" \
OUTPUT_DIR "('./logs/msmt')" INPUT.AUGMIX "(True)" \
INPUT.ERASING_TYPE "('soft')" \
INPUT.RE_PROB "(0.5)" \
INPUT.MIXING_COEFF "([0.5, 1.0])"
