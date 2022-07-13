# python test.py \
# --config_file configs/MSMT17/resnet_base.yml \
# MODEL.DEVICE_ID "('2')" \
# OUTPUT_DIR "('./logs/msmt/')" \
# TEST.WEIGHT "('./logs/msmt/resnet50_120.pth')"

python test.py \
--config_file configs/MSMT17/resnet_base.yml \
MODEL.DEVICE_ID "('0')" \
OUTPUT_DIR "('./logs/msmt/official/')" \
TEST.WEIGHT "('/home/wangzhiqiang/Github/CIL-ReID/logs/final_ckpt/msmt17_resnet50.pth')"
