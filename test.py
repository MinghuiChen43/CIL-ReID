import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file",
                        default="",
                        help="path to config file",
                        type=str)
                        
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, corrupted_val_loader, corrupted_query_loader, corrupted_gallery_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
        cfg)

    # market: 751
    # cuhk: 767
    # msmt: 1041
    num_classes = 751

    model = make_model(cfg,
                       num_class=num_classes,
                       camera_num=camera_num,
                       view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)


    for eval_epoch in range(10):
        print("Eval epoch ", eval_epoch)
        print("=" * 64)
        loader_list = [
            val_loader, corrupted_val_loader, corrupted_query_loader,
            corrupted_gallery_loader
        ]
        name = [
            "Clean eval", "Corrupted eval", "Corrupted query",
            "Corrupted gallery"
        ]
        for loader_i in range(4):
            print("Evaluating on ", name[loader_i])
            mINP, mAP, rank1, rank5, rank10 = do_inference(
                cfg, model, loader_list[loader_i], num_query)
            mINP = round(mINP * 100, 2)
            mAP = round(mAP * 100, 2)
            rank1 = round(rank1 * 100, 2)
            rank5 = round(rank5 * 100, 2)
            rank10 = round(rank10 * 100, 2)
            path = cfg.OUTPUT_DIR + '/' + cfg.DATASETS.NAMES + '_eval_info.csv'
            import csv
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [mINP, mAP, rank1, rank5, rank10]
                csv_write.writerow(data_row)
