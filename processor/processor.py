import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torchvision
import torch.nn.functional as F

import numpy as np
import math
import random


def do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer,
             optimizer_center, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD' or epoch <= cfg.SOLVER.WARMUP_EPOCHS:
            print("scheduler lr...")
            scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            if cfg.INPUT.SELF_ID:
                img, img_aug1, img_aug2 = img.chunk(3, 1)
                img = img.to(device)
                img_aug1 = img_aug1.to(device)
                img_aug2 = img_aug2.to(device)
            else:
                img = img.to(device)

            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                if cfg.INPUT.SELF_ID:
                    img_all = torch.cat([img, img_aug1, img_aug2], 0)
                    target_cam_all = torch.cat(
                        [target_cam, target_cam, target_cam], 0)
                    score_all, feat_all = model(img_all,
                                                target,
                                                cam_label=target_cam_all,
                                                view_label=target_view)
                    score, score_aug1, score_aug2 = torch.split(
                        score_all, img.size(0))
                    feat, feat_aug1, feat_aug2 = torch.split(
                        feat_all, img.size(0))

                    if cfg.INPUT.MEAN_FEAT:
                        feat = (feat + feat_aug1 + feat_aug2) / 3.0

                    loss = loss_fn(score, feat, target, target_cam)

                    if cfg.INPUT.FEATURE_REG:
                        p_feat_clean, p_feat_aug1, p_feat_aug2 = F.softmax(
                            feat,
                            dim=1), F.softmax(feat_aug1,
                                              dim=1), F.softmax(feat_aug2,
                                                                dim=1)
                        p_feat_mixture = torch.clamp(
                            (p_feat_clean + p_feat_aug1 + p_feat_aug2) / 3.,
                            1e-7, 1).log()
                        self_id_loss = 12 * (
                            F.kl_div(p_feat_mixture,
                                     p_feat_clean,
                                     reduction='batchmean') +
                            F.kl_div(p_feat_mixture,
                                     p_feat_aug1,
                                     reduction='batchmean') +
                            F.kl_div(p_feat_mixture,
                                     p_feat_aug2,
                                     reduction='batchmean')) / 3.
                    else:
                        p_score_clean, p_score_aug1, p_score_aug2 = F.softmax(
                            score,
                            dim=1), F.softmax(score_aug1,
                                              dim=1), F.softmax(score_aug2,
                                                                dim=1)
                        p_score_mixture = torch.clamp(
                            (p_score_clean + p_score_aug1 + p_score_aug2) / 3.,
                            1e-7, 1).log()
                        self_id_loss = 12 * (
                            F.kl_div(p_score_mixture,
                                     p_score_clean,
                                     reduction='batchmean') +
                            F.kl_div(p_score_mixture,
                                     p_score_aug1,
                                     reduction='batchmean') +
                            F.kl_div(p_score_mixture,
                                     p_score_aug2,
                                     reduction='batchmean')) / 3.
                    loss += self_id_loss

                else:
                    score, feat = model(img,
                                        target,
                                        cam_label=target_cam,
                                        view_label=target_view)
                    loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if cfg.INPUT.SELF_ID:
                    print("Self ID loss: ", self_id_loss.item())
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg,
                            optimizer.state_dict()['param_groups'][0]['lr']))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
            .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device)
                    target_view = target_view.to(device)
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, mINP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, mINP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mINP: {:.2%}".format(mINP))
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    return mINP, mAP, cmc[0], cmc[4], cmc[9]
