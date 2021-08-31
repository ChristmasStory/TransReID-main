import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
#cam
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from utils.condidate_list import save_condidate_list

def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def do_inference(args,cfg,
                 model,
                 val_loader,
                 num_query):
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}
    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

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
    # 求最后一层的梯度
    target_layer = model.base.blocks[-2].norm1
    # target_layer = model.bottleneck
    # target_layer = model.base.norm

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")




    img_path_list = []
    target_views = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            # ndarray类型 target_view
            target_views.extend(np.asarray(target_view))
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)

            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)




    cmc, mAP, _, _, _, _, _, indices, matches,q_pids, g_pids, all_AP = evaluator.compute()

    for k in model.state_dict():
        print(k)

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)
    rank = 10
    save_condidate_list(args,
                        cfg,
                        img_path_list,
                        num_query,
                        indices,
                        matches,
                        q_pids,
                        g_pids,
                        all_AP,
                        cam,
                        evaluator.camids,
                        target_views,
                        rank)

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]