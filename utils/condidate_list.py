import cv2
import os
from PIL import Image
import numpy as np
import torch
import os.path as osp
from os import listdir
import torchvision.transforms as T
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

#加红色绿色边框
def image_border(img_ori, loc='a', width=3, color=(0, 0, 0)):
    '''
    img_ori: (str) 需要加边框的图片路径
    dst: (str) 加边框的图片保存路径
    loc: (str) 边框添加的位置, 默认是'a'(
        四周: 'a' or 'all'
        上: 't' or 'top'
        右: 'r' or 'rigth'
        下: 'b' or 'bottom'
        左: 'l' or 'left'
    )
    width: (int) 边框宽度 (默认是3)
    color: (int or 3-tuple) 边框颜色 (默认是0, 表示黑色; 也可以设置为三元组表示RGB颜色)
    '''
    # 读取图片
    img_ori = Image.fromarray(img_ori)
    w = img_ori.size[0]
    h = img_ori.size[1]

    # 添加边框
    if loc in ['a', 'all']:
        w += 2*width
        h += 2*width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (width, width))
    elif loc in ['t', 'top']:
        h += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, width, w, h))
    elif loc in ['r', 'right']:
        w += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w-width, h))
    elif loc in ['b', 'bottom']:
        h += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w, h-width))
    elif loc in ['l', 'left']:
        w += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (width, 0, w, h))
    else:
        pass

    # 返回图片
    return img_new
#合并图像
def concatenate_img(images,match,img_name):
    ims = []
    for i in range(len(match)):
        # ims.append(images[i])
        if match[i] == 1:
            #true
            ims.append(image_border(images[i],color=(0,255,0)))
        else:
            ims.append(image_border(images[i], color=(255,0, 0)))
    # 获取当前文件夹下所以图片
    #ims = [Image.open('../images/%s' % fn) for fn in listdir('../images') if fn.endswith('.jpg')]

    ims_size = [list(im.size) for im in ims]
    middle_width = sorted(ims_size, key=lambda im: im[0])[int(len(ims_size) / 2)][0]  # 中位数宽度
    ims = [im for im in ims if im.size[0] > middle_width / 2]  # 过滤宽度过小的无效图片

    # 过滤后重新计算
    ims_size = [list(im.size) for im in ims]
    middle_width = sorted(ims_size, key=lambda im: im[0])[int(len(ims_size) / 2)][0]  # 中位数宽度
    ims = [im for im in ims if im.size[0] > middle_width / 2]  # 过滤宽度过小的无效图片

    # 计算相对长图目标宽度尺寸
    for i in range(len(ims_size)):
        rate = middle_width / ims_size[i][0]
        ims_size[i][0] = middle_width
        ims_size[i][1] = int(rate * ims_size[i][1])
    sum_height = sum([im[1] for im in ims_size])
    # 创建空白长图
    result = Image.new(ims[0].mode, (middle_width, sum_height))
    # 拼接
    top = 0
    for i, im in enumerate(ims):
        mew_im = im.resize(ims_size[i], Image.ANTIALIAS)  # 等比缩放
        result.paste(mew_im, box=(0, top))
        top += ims_size[i][1]
    # 保存
    result.save(img_name)
# CAM可解释性可视化
def Cam(image_path,cam,cfg,target_view,camid):
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    target_view = torch.tensor(target_view, dtype=torch.int64)
    camid = torch.tensor(camid, dtype=torch.int64)

    # rgb_img = Image.open(image_path).convert('RGB')
    # input_tensor = val_transforms(rgb_img)
    # input_tensor.unsqueeze_(0)
    # rgb_img = rgb_img.resize((256, 256), Image.ANTIALIAS)
    # rgb_img = np.float32(rgb_img) / 255


    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (256, 256))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    cam.set_camid_and_target_veiew(camid,target_view)
    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)
                        # eigen_smooth=args.eigen_smooth,
                        # aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    # 是否是等比例缩放?
    return cam_image

def save_condidate_list(args, cfg, img_path_list, num_query, indices, matches, q_pids, g_pids, all_AP, cam=None,camids=None,target_view=None, rank=10 ):
    # 查询图像总列表
    query_img_path_lists = np.asarray(img_path_list[:num_query])
    # 测试专用，只提取出第一行
    # order = (indices[0,:rank]).astype(np.int32)
    # match = (matches[0,:rank]).astype(np.int32)
    # 提取出所有行的前rank个gallery图像
    order = (indices[:,:rank]).astype(np.int32)
    match = (matches[:,:rank]).astype(np.int32)

    mAP_order = []
    mAP_match = []
    mAP_q_pid = []
    query_img_path_list = []
    num = 0
    for i in range(num_query):
        q_pid = q_pids[i]
        if all_AP[i] < 0.8:
            mAP_order.append(order[i])
            mAP_q_pid.append(q_pid)
            query_img_path_list.append(query_img_path_lists[i])
            mAP_match.append(match[i])
            num +=1

    query_img_path_list = np.asarray(query_img_path_list)
    # gallery_images 是二维的数组
    gallery_img_path_list = np.asarray(img_path_list[num_query:])
    gallery_images = gallery_img_path_list[mAP_order]

    gallery_camids_list = np.asarray(camids[num_query:])
    gallery_camids = gallery_camids_list[mAP_order]

    gallery_target_view_list = np.asarray(target_view[num_query:])
    gallery_target_view = gallery_target_view_list[mAP_order]

    # 设置存储路径
    save_root_dir = os.path.join(cfg.OUTPUT_DIR,'result')
    save_cam_image = os.path.join(save_root_dir, 'cam_images')
    if not osp.exists(save_cam_image):
        os.makedirs(save_cam_image)
    print(num)
    for i in range(num):
        print(i)
        # 取query图像
        query_path = os.path.join(cfg.DATASETS.ROOT_DIR,'VeRi/image_query',query_img_path_list[i])
        query_img = cv2.imread(query_path)
        file_name = query_img_path_list[i].split('.')[0]
        # save_path = ../log/veri_base/result/query_name
        save_path = os.path.join(save_root_dir,file_name)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path,'query.jpg'),query_img)

        # 取前10个gallery图像做condidate list
        gallery_images_path = []
        for j in range(rank):
            gallery_images_path.append(os.path.join(cfg.DATASETS.ROOT_DIR,'VeRi/image_test',gallery_images[i][j]))
        gallery_cam_image = []
        gallery_images = []
        for k in range(len(gallery_images_path)):
            gallery_cam_image.append(Cam(gallery_images_path[k],cam,cfg,gallery_target_view[i][k],gallery_camids[i][k]))

        for k in range(len(gallery_cam_image)):
            g_save_path = os.path.join(save_path,'{}.jpg'.format(k))

            cv2.imwrite(g_save_path,gallery_cam_image[k])

        gallery_save_path = os.path.join(save_cam_image,'{}.jpg'.format(file_name))
        concatenate_img(gallery_cam_image, mAP_match[i],img_name=gallery_save_path)