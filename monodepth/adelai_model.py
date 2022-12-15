import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from monodepth.depth_model import DepthModel
from adelai.lib.multi_depth_model_woauxi import RelDepthModel
from adelai.lib.net_tools import load_ckpt

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Configs for LeReS')
#     parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
#     parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

#     args = parser.parse_args()
#     return args

# def scale_torch(img):
#     """
#     Scale the image and output it in torch.tensor.
#     :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
#     :param scale: the scale factor. float
#     :return: img. [C, H, W]
#     """
#     if len(img.shape) == 2:
#         img = img[np.newaxis, :, :]
#     if img.shape[2] == 3:
#         transform = transforms.Compose([transforms.ToTensor(),
# 		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
#         img = transform(img)
#     else:
#         img = img.astype(np.float32)
#         img = torch.from_numpy(img)
#     return img


class AdelaiModel(DepthModel):
    
    def __init__(self, support_cpu=False): 
        super().__init__() # DepthModel 包含forward
        args = self.parse_args()

        # 開GPU
        if support_cpu:
            # Allow the model to run on CPU when GPU is not available.
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Rather raise an error when GPU is not available.
            self.device = torch.device("cuda")

        # create depth model
        self.model = RelDepthModel(backbone=args.backbone)
        self.model.eval()

        # load checkpoint
        load_ckpt(args, self.model, None, None)
        self.model.cuda()
        
        self.norm_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        self.norm_stdev = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
    
    def parse_args(self):
        parser = argparse.ArgumentParser(
            description='Configs for LeReS')
        parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
        parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

        args = parser.parse_args()
        return args
    
    def estimate_depth(self, images):
        """
        Scale the image and output it in torch.tensor.
        :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]

        same as midas
        """
        shape = images.shape
        C, H, W = shape[-3:]
        input_ = images.reshape(-1, C, H, W).to(self.device)

        input_ = (input_ - self.norm_mean.to(self.device)) / self.norm_stdev.to(
            self.device
        )
        output = self.model.inference(input_) # 有做 pred_depth_out = depth - depth.min() + 0.01
        depth = cv2.resize(output, (images.shape[1], images.shape[0]))
        return depth

    def save(self, file_name):
        state_dict = self.model.state_dict()
        torch.save(state_dict, file_name)
# estimate depth and save
        # for i, v in enumerate(imgs_path):
        #     print('processing (%04d)-th image... %s' % (i, v))
        #     rgb = cv2.imread(v)
        #     rgb_c = rgb[:, :, ::-1].copy()
        #     gt_depth = None
        #     A_resize = cv2.resize(rgb_c, (448, 448))
        #     rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        #     img_torch = scale_torch(A_resize)[None, :, :, :]
        #     # 得到深度
        #     pred_depth = self.model.inference(img_torch).cpu().numpy().squeeze()
        #     pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        #     # if GT depth is available, uncomment the following part to recover the metric depth
        #     #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

        #     img_name = v.split('/')[-1]
        #     cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        #     # save depth
        #     plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
        #     cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
