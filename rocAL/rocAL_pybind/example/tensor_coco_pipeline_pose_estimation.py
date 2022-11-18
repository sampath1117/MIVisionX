from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import itertools
from math import fabs, sqrt

import torch
import numpy as np
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

import sys
import cv2
import os
from math import sqrt
import ctypes

#additional packages used in HRNet
from datetime import datetime
import argparse

class ROCALCOCOIterator(object):
    """
    COCO ROCAL iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, display=False, output_width = 288, output_height = 384):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = "cpu"
        self.device_id = self.loader._device_id
        self.display = True
        self.bs = self.loader._batch_size
        self.output_width = output_width
        self.output_height = output_height

        #Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")

        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")

    def next(self):
        return self.__next__()

    def __next__(self):
        if(self.loader.isEmpty()):
            raise StopIteration
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.rocalGetOutputTensors()

        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels
        self.w = self.output_tensor_list[0].batch_width()
        self.h = self.output_tensor_list[0].batch_height()
        print("w,h :",self.w,self.h)
        self.bs = self.output_tensor_list[0].batch_size()
        self.color_format = self.output_tensor_list[0].color_format()

        torch_gpu_device = torch.device('cpu', self.device_id)

        #NHWC default for now
        self.out = torch.empty((self.bs, self.h, self.w, self.color_format,), dtype=torch.uint8, device=torch_gpu_device)
        self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))

        # Image id of a batch of images
        self.loader.GetImageId(self.image_id)

        self.joints_data = dict({})
        self.loader.rocalGetJointsData(self.joints_data)

        h = int(self.output_height / 4)
        w = int(self.output_width / 4)
        # print("h, w: ", h, w)

        #Targets, Target Weights of a batch
        # self.targets = np.zeros((self.bs * 17 * w * h), dtype = "float32")
        self.targets = self.loader.rocalGetTarget()
        self.target_weights = self.loader.rocalGetTargetWeight()

        # print(self.targets.reshape((self.bs * 17,96,72)))
        target_tensor = torch.tensor(self.targets).view(self.bs , 17 , h , w).contiguous()
        target_weights_tensor = torch.tensor(self.target_weights).view(self.bs , 17 ,-1 ).contiguous()
        joints_data_tensor = self.joints_data

        return self.out ,target_tensor , target_weights_tensor , joints_data_tensor


    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self


def draw_patches(img, idx, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    htot, wtot, _ = img.shape

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.UMat(image).get()
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER_KEYPOINTS/" + str(idx)+"_"+"train"+".png", image)

def main(exp_name,
         epochs,
         batch_size,
         num_workers,
         lr,
         disable_lr_decay,
         lr_decay_steps,
         lr_decay_gamma,
         optimizer,
         weight_decay,
         momentum,
         nesterov,
         pretrained_weight_path,
         checkpoint_path,
         log_path,
         disable_tensorboard_log,
         model_c,
         model_nof_joints,
         model_bn_momentum,
         disable_flip_test_images,
         image_width,
         image_height,
         image_path,
         annotation_path,
         coco_bbox_path,
         seed,
         device,
         display,
         device_id,
         is_train,
         flip_prob,
         rotate_prob,
         half_body_prob,
         scale_factor,
         rotation_factor):

    path= "OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER_KEYPOINTS/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    if(device == "cpu"):
        _rocal_cpu = True
    else:
        _rocal_cpu = False

    # _rocal_cpu = False
    bs = int(batch_size)
    nt = int(num_workers)
    di = device_id
    local_rank = 0
    world_size = 1

    dboxes = []
    gauss_sigma = 3.0
    is_train = bool(is_train)

    coco_train_pipeline = Pipeline(batch_size=bs, num_threads=nt, device_id=device_id, seed=seed, rocal_cpu=_rocal_cpu)
    with coco_train_pipeline:
        jpegs, bboxes, labels = fn.readers.coco_keypoints(file_root=image_path, annotations_file=annotation_path, random_shuffle=False,shard_id=0, num_shards=world_size,seed=seed, is_box_encoder=False, sigma = gauss_sigma, output_image_width = image_width, output_image_height = image_height)
        images_decoded = fn.decoders.image(jpegs, device="cpu", output_type=types.RGB, file_root=image_path,
                                                 annotations_file=annotation_path, random_shuffle=False, seed=seed, num_shards=world_size, shard_id=local_rank)
        h_flip = fn.random.coin_flip(probability=0.9)
        v_flip = fn.random.coin_flip(probability=0.0)
        # if is_train and flip_prob != 0.0:
        #     images = fn.flip(images_decoded, device="cpu", h_flip = h_flip, v_flip = v_flip)
        warp_affine_images = fn.warp_affine(images_decoded, device="cpu", is_train = is_train, size=(384, 288), rotate_probability = rotate_prob, half_body_probability = half_body_prob, rotation_factor = rotation_factor, scale_factor = scale_factor)
        coco_train_pipeline.set_outputs(warp_affine_images)

    coco_train_pipeline.build()
    COCOIteratorPipeline = ROCALCOCOIterator(coco_train_pipeline, output_width = image_width, output_height = image_height)
    cnt = 0
    for epoch in range(int(epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(COCOIteratorPipeline):
            print("**************", i, "*******************")
            print("**************starts*******************")
            # print("\nTargets:\n", it[1])
            # print("\nTarget Weights:\n", it[2])
            print("\nImage ID:", it[3]["imgId"])
            print("\nAnnotation ID:", it[3]["annId"])
            print("\nImage Path:", it[3]["imgPath"])
            print("\nCenter:", it[3]["center"])
            print("\nScale:", it[3]["scale"])
            print("\nJoints:\n", it[3]["joints"])
            print("\nJoints Visibility:\n", it[3]["joints_visibility"])
            print("\nScore:", it[3]["score"])
            print("Rotation:", it[3]["rotation"])
            # for i in range(batch_size):
            #     print("Rotation:", it[3]["rotation"][i])
            print("**************ends*******************")
            print("**************", i, "*******************")

            for img in it[0]:
                print(img.shape)
                cnt = cnt + 1
                draw_patches(img, cnt, "cpu")
        COCOIteratorPipeline.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Path for images", type=str, default="/media/sampath/datasets/COCO/coco_10_img_person/val2017/",required = False)
    parser.add_argument("--annotation_path", help="Path for JSON file", type=str, default="/media/sampath/datasets/COCO/coco_10_img_person/annotations/person_keypoints_val2017.json",required = False)
    parser.add_argument("--device", "-d", help="device", type=str, default="cpu", required = True)
    parser.add_argument("--batch_size", "-b", help="batch size", type=int, default=1,required = True)
    parser.add_argument("--display", help="display", type=int, default=None, required = True)
    parser.add_argument("--num_workers", "-w", help="number of DataLoader workers", type=int, default=1,required = False)
    parser.add_argument("--device_id", "-id", help="Device id", type=int, default=0,required = False)
    parser.add_argument("--image_width", "-iw", help="output image width", type=int, default= 288, required = False)
    parser.add_argument("--image_height", "-ih", help="output image height", type=int, default= 384,required = False)
    parser.add_argument("--exp_name", "-n",
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, default=str(datetime.now().strftime("%Y%m%d_%H%M")),required = False)
    parser.add_argument("--epochs", "-e", help="number of epochs", type=int, default=1,required = False)
    parser.add_argument("--lr", "-l", help="initial learning rate", type=float, default=0.001,required = False)
    parser.add_argument("--disable_lr_decay", help="disable learning rate decay", action="store_true",required = False)
    parser.add_argument("--lr_decay_steps", help="learning rate decay steps", type=str, default="(170, 200)",required = False)
    parser.add_argument("--lr_decay_gamma", help="learning rate decay gamma", type=float, default=0.1,required = False)
    parser.add_argument("--optimizer", "-o", help="optimizer name. Currently, only `SGD` and `Adam` are supported.",
                        type=str, default='Adam',required = False)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0.,required = False)
    parser.add_argument("--momentum", "-m", help="momentum", type=float, default=0.9,required = False)
    parser.add_argument("--nesterov", help="enable nesterov", action="store_true",required = False)
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None,required = False)
    parser.add_argument("--checkpoint_path", "-c",
                        help="previous checkpoint path. Checkpoint will be loaded before training starts. It includes "
                             "the model, the optimizer, the epoch, and other parameters.",
                        type=str, default=None,required = False)
    parser.add_argument("--log_path", help="log path. tensorboard logs and checkpoints will be saved here.",
                        type=str, default='./logs',required = False)
    parser.add_argument("--disable_tensorboard_log", "-u", help="disable tensorboard logging", action="store_true",required = False)
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=48,required = False)
    parser.add_argument("--model_nof_joints", help="HRNet nof_joints parameter", type=int, default=17,required = False)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1,required = False)
    parser.add_argument("--disable_flip_test_images", help="disable image flip during evaluation", action="store_true",required = False)
    parser.add_argument("--coco_bbox_path", help="path of detected bboxes to use during evaluation",
                        type=str, default=None,required = False)
    parser.add_argument("--seed", "-s", help="seed", type=int, default=1,required = False)
    parser.add_argument("--is_train", help="flag for is_train (used for randomization)", type=int, default=0, required = False)
    parser.add_argument("--flip_prob", help="Probability for flipping image", type=float, default=0.0, required = False)
    parser.add_argument("--rotate_prob", help="Probability for rotating image", type=float, default=0.5, required = False)
    parser.add_argument("--half_body_prob", help="Probability for halfbody augmentation", type=float, default=0.3, required = False)
    parser.add_argument("--scale_factor", help="scale factor used for defining scale range", type=float, default=0.35, required = False)
    parser.add_argument("--rotation_factor", help="rotation factor used for defining rotation angle range", type=float, default=45.0, required = False)
    args = parser.parse_args()

    main(**args.__dict__)