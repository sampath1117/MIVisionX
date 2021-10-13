from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt

import torch
import random
import itertools

from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import numpy as np

#additional packages used in HRNet
from datetime import datetime
import argparse

class COCOPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, data_dir, ann_dir, default_boxes,  crop, rali_cpu=True, display=False, output_image_width = 288, output_image_height = 384, sigma = 3.0):
        super(COCOPipeline, self).__init__(batch_size, num_threads,
                                           device_id, seed=seed, rali_cpu=rali_cpu)
        
        print("Entered init function")
        keypoint = True
        self.input = ops.COCOReader(
            file_root=data_dir, annotations_file=ann_dir, random_shuffle=True, seed=seed, sigma  = sigma, is_keypoint = keypoint, output_image_width = output_image_width, output_image_height = output_image_height)
        
        print("Complete reading data")
        rali_device = 'cpu' if rali_cpu else 'gpu'
        decoder_device = 'cpu' if rali_cpu else 'mixed'

        self.decode = ops.ImageDecoder(
            device=decoder_device, output_type=types.RGB)

        self.res = ops.Resize(device=rali_device, resize_x=crop, resize_y=crop)
        self.twist = ops.ColorTwist(device=rali_device)

        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.05, 0.05])
        self.coin_flip = ops.CoinFlip(probability=0.5)
        self.flip = ops.Flip(flip=1)
        

    def define_graph(self):
        coin = self.coin_flip()
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()

        self.jpegs,self.bb,self.labels= self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = self.flip(images)
        output = self.twist(images)

        
        # Encoded Bbox and labels output in "xcycwh" format
        return [output,self.bb,self.labels]


class RALICOCOIterator(object):
    """
    COCO RALI iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, display=False):

        # self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.bs = self.loader._batch_size
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim = self.loader.getRemainingImages()
        self.display = display
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        if self.tensor_dtype == types.FLOAT:
            self.out = np.zeros(
                (self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype="float32")
        elif self.tensor_dtype == types.FLOAT16:
            self.out = np.zeros(
                (self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype="float16")

    def next(self):
        return self.__next__()

    def __next__(self):
        #print("In the next routine of COCO Iterator")
        if(self.loader.isEmpty()):
            timing_info = self.loader.Timing_Info()
            print("Load     time ::", timing_info.load_time)
            print("Decode   time ::", timing_info.decode_time)
            print("Process  time ::", timing_info.process_time)
            print("Transfer time ::", timing_info.transfer_time)
            raise StopIteration

        

        if self.loader.run() != 0:
            raise StopIteration
        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels

        if(types.NCHW == self.tensor_format):
            self.loader.copyToTensorNCHW(
                self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        else:
            self.loader.copyToTensorNHWC(
                self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))

        self.joints_data = dict({})
        self.loader.GetJointsData(self.joints_data)
        print("Joints data is: ")
        print(self.joints_data)
        
        # Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        self.loader.GetImageId(self.image_id)
        # print(self.image_id)

        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.loader.GetImgSizes(self.img_size)
        # print("Image sizes:", self.img_size)

        self.joints = np.zeros((self.bs * 17 * 2), dtype = "float32")
        self.joints_vis = np.zeros((self.bs * 17 * 2), dtype = "float32")
        self.loader.GetImageKeyPoints(self.joints, self.joints_vis)

        # print(self.joints)

        self.targets = np.zeros((self.bs * 17 * 96 * 72), dtype = "float32")
        self.target_weights = np.zeros((self.bs * 17 ), dtype = "float32")
        self.loader.GetImageTargets(self.targets, self.target_weights)

        # print(self.targets.reshape((self.bs * 17,96,72)))
        target_tensor = torch.tensor(self.targets).view(-1, self.bs , 17, 96, 72)
        target_weights_tensor = torch.tensor(self.target_weights).view( -1, self.bs ,  17)
        # joints_data_tensor = torch.tensor(self.joints_data)
        joints_data_tensor = self.joints_data
        # print(target_tensor.shape)
        # print(target_tensor.shape)

        cnt = 0
    
        # for k in range(self.bs * 17):
        #     for i in range(96):
        #         for j in range(71):
        #             cnt  = cnt+1
        #             if(self.targets[cnt]!=0):
        #                 print(self.targets[cnt],end=" ")
                # print("")
            # print("")

        #print("Targets\n",self.targets)
        #print("Target Weights\n",self.target_weights)
        # for i in range(self.bs):
        for i in range(self.bs):
            if self.display:
                img = torch.from_numpy(self.out)
                draw_patches(img[i], self.image_id[i])
        
        if self.tensor_dtype == types.FLOAT:
            return torch.from_numpy(self.out) ,target_tensor , target_weights_tensor , joints_data_tensor
        elif self.tensor_dtype == types.FLOAT16:
            return torch.from_numpy(self.out.astype(np.float16)),target_tensor, target_weights_tensor , joints_data_tensor


    def reset(self):
        self.loader.raliResetLoaders()

    def __iter__(self):
        return self


def draw_patches(img, idx):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, htot, wtot = img.shape
    
    image = cv2.UMat(image).get()
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)


def main(exp_name,
         epochs=210,
         batch_size=1,
         num_workers=4,
         lr=0.001,
         disable_lr_decay=False,
         lr_decay_steps='(170, 200)',
         lr_decay_gamma=0.1,
         optimizer='Adam',
         weight_decay=0.,
         momentum=0.9,
         nesterov=False,
         pretrained_weight_path=None,
         checkpoint_path=None,
         log_path='./logs',
         disable_tensorboard_log=False,
         model_c=48,
         model_nof_joints=17,
         model_bn_momentum=0.1,
         disable_flip_test_images=False,
         image_width = 288,
         image_height = 384,
         image_path="../../../datasets/COCO/val2017_person_10_img/val2017",
         ann_path = "../../../datasets/COCO/val2017_person_10_img/annotations/person_keypoints_val2017.json",
         coco_bbox_path=None,
         seed=1,
         device="cpu",
         display=0,
         device_id =0):

    if(device == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(batch_size)
    nt = int(num_workers)
    di = device_id
    crop_size = 300
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    dboxes = []
    gauss_sigma = 3.0


    pipe = COCOPipeline(batch_size=bs, num_threads=nt, device_id=di, seed=random_seed,
                        data_dir=image_path, ann_dir=ann_path, crop=crop_size, rali_cpu=_rali_cpu, default_boxes=dboxes, display=display,
                        output_image_width = image_width,output_image_height = image_height,sigma = gauss_sigma)
    
    pipe.build()


    data_loader = RALICOCOIterator(pipe, multiplier=pipe._multiplier, 
    offset=pipe._offset, display=display)

    epochs = 1
    for epoch in range(int(epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(data_loader):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\nTargets:\n", it[1])
            print("\nTarget Weights:\n", it[2])
            print("\nImage ID:", it[3]["imgId"])
            print("\nAnnotation ID:", it[3]["annId"])
            print("\nImage Path:", it[3]["imgPath"])
            print("\nCenter:", it[3]["center"])
            print("\nScale:", it[3]["scale"])
            print("\nJoints:\n", it[3]["joints"])
            print("\nJoints Visibility:\n", it[3]["joints_visibility"])
            print("\nScore:", it[3]["score"])
            print("\nRotation:", it[3]["rotation"])
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Path for images", type=str, default="../../../datasets/COCO/val2017_person_10_img/val2017",required = False)
    parser.add_argument("--ann_path", help="Path for JSON file", type=str, default="../../../datasets/COCO/val2017_person_10_img/annotations/person_keypoints_val2017.json",required = False)
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
    parser.add_argument("--epochs", "-e", help="number of epochs", type=int, default=200,required = False)
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
    args = parser.parse_args()

    main(**args.__dict__)
