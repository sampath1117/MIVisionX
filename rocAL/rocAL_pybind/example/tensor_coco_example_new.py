from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import itertools

import torch
import numpy as np
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
# import rali_pybind.tensor
import sys
import cv2
import os
import math
from math import sqrt
import ctypes

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

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False, num_anchors=120087):

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
        print("\nself.tensor_dtype : ", self.tensor_dtype)
        self.device = device
        print(self.device)
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.num_anchors = num_anchors
        self.display = True

        #Number of batch size handled by each GPU
        self.len = math.ceil(self.loader.getRemainingImages()/self.bs)
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 3), dtype="int32")

    def next(self):
        return self.__next__()

    def __next__(self):
        if(self.loader.isEmpty()):
            raise StopIteration
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.rocalGetOutputTensors()

        #From init

        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels
        self.w = self.output_tensor_list[0].batch_width()
        self.h = self.output_tensor_list[0].batch_height()
        self.bs = self.output_tensor_list[0].batch_size()
        #print("\n self.bs: ", self.bs)
        self.color_format = self.output_tensor_list[0].color_format()

        torch_gpu_device = torch.device('cuda', self.device_id)

        if self.device == "gpu":
            #NHWC default for now
            self.out = torch.empty((self.bs, self.color_format, self.h, self.w), dtype=torch.uint8, device=torch_gpu_device)
            self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))
            print("\nImages : ", self.out)

            # 1D labels & bboxes array
            labels_array, boxes_array = self.loader.getEncodedBoxesAndLables(self.bs, int(self.num_anchors))
            self.encoded_bboxes = torch.as_tensor(boxes_array, dtype=torch.float32, device=torch_gpu_device)
            self.encoded_bboxes = self.encoded_bboxes.view(self.bs, self.num_anchors, 4)
            self.encoded_labels = torch.as_tensor(labels_array, dtype=torch.int32, device=torch_gpu_device)
            encoded_bboxes_tensor = self.encoded_bboxes.cpu()
            encodded_labels_tensor = self.encoded_labels.cpu()
            #print("\n Self.encoded_labels : ", self.encoded_labels)
            #print("\n Self.encoded_boxes : ", self.encoded_bboxes)
        else:
            #NCHW default for now
            #self.out = torch.empty((self.bs, self.color_format, self.h, self.w), dtype=torch.float32)
            self.out = torch.empty((self.bs, self.color_format, self.h, self.w,), dtype=torch.float32)
            self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))
            #print("\nImages : ", self.out)

            labels_array = self.loader.rocalGetBoundingBoxLabel()
            encodded_labels_tensor = []
            encoded_bboxes_tensor = []
            for label in labels_array:
                self.encoded_labels = torch.as_tensor(label, dtype=torch.int64)
                encodded_labels_tensor.append(self.encoded_labels)
            #print("\n encodded_labels_tensor : ", encodded_labels_tensor)

            boxes_array = self.loader.rocalGetBoundingBoxCords()
            for box in boxes_array:
                self.encoded_bboxes = torch.as_tensor(box, dtype=torch.float16)
                self.encoded_bboxes = self.encoded_bboxes * 800
                self.encoded_bboxes = self.encoded_bboxes.view(-1, 4)
                encoded_bboxes_tensor.append(self.encoded_bboxes)
            #print("\n encoded_bboxes_tensor : ", encoded_bboxes_tensor)

            matched_idxs = self.loader.rocalGetMatchedIndices()
            self.matched_idxs = torch.as_tensor(matched_idxs, dtype=torch.int64)
            matched_idxs_tensor = self.matched_idxs.view(-1, 120087)
            # if self.device == "cpu":
            #     for matched_index in matched_idxs:
            #         if(matched_index != -1):
            #             print("Matched indices:",matched_index)

        # Image sizes and Image id of a batch
        self.loader.GetImgSizes(self.img_size)
        image_size_tensor = torch.tensor(self.img_size).view(self.bs, 3)

        '''
        for i in range(self.bs):
            index_list = []
            actual_bboxes = []
            actual_labels = []
            for idx, x in enumerate(encodded_labels_tensor[i]):
                if x != 0:
                    index_list.append(idx)
                    actual_bboxes.append(encoded_bboxes_tensor[i][idx].tolist())
                    actual_labels.append(encodded_labels_tensor[i][idx].tolist())
            #print("\n Actual boxes : ", actual_bboxes)
            #print("\n Actual labels : ", actual_labels)
            #if self.display:
            img = self.out
            draw_patches(img[i], image_size_tensor[0][i][2].item(),
                            actual_bboxes, self.device)
        '''

        targets = { 'boxes' : encoded_bboxes_tensor,
                    'labels' : encodded_labels_tensor,
                    'image_id' : image_size_tensor[:, 2:3].cuda(),
                    'original_image_size' : image_size_tensor[:, 0:2].cuda(),
                    'matched_idxs' : matched_idxs_tensor
        }
        return (self.out).to(torch.float16), targets
        #return (self.out), encoded_bboxes_tensor, encodded_labels_tensor, image_id_tensor, image_size_tensor

    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img, idx, bboxes, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    htot, wtot, _ = img.shape

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        #print("\n loc : ", loc_)
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        #image = cv2.rectangle(image, (int(loc_[0]*wtot), int(loc_[1] * htot)), (int(
        #    (loc_[2] * wtot)), int((loc_[3] * htot))), color, thickness)
        image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
            (loc_[2])), int((loc_[3]))), color, thickness)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER/" + str(idx)+"_"+"train"+".png", image)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_path annotation_path cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    image_path = sys.argv[1]
    annotation_path = sys.argv[2]
    device = sys.argv[3]
    if(device == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=800

    local_rank = 0
    world_size = 1

    rali_device = 'gpu'
    decoder_device = 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    # Anchors - load default anchors from a text file
    with open('/media/sampath/Default_anchors_retinanet_1.txt', 'r') as f_read:
        anchors = f_read.readlines()
    anchor_list = [float(x.strip())/800 for x in anchors]
    f_read.close()

    print("*********************************************************************")

    coco_train_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id,seed=random_seed, rocal_cpu=_rali_cpu)

    with coco_train_pipeline:
        jpegs, bboxes, labels = fn.readers.coco(file_root=image_path,
                                                 annotations_file=annotation_path,
                                                 random_shuffle=False,
                                                 shard_id=local_rank,
                                                 num_shards=world_size,
                                                 seed=random_seed,
                                                 is_box_encoder=False)

        print("*********************** SHARD ID ************************",local_rank)
        print("*********************** NUM SHARDS **********************",world_size)
        images_decoded = fn.decoders.image(jpegs, device=decoder_device, output_type = types.RGB, file_root=image_path, annotations_file=annotation_path, random_shuffle=False,shard_id=local_rank, num_shards=world_size)
        #res_images = fn.resize(images_decoded, device=rali_device, resize_width=crop, resize_height=crop, rocal_tensor_layout = types.NHWC, rocal_tensor_output_type = types.UINT8)
        # flip_coin = fn.random.coin_flip(probability=0.5)
        # images = fn.resize_mirror_normalize(images_decoded, device="gpu",
        #                                     image_type=types.RGB,
        #                                     resize_width=crop, resize_height=crop,
        #                                     mirror=flip_coin,
        #                                     rocal_tensor_layout = types.NCHW,
        #                                     rocal_tensor_output_type = types.FLOAT,
        #                                     mean=[0.485*255,0.456*255 ,0.406*255 ],
        #                                     std=[0.229*255 ,0.224*255 ,0.225*255 ])
        Matched_idxs  = fn.box_iou_matcher(anchors=anchor_list, criteria=0.5,
                                     high_threshold=0.5, low_threshold=0.4,
                                     allow_low_quality_matches=True)

        coco_train_pipeline.set_outputs(images_decoded)
    coco_train_pipeline.build()
    COCOIteratorPipeline = ROCALCOCOIterator(coco_train_pipeline)
    cnt = 0
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(COCOIteratorPipeline):
            temp =1
            # print("************************************** i *************************************",i)
            # for img in it[0]:
            #     print(img.shape)
            #     cnt = cnt + 1
            #     draw_patches(img, cnt, "cpu")
        COCOIteratorPipeline.reset()
    print("*********************************************************************")
    exit(0)

    import timeit
    start = timeit.default_timer() #Timer starts



    stop = timeit.default_timer() #Timer Stops
    print('\n Time: ', stop - start)

if __name__ == '__main__':
    main()
