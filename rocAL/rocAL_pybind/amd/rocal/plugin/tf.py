# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import rocal_pybind as b
import amd.rocal.types as types


class ROCALGenericImageIterator(object):
    def __init__(self, pipeline):
        self.loader = pipeline
        self.output_list = None
        self.bs = pipeline._batch_size

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.getOutputTensors()

        if self.output_list is None:
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                self.dtype = self.output_tensor_list[i].dtype()
                self.output = np.empty(self.dimensions, dtype = self.dtype)

                self.output_tensor_list[i].copy_data_numpy(self.output)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data_numpy(self.output_list[i])
        return self.output_list

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self


class ROCALGenericIteratorDetection(object):
    def __init__(self, pipeline, tensor_layout=types.NCHW, reverse_channels=False, multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):
        self.loader = pipeline
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.bs = pipeline._batch_size
        self.output_list = self.dimensions = self.dtype = None
        if self.loader._name is None:
            self.loader._name = self.loader._reader

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.getOutputTensors()

        if self.output_list is None:
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                self.dtype = self.output_tensor_list[i].dtype()
                self.output = np.empty(self.dimensions, dtype = self.dtype)
                self.output_tensor_list[i].copy_data_numpy(self.output)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data_numpy(self.output_list[i])

        if (self.loader._name == "TFRecordReaderDetection"):
            self.bbox_list = []
            self.label_list = []
            self.num_bboxes_list = []
            # Count of labels/ bboxes in a batch
            self.count_batch = self.loader.getBoundingBoxCount()
            self.num_bboxes_list = self.bboxes_label_count.tolist()
            # 1D labels array in a batch
            self.labels = self.loader.getBoundingBoxLabels()
            # 1D bboxes array in a batch
            self.bboxes = self.loader.getBoundingBoxCords()
            # 1D Image sizes array of image in a batch
            self.img_size = np.zeros((self.bs * 2), dtype="int32")
            self.loader.getImgSizes(self.img_size)
            count = 0  # number of bboxes per image
            sum_count = 0  # sum of the no. of the bboxes
            for i in range(self.bs):
                count = self.count_batch[i]
                self.label_2d_numpy = self.labels[i]
                self.label_2d_numpy = np.reshape(self.label_2d_numpy, (-1, 1)).tolist()
                self.bb_2d_numpy = (self.bboxes[i])
                self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()
                self.label_list.append(self.label_2d_numpy)
                self.bbox_list.append(self.bb_2d_numpy)
                sum_count = sum_count + count

            self.target = self.bbox_list
            self.target1 = self.label_list
            max_cols = max([len(row)
                           for batch in self.target for row in batch])
            # max_rows = max([len(batch) for batch in self.target])
            max_rows = 100
            bb_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
            bb_padded_1 = [row + [0] * (max_cols - len(row))
                           for batch in bb_padded for row in batch]
            arr = np.asarray(bb_padded_1)
            self.res = np.reshape(arr, (-1, max_rows, max_cols))
            max_cols = max([len(row)
                           for batch in self.target1 for row in batch])
            # max_rows = max([len(batch) for batch in self.target1])
            max_rows = 100
            lab_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target1]
            lab_padded_1 = [row + [0] * (max_cols - len(row))
                            for batch in lab_padded for row in batch]
            labarr = np.asarray(lab_padded_1)
            self.l = np.reshape(labarr, (-1, max_rows, max_cols))
            self.num_bboxes_arr = np.array(self.num_bboxes_list)

            return self.output_tensor_list, self.res, self.l, self.num_bboxes_arr
        elif (self.loader._name == "TFRecordReaderClassification"):
            if (self.loader._one_hot_encoding == True):
                self.labels = np.zeros((self.bs)*(self.loader._num_classes), dtype="int32")
                self.loader.getOneHotEncodedLabels(self.labels, device="cpu")
                self.labels = np.reshape(self.labels, (-1, self.bs, self.loader._num_classes))
            else:
                self.labels = self.loader.getImageLabels()

            return self.output_tensor_list, self.labels

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __del__(self):
        b.rocalRelease(self.loader._handle)

class ROCALIterator(ROCALGenericIteratorDetection):
    """
    ROCAL iterator for detection and classification tasks for PyTorch. It returns 2 or 3 outputs
    (data and label) or (data , bbox , labels) in the form of PyTorch's Tensor.
    Calling
    .. code-block:: python
       ROCALIterator(pipelines, size)
    is equivalent to calling
    .. code-block:: python
       ROCALGenericIteratorDetection(pipelines, ["data", "label"], size)


    """

    def __init__(self,
                 pipelines,
                 size=0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(ROCALIterator, self).__init__(pipe, tensor_layout=pipe._tensor_layout, tensor_dtype=pipe._tensor_dtype,
                                            multiplier=pipe._multiplier, offset=pipe._offset)


class ROCAL_iterator(ROCALGenericImageIterator):
    """
    ROCAL iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

    """

    def __init__(self,
                 pipelines,
                 size=0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(ROCAL_iterator, self).__init__(pipe)
