# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

import cupy as cp
import numpy as np
import rocal_pybind as b
import amd.rocal.types as types


class ROCALGenericIterator(object):
    def __init__(self, pipeline, tensor_layout=types.NCHW, reverse_channels=False, multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], tensor_dtype=types.FLOAT, display=False, device="cpu", device_id=0):
        self.loader = pipeline
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.device = device
        self.device_id = device_id
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.display = display
        self.batch_size = pipeline._batch_size
        if self.loader._name is None:
            self.loader._name = self.loader._reader
        self.labels_size = ((self.batch_size * self.loader._num_classes) if self.loader._one_hot_encoding else self.batch_size)
        self.output_list = self.dimensions = self.dtype = None
        self.labels_tensor = None
        self.len = b.getRemainingImages(self.loader._handle) // self.batch_size  # iteration length

    def next(self):
        return self.__next__()

    def __next__(self):

        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.getOutputTensors()

        if self.output_list is None:  # Checking if output_list is empty and initializing the buffers
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                if self.device == "cpu":
                    self.dtype = self.output_tensor_list[i].dtype()
                    self.output = np.empty(self.dimensions, dtype=self.dtype)
                    self.labels = np.empty(self.labels_size, dtype=self.dtype)
                else:
                    self.dtype = self.output_tensor_list[i].dtype()
                    with cp.cuda.Device(device=self.device_id):
                        self.output = cp.empty(self.dimensions, dtype=self.dtype)
                        self.labels = cp.empty(self.labels_size, dtype=self.dtype)

                if self.device == "cpu":
                    self.output_tensor_list[i].copy_data_numpy(self.output)
                else:
                    self.output_tensor_list[i].copy_data_cupy(self.output.data.ptr)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                if self.device == "cpu":
                    self.output_tensor_list[i].copy_data_numpy(self.output_list[i])
                else:
                    self.output_tensor_list[i].copy_data_cupy(self.output_list[i].data.ptr)

        if (self.loader._name == "labelReader"):
            if (self.loader._one_hot_encoding == True):
                self.loader.getOneHotEncodedLabels(self.labels, self.device)
                self.labels_tensor = self.labels.reshape(-1, self.batch_size, self.loader._num_classes)
            else:
                if self.display:
                    for output in self.output_tensor_list:
                        for i in range(self.batch_size):
                            draw_patches(output[i], i, 0)
                self.labels = self.loader.getImageLabels()
                if self.device == "cpu":
                    self.labels_tensor = self.labels.astype(dtype=np.int_)
                else:
                    with cp.cuda.Device(device=self.device_id):
                        self.labels_tensor = self.labels.astype(dtype=cp.int_)

        return self.output_list, self.labels_tensor

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def __del__(self):
        b.rocalRelease(self.loader._handle)


class ROCALClassificationIterator(ROCALGenericIterator):
    """
    ROCAL iterator for classification tasks for generic images. It returns 2 outputs
    (data and label) in the form of numpy/cupy Tensor.

    Calling

    .. code-block:: python

       ROCALClassificationIterator(pipelines, size)

    is equivalent to calling

    .. code-block:: python

       ROCALGenericIterator(pipelines, ["data", "label"], size)

    Please keep in mind that Tensors returned by the iterator are
    still owned by ROCAL. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Number of samples in the epoch (Usually the size of the dataset).
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the ROCAL pipeline can
                 change during execution. If True, the numpy tensor will be resized accordingly
                 if the shape of ROCAL returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by ROCAL is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one.

    Example
    -------
    With the data set [1,2,3,4,5,6,7] and the batch size 2:
    fill_last_batch = False, last_batch_padded = True  -> last batch = [7], next iteration will return [1, 2]
    fill_last_batch = False, last_batch_padded = False -> last batch = [7], next iteration will return [2, 3]
    fill_last_batch = True, last_batch_padded = True   -> last batch = [7, 7], next iteration will return [1, 2]
    fill_last_batch = True, last_batch_padded = False  -> last batch = [7, 1], next iteration will return [2, 3]
    """

    def __init__(self,
                 pipelines,
                 size=0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 display=False,
                 device="cpu",
                 device_id=0):
        pipe = pipelines
        super(ROCALClassificationIterator, self).__init__(pipe, tensor_layout=pipe._tensor_layout, tensor_dtype=pipe._tensor_dtype,
                                                          multiplier=pipe._multiplier, offset=pipe._offset, display=display, device=device, device_id=device_id)


def draw_patches(img, idx, bboxes):
    # image is expected as a tensor, bboxes as numpy
    import cv2
    img = img.cpu()
    image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.UMat(image).get()
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)
