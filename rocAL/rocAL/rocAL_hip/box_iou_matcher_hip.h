/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once
#include "commons.h"

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include <vector>
#include "meta_data.h"

// assume total number of boxes per batch for device memory allocation
#define MAX_NUM_BOXES_PER_BATCH     23296 // for SSD Retinanet Max bboxes possible per images is 768, Batchsize used is 32 so 768 x 32 = 23296
#define HIP_ERROR_CHECK_STATUS(call) { hipError_t err = (call); if(err != hipSuccess){ THROW("ERROR: Hip call failed with status " + TOSTR(err))}}

struct BoxIoUMatcherSampleDesc {
  const float4 *boxes_in;
  int in_box_count;
};

class BoxIoUMatcherGpu {
public:
    static constexpr int BlockSize = 256;   // hip kernel blocksize
    explicit BoxIoUMatcherGpu(int batch_size, std::vector<float> &anchors, float high_threshold, float low_threshold, bool allow_low_quality_matches, const hipStream_t &stream, bool pinnedMem):
                          _anchors(anchors), _high_threshold(high_threshold), _low_threshold(low_threshold), _allow_low_quality_matches(allow_low_quality_matches), _stream(stream), _pinnedMem(pinnedMem)
    {
         Initialize(batch_size);
         prepare_anchors(_anchors);
    }
    void Run(pMetaDataBatch full_batch_meta_data, int *matched_indices);

    virtual ~BoxIoUMatcherGpu() { UnInitialize();};

protected:
    void Initialize(int cur_batch_size) {
        _cur_batch_size = cur_batch_size;
        _anchor_count = _anchors.size() / 4;
        _best_box_idx.resize(cur_batch_size * _anchor_count);
        _best_box_iou.resize(cur_batch_size * _anchor_count);

        hipError_t err = hipHostMalloc((void **)&_samples_host_buf, cur_batch_size * sizeof(BoxIoUMatcherSampleDesc), hipHostMallocDefault/*hipHostMallocMapped|hipHostMallocWriteCombined*/);
        if(err != hipSuccess || !_samples_host_buf)
            THROW("hipHostMalloc failed for BoxIoUMatcherSampleDesc" + TOSTR(err));

        if (_pinnedMem)
            err = hipHostGetDevicePointer((void **)&_samples_dev_buf, _samples_host_buf, 0 );
        else
            err = hipMalloc((void **)&_samples_dev_buf, cur_batch_size * sizeof(BoxIoUMatcherSampleDesc));
        if(err != hipSuccess || !_samples_dev_buf)
            THROW("hipMalloc failed for BoxIoUMatcherSampleDesc" + TOSTR(err));

        // allocate device buffers
        HIP_ERROR_CHECK_STATUS(hipMalloc((void **)&_anchors_data_dev, _anchor_count * 4 * sizeof(float)));
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_boxes_in_dev, MAX_NUM_BOXES_PER_BATCH * sizeof(float) * 4));
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_best_box_idx_dev, _best_box_idx.size() * sizeof(float)));
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_best_box_iou_dev, _best_box_iou.size() * sizeof(float)));
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_low_quality_preds_dev, _anchor_count * cur_batch_size * sizeof(int)));
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_anchor_iou_dev, _anchor_count * cur_batch_size * sizeof(float)));
    }

    void UnInitialize() {
        if (_samples_host_buf) HIP_ERROR_CHECK_STATUS(hipHostFree(_samples_host_buf));
        if (!_pinnedMem) HIP_ERROR_CHECK_STATUS(hipFree(_samples_dev_buf));
        if (_anchors_data_dev) HIP_ERROR_CHECK_STATUS(hipFree(_anchors_data_dev));
        if (_boxes_in_dev) HIP_ERROR_CHECK_STATUS(hipFree(_boxes_in_dev));
        if (_best_box_iou_dev) HIP_ERROR_CHECK_STATUS(hipFree(_best_box_iou_dev));
        if (_best_box_idx_dev) HIP_ERROR_CHECK_STATUS(hipFree(_best_box_idx_dev));
        if (_low_quality_preds_dev) HIP_ERROR_CHECK_STATUS(hipFree(_low_quality_preds_dev));
        if (_anchor_iou_dev) HIP_ERROR_CHECK_STATUS(hipFree(_anchor_iou_dev));
    }

private:
    void prepare_anchors(const std::vector<float> &anchors);
    std::pair<int*, float*> ResetBuffers();

    int _cur_batch_size;
    std::vector<float> _anchors;
    const float _low_threshold, _high_threshold;
    const hipStream_t _stream;
    bool  _pinnedMem;
    bool _allow_low_quality_matches;

    int _anchor_count;
    float *_boxes_in_dev;
    std::vector<int> _best_box_idx;
    std::vector<float> _best_box_iou;
    int *_best_box_idx_dev;
    float *_best_box_iou_dev;
    int * _low_quality_preds_dev;
    float *_anchor_iou_dev;
    std::vector<BoxIoUMatcherSampleDesc *> _samples;
    BoxIoUMatcherSampleDesc *_samples_host_buf, *_samples_dev_buf;
    float4 *_anchors_data_dev;
    std::vector<std::vector<size_t>> _output_shape;

};