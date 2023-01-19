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

#include "box_iou_matcher_hip.h"

__device__ __forceinline__ float CalculateIou(const float4 &b1, const float4 &b2) {
    float l = fmaxf(b1.x, b2.x);
    float t = fmaxf(b1.y, b2.y);
    float r = fminf(b1.z, b2.z);
    float b = fminf(b1.w, b2.w);
    float first = fmaxf(r - l, 0.0f);
    float second = fmaxf(b - t, 0.0f);
    volatile float intersection = first * second;
    volatile float area1 = (b1.w - b1.y) * (b1.z - b1.x);
    volatile float area2 = (b2.w - b2.y) * (b2.z - b2.x);

    return intersection / (area1 + area2 - intersection);
}

__device__ inline void FindBestMatch(const int N, volatile float *vals, volatile int *idx) {
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (vals[threadIdx.x] <= vals[threadIdx.x + stride]) {
        if (vals[threadIdx.x] == vals[threadIdx.x + stride]) {
          idx[threadIdx.x] = max(idx[threadIdx.x], idx[threadIdx.x + stride]);
        } else {
          vals[threadIdx.x] = vals[threadIdx.x + stride];
          idx[threadIdx.x] = idx[threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }
}

__device__ void MatchBoxWithAnchors(const float4 &box, const int box_idx, unsigned int anchor_count, const float4 *anchors,
                                    volatile int *best_anchor_idx_buf, volatile float *best_anchor_iou_buf,
                                    volatile int *best_box_idx, volatile float *best_box_iou) {
    float best_anchor_iou = -1.0f;
    int best_anchor_idx = -1;

    for (unsigned int anchor = threadIdx.x; anchor < anchor_count; anchor += blockDim.x) {
      float new_val = CalculateIou(box, anchors[anchor]);
      if (new_val >= best_anchor_iou) {
          best_anchor_iou = new_val;
          best_anchor_idx = anchor;
      }
      if (new_val >= best_box_iou[anchor]) {
          best_box_iou[anchor] = new_val;
          best_box_idx[anchor] = box_idx;
      }
    }
    best_anchor_iou_buf[threadIdx.x] = best_anchor_iou;
    best_anchor_idx_buf[threadIdx.x] = best_anchor_idx;
}

__device__ void getLowQualityPreds(const float4 &box, unsigned int anchor_count, const float4 *anchors, int *low_quality_preds, float max_iou)
{
    float best_anchor_iou = -1.0f;
    int best_anchor_idx = -1;
    for (unsigned int anchor = threadIdx.x; anchor < anchor_count; anchor += blockDim.x) {
      float new_val = CalculateIou(box, anchors[anchor]);
      if(fabs(new_val - max_val) < 1e-6)
        low_quality_preds[anchor] = anchor;
    }
}

__device__ void WriteMatchesToOutput(unsigned int anchor_count, float high_threshold, float low_threshold, volatile int *best_box_idx, volatile float *best_box_iou) {

    for (unsigned int anchor = threadIdx.x; anchor < anchor_count; anchor += blockDim.x)
    {
        if (best_box_iou[anchor] < low_threshold)
            best_box_idx[anchor] = -1;
        else if(best_box_iou[anchor] >= low_threshold && best_box_iou[anchor] < high_threshold)
            best_box_idx[anchor] = -2;
    }
}

template <int BLOCK_SIZE>
__global__ void __attribute__((visibility("default")))
BoxIoUMatcher(const BoxIoUMatcherSampleDesc *samples, const int anchor_cnt, const float4 *anchors,
              const float high_thresold, const float low_thresold, bool allow_low_quality_matches,
              int *box_idx_buffer, float *box_iou_buffer, int *all_matches, int *low_quality_preds)
{
    const int sample_idx = blockIdx.x;
    const auto &sample = samples[sample_idx];

    __shared__ volatile int best_anchor_idx_buf[BLOCK_SIZE];
    __shared__ volatile float best_anchor_iou_buf[BLOCK_SIZE];

    volatile int *best_box_idx = box_idx_buffer + sample_idx * anchor_cnt;
    volatile float *best_box_iou = box_iou_buffer + sample_idx * anchor_cnt;
    int *all_matches_buf = all_matches + sample_idx * anchor * anchor_cnt;

    for (int box_idx = 0; box_idx < sample.in_box_count; box_idx++)
    {
      MatchBoxWithAnchors(
        sample.boxes_in[box_idx],
        box_idx,
        anchor_cnt,
        anchors,
        best_anchor_idx_buf,
        best_anchor_iou_buf,
        best_box_idx,
        best_box_iou);

      __syncthreads();

      // FindBestMatch(blockDim.x, best_anchor_iou_buf, best_anchor_idx_buf);
      // __syncthreads();

      // if (threadIdx.x == 0) {
      //   int idx = best_anchor_idx_buf[0];
      //   float iou = best_anchor_iou_buf[0];
      //   all_matches[idx] = box_idx;
      // }
      // __syncthreads();

      // if(allow_low_quality_matches)
      //   getLowQualityPreds(box, anchor_count, anchors, low_quality_preds, best_anchor_idx_buf[0]);
      // __syncthreads();
    }

    __syncthreads();

    WriteMatchesToOutput(
      anchor_cnt,
      high_thresold,
      low_thresold,
      best_box_idx,
      best_box_iou);
}

void BoxIoUMatcherGpu::prepare_anchors(const std::vector<float> &anchors) {

    if ((anchors.size() % 4) != 0)
        THROW("BoxIoUMatcherGpu anchors not a multiple of 4");

    int anchor_count = anchors.size() / 4;
    int anchor_data_size = anchor_count * 4 * sizeof(float);
    auto anchors_data_cpu = reinterpret_cast<const float4 *>(anchors.data());
    HIP_ERROR_CHECK_STATUS(hipMemcpy((void *)_anchors_data_dev, anchors.data(), anchor_data_size, hipMemcpyHostToDevice));
}

std::pair<int *, float *> BoxIoUMatcherGpu::ResetBuffers() {
    HIP_ERROR_CHECK_STATUS(hipMemsetAsync(_best_box_idx_dev, 0, _cur_batch_size * _anchor_count * sizeof(int), _stream));
    HIP_ERROR_CHECK_STATUS(hipMemsetAsync(_best_box_iou_dev, 0, _cur_batch_size * _anchor_count * sizeof(float), _stream));
    return std::make_pair(_best_box_idx_dev, _best_box_iou_dev);
}

void BoxIoUMatcherGpu::Run(pMetaDataBatch full_batch_meta_data, int *matched_indices) {

    if (_cur_batch_size != full_batch_meta_data->size() || (_cur_batch_size <= 0))
        THROW("BoxIoUMatcherGpu::Run Invalid input metadata");

    // _best_box_idx_dev = matched_indices;
    const auto buffers = ResetBuffers();    // reset temp buffers
    int total_num_boxes = 0;
    for (int i = 0; i < _cur_batch_size; i++) {
        auto sample = &_samples_host_buf[i];
        sample->in_box_count = full_batch_meta_data->get_bb_labels_batch()[i].size();
        total_num_boxes += sample->in_box_count;
    }

    if (total_num_boxes > MAX_NUM_BOXES_TOTAL)
        THROW("BoxIoUMatcherGpu::Run total_num_boxes exceeds max");

    float *boxes_in_temp = _boxes_in_dev;
    for (int sample_idx = 0; sample_idx < _cur_batch_size; sample_idx++) {
        auto sample = &_samples_host_buf[sample_idx];
        HIP_ERROR_CHECK_STATUS(hipMemcpyHtoDAsync((void *)boxes_in_temp, full_batch_meta_data->get_bb_cords_batch()[sample_idx].data(), sample->in_box_count * sizeof(float) * 4, _stream));

        sample->boxes_in = reinterpret_cast<const float4 *>(boxes_in_temp);
        boxes_in_temp += (sample->in_box_count * 4);
        _output_shape.push_back(std::vector<size_t>(1, _anchor_count));
    }

    // if there is no mapped memory, do explicit copy from host to device
    if (!_pinnedMem)
        HIP_ERROR_CHECK_STATUS(hipMemcpyHtoD(_samples_dev_buf, _samples_host_buf, _cur_batch_size*sizeof(BoxIoUMatcherSampleDesc)));
    HIP_ERROR_CHECK_STATUS(hipStreamSynchronize(_stream));

    // call the kernel for box iou matching
    hipLaunchKernelGGL(BoxIoUMatcher<BlockSize>, dim3(_cur_batch_size), dim3(BlockSize), 0, _stream,
                       _samples_dev_buf,
                       _anchor_count,
                       _anchors_data_dev,
                       _high_threshold,
                       _low_threshold,
                       _allow_low_quality_matches,
                       buffers.first,
                       buffers.second,
                       _all_matches_dev,
                       _low_quality_preds_dev);

   hipMemcpyDtoH(buffers.first, _best_box_idx.data(),  _cur_batch_size * _anchor_cnt);
   for(int i = 0; i < bs; i++)
   {
      std::cerr<<"Pritning for image: "<<i<<std::endll;
      for(int j = 0; j < _anchor_cnt; j++)
      {
        std::cerr"Matched indices: "<<_best_box_idx[i * _anchor_cnt + j];
      }
   }
}