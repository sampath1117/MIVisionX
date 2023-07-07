
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

#include <cstdio>
#if !ENABLE_HIP
#include <CL/cl.h>
#endif
#include <vx_ext_amd.h>

#include <cstring>
#include <stdexcept>

#include "commons.h"
#include "tensor.h"

#include <omp.h>

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

const __m128 xmm_p0 = _mm_set1_ps(0.0f);
const __m128 xmm_p3 = _mm_set1_ps(3.0f);

void compute_diff_square_sum(float &output, float *input, int inputStride, int numElements, float mean)
{
    const int stride = 1;
    if (numElements > 32)
    {
        int currElements = numElements >> 1;
        float tmp1 = 0, tmp2 = 0;

        // reduce first half and accumulate
        compute_diff_square_sum(tmp1, input, stride, currElements, mean);

        // reduce second half and accumulate
        compute_diff_square_sum(tmp2, input + currElements * stride, stride, numElements - currElements, mean);

        tmp1 += tmp2;
        output += tmp1;
    }
    else
    {
        // reduce to a temporary
        float tmp = 0;
        for (int i = 0; i < numElements; i++)
        {
            float curr = (input[i * stride] - mean);
            auto curnew = curr * curr;
            tmp += curnew;
        }

        // accumulate in target value
        output += tmp;
    }
}

void compute_sum(float &output, float *input, int inputStride, int numElements)
{
    const int stride = 1;
    if (numElements > 32)
    {
        int currElements = numElements >> 1;
        float tmp1 = 0, tmp2 = 0;

        // reduce first half and accumulate
        compute_sum(tmp1, input, stride, currElements);

        // reduce second half and accumulate
        compute_sum(tmp2, input + currElements * stride, stride, numElements - currElements);

        tmp1 += tmp2;
        output += tmp1;
    }
    else
    {
        // reduce to a temporary
        float tmp = 0;
        for (int i = 0; i < numElements; i++)
            tmp += input[i * stride];

        // accumulate in target value
        output += tmp;
    }
}

float rpp_rsqrt(float x)
{
    // Use SSE intrinsic and one Newton-Raphson refinement step
    // - faster and less hacky than the hack below.
    __m128 X = _mm_set_ss(x);
    __m128 tmp = _mm_rsqrt_ss(X);
    float y = _mm_cvtss_f32(tmp);
    return y * (1.5f - x * 0.5f * y * y);
}

static void rpp_rsqrt_sse(float *input, int numElements, float eps, float rdiv, float mul)
{
    int i = 0;
    __m128 rdivx4 = _mm_set1_ps(rdiv);
    __m128 mulx4 = _mm_set1_ps(mul * 0.5f);
    if (eps) // epsilon is present - no need for masking, but we need to add it
    {
        __m128 epsx4 = _mm_set1_ps(eps);
        for (; i + 4 <= numElements; i += 4)
        {
            __m128 x = _mm_loadu_ps(&input[i]);
            x = _mm_mul_ps(x, rdivx4);
            x = _mm_add_ps(x, epsx4);
            __m128 y = _mm_rsqrt_ps(x);
            __m128 y2 = _mm_mul_ps(y, y);
            __m128 xy2 = _mm_mul_ps(x, y2);
            __m128 three_minus_xy2 = _mm_sub_ps(xmm_p3, xy2);
            y = _mm_mul_ps(y, three_minus_xy2);
            y = _mm_mul_ps(y, mulx4);
            _mm_storeu_ps(&input[i], y);
        }
    }
    else
    {
        for (; i + 4 <= numElements; i += 4)
        {
            __m128 x = _mm_loadu_ps(&input[i]);
            x = _mm_mul_ps(x, rdivx4);
            __m128 mask = _mm_cmpneq_ps(x, xmm_p0);
            __m128 y = _mm_rsqrt_ps(x);
            y = _mm_and_ps(y, mask);
            __m128 y2 = _mm_mul_ps(y, y);
            __m128 xy2 = _mm_mul_ps(x, y2);
            __m128 three_minus_xy2 = _mm_sub_ps(xmm_p3, xy2);
            y = _mm_mul_ps(y, three_minus_xy2);
            y = _mm_mul_ps(y, mulx4);
            _mm_storeu_ps(&input[i], y);
        }
    }
    if (eps)
    {
        for (; i < numElements; i++)
            input[i] = rpp_rsqrt(input[i] * rdiv + eps) * mul;
    }
    else
    {
        for (; i < numElements; i++)
        {
            float x = input[i] * rdiv;
            input[i] = x ? rpp_rsqrt(x) * mul : 0;
        }
    }
}

void compute_2D_mean(float *srcPtr, float *meanPtr, uint *dims, uint *stride)
{
    float *srcPtrTemp = srcPtr;
    float normFactor = 1.0 / dims[1];
    for(uint i = 0; i < dims[0]; i++)
    {
        meanPtr[i] = 0;
        compute_sum(meanPtr[i], srcPtrTemp, 1, dims[1]);
        srcPtrTemp += stride[1];
        meanPtr[i] = meanPtr[i] * normFactor;
    }
}

void compute_2D_inv_std_dev(float *srcPtr, float *meanPtr, float *stdDevPtr, uint *dims, uint *stride) {

    float *srcPtrTemp = srcPtr;
    float normFactor = (float)(1.0 / dims[1]);
    for(uint i = 0; i < dims[0]; i++)
    {
        stdDevPtr[i] = 0;
        compute_diff_square_sum(stdDevPtr[i], srcPtrTemp, 1, dims[1], meanPtr[i]);
        srcPtrTemp += stride[1];
    }
    rpp_rsqrt_sse(stdDevPtr, (long int)dims[0], 0, normFactor, 1);
}

void normalize_2D_tensor_avx_axis2(float *srcPtr, uint srcStride, float *dstPtr, uint dstStride,
                                   float *meanPtr, float *invStdDevPtr, float shift, uint *dims, uint *paramStride)
{
    float *srcPtrTemp = srcPtr;
    float *dstPtrTemp = dstPtr;
    int paramIdx = 0;
    uint vectorIncrement = 8;
    uint bufferLength = dims[1];
    uint alignedLength = (bufferLength / 8) * 8;
    uint numRows = dims[0];

    __m256 pShift = _mm256_set1_ps(shift);
    for(uint i = 0; i < numRows; i++)
    {
        float *srcPtrTempRow = srcPtrTemp + i * srcStride;
        float *dstPtrTempRow = dstPtrTemp + i * dstStride;

        // set mean and stddev
        float mean = meanPtr[i];
        float invStdDev = invStdDevPtr[i];
        __m256 pMean, pInvStdDev;
        pMean = _mm256_set1_ps(mean);
        pInvStdDev = _mm256_set1_ps(invStdDev);

        uint vectorLoopCount = 0;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += 8)
        {
            __m256 pSrc = _mm256_loadu_ps(srcPtrTempRow);
            __m256 pDst = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(pSrc, pMean), pInvStdDev), pShift);
            _mm256_storeu_ps(dstPtrTempRow, pDst);
            srcPtrTempRow += 8;
            dstPtrTempRow += 8;
        }
        for(; vectorLoopCount < dims[1] ; vectorLoopCount += 8)
             *dstPtrTempRow++ = (*srcPtrTempRow++ - mean) * invStdDev + shift;
    }
}


vx_enum vx_mem_type(RocalMemType mem) {
    switch (mem) {
        case RocalMemType::OCL:
            return VX_MEMORY_TYPE_OPENCL;
        case RocalMemType::HOST:
            return VX_MEMORY_TYPE_HOST;
        case RocalMemType::HIP:
            return VX_MEMORY_TYPE_HIP;
        default:
            throw std::runtime_error("Memory type not valid");
    }
}

vx_size tensor_data_size(RocalTensorDataType data_type) {
    switch (data_type) {
        case RocalTensorDataType::FP32:
            return sizeof(vx_float32);
        case RocalTensorDataType::FP16:
            return sizeof(vx_int16);
        case RocalTensorDataType::UINT8:
            return sizeof(vx_uint8);
        case RocalTensorDataType::UINT32:
            return sizeof(vx_uint32);
        case RocalTensorDataType::INT32:
            return sizeof(vx_int32);
        default:
            throw std::runtime_error("tensor data_type not valid");
    }
}

//! Converts the Rocal data_type to OpenVX
vx_enum interpret_tensor_data_type(RocalTensorDataType data_type) {
    switch (data_type) {
        case RocalTensorDataType::FP32:
            return VX_TYPE_FLOAT32;
        case RocalTensorDataType::FP16:
            return VX_TYPE_FLOAT16;
        case RocalTensorDataType::UINT8:
            return VX_TYPE_UINT8;
        case RocalTensorDataType::INT32:
            return VX_TYPE_INT32;
        case RocalTensorDataType::UINT32:
            return VX_TYPE_UINT32;
        default:
            THROW("Unsupported Tensor type " + TOSTR(data_type))
    }
}

bool operator==(const rocalTensorInfo &rhs, const rocalTensorInfo &lhs) {
    return (rhs.dims() == lhs.dims() &&
            rhs.mem_type() == lhs.mem_type() &&
            rhs.data_type() == lhs.data_type() &&
            rhs.color_format() == lhs.color_format() &&
            rhs.layout() == lhs.layout());
}

void allocate_host_or_pinned_mem(void **ptr, size_t size, RocalMemType mem_type) {
    if (mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
    hipError_t err = hipHostMalloc((void **)ptr, size, hipHostMallocDefault);
    if(err != hipSuccess || !*ptr)
        THROW("hipHostMalloc of size " + TOSTR(size) + " failed " + TOSTR(err))
    err = hipMemset((void *)*ptr, 0, size);
    if(err != hipSuccess)
        THROW("hipMemset of size " + TOSTR(size) + " failed " + TOSTR(err))
#endif
    } else {
        *ptr = (void *)malloc(size);
        memset((void *)*ptr, 0, size);
    }
}

void rocalTensorInfo::reset_tensor_roi_buffers() {
    allocate_host_or_pinned_mem((void **)&_roi_buf, _batch_size * 4 * sizeof(unsigned), _mem_type);
    if (_mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        _roi.reset(_roi_buf, hipHostFree);
#endif
    } else {
        _roi.reset(_roi_buf, free);
    }
    if (_is_image) {
        auto roi = get_roi();
        for (unsigned i = 0; i < _batch_size; i++) {
            roi[i].x2 = _max_dims[0];
            roi[i].y2 = _max_dims[1];
        }
    } else {
        // TODO - For other tensor types
    }
}



void rocalTensorInfo::reallocate_tensor_sample_rate_buffers() {
    _sample_rate = std::make_shared<std::vector<float>>(_batch_size);

    // if (_sample_rate.size()) _sample_rate.clear();
    _sample_rate->resize(_batch_size);
    if (_is_image) {
        THROW("No sample rate available for Image data")
    } else if(!_is_metadata)
    {
        // for (unsigned i = 0; i < _batch_size; i++)
        // {
        //     _sample_rate.at(i) = 0;
        // }
    }
}

rocalTensorInfo::rocalTensorInfo()
    : _type(Type::UNKNOWN),
      _num_of_dims(0),
      _dims({}),
      _mem_type(RocalMemType::HOST),
      _data_type(RocalTensorDataType::FP32) {}

rocalTensorInfo::rocalTensorInfo(std::vector<size_t> dims,
                                 RocalMemType mem_type,
                                 RocalTensorDataType data_type)
    : _type(Type::UNKNOWN),
      _dims(dims),
      _mem_type(mem_type),
      _data_type(data_type) {
    _batch_size = dims.at(0);
    _num_of_dims = dims.size();
    _data_size = tensor_data_size(data_type);
    for (unsigned i = 0; i < _num_of_dims; i++) _data_size *= dims.at(i);
    // std::cerr << "rocalTensorInfo" ;
    if (_num_of_dims <= 3) _is_image = false;
    // std::cerr << "\n rocalTensorInfo 1";
}

void rocalTensor::update_audio_tensor_sample_rate(const std::vector<float> &sample_rate) {
    if (_info.is_image()) {
        THROW("No sample rate available for Image data")
    }
    else if(!_info.is_metadata())
    {
        // if (_info.get_sample_rate().size() >0) _info.get_sample_rate().clear();
        // _info.get_sample_rate()->resize(_info.batch_size());
        for (unsigned i = 0; i < info().batch_size(); i++)
        {
            _info.get_sample_rate()->at(i) = sample_rate[i];
        }
    }
}

void rocalTensor::update_tensor_roi(const std::vector<uint32_t> &width,
                                    const std::vector<uint32_t> &height) {
    if (_info.is_image()) {
        auto max_dims = _info.max_dims();
        unsigned max_width = max_dims.at(0);
        unsigned max_height = max_dims.at(1);

        if (width.size() != height.size())
            THROW("Batch size of Tensor height and width info does not match")

        if (width.size() != info().batch_size())
            THROW("The batch size of actual Tensor height and width different from Tensor batch size " + TOSTR(width.size()) + " != " + TOSTR(info().batch_size()))

        for (unsigned i = 0; i < info().batch_size(); i++) {
            if (width[i] > max_width) {
                WRN("Given ROI width is larger than buffer width for tensor[" + TOSTR(i) + "] " + TOSTR(width[i]) + " > " + TOSTR(max_width))
                _info.get_roi()[i].x2 = max_width;
            } else {
                _info.get_roi()[i].x2 = width[i];
            }
            if (height[i] > max_height) {
                WRN("Given ROI height is larger than buffer height for tensor[" + TOSTR(i) + "] " + TOSTR(height[i]) + " > " + TOSTR(max_height))
                _info.get_roi()[i].y2 = max_height;
            } else {
                _info.get_roi()[i].y2 = height[i];
            }
        }
    }
    else if(!_info.is_metadata())
    {
        auto max_dims = _info.max_dims();
        unsigned max_samples = max_dims.at(0);
        unsigned max_channels = max_dims.at(1);

        auto samples = width;
        auto channels = height;

        if (samples.size() != channels.size())
            THROW("Batch size of Tensor height and width info does not match")

        if (samples.size() != info().batch_size())
            THROW("The batch size of actual Tensor height and width different from Tensor batch size " + TOSTR(samples.size()) + " != " + TOSTR(info().batch_size()))

        for (unsigned i = 0; i < info().batch_size(); i++)
        {
            // std::cerr<< "\n Printing _info.get_roi()[i].x1 "<< samples[i];
            // std::cerr<< "\n Printing _info.get_roi()[i].y1 "<< channels[i];

            if (samples[i] > max_samples)
            {
                ERR("Given ROI width is larger than buffer width for tensor[" + TOSTR(i) + "] " + TOSTR(samples[i]) + " > " + TOSTR(max_samples))
                _info.get_roi()[i].x1 = max_samples;
            }
            else
            {
                _info.get_roi()[i].x1 = samples[i];
            }
            if (channels[i] > max_channels)
            {
                ERR("Given ROI height is larger than buffer with for tensor[" + TOSTR(i) + "] " + TOSTR(channels[i]) + " > " + TOSTR(max_channels))
                _info.get_roi()[i].y1 = max_channels;
            }
            else
            {
                _info.get_roi()[i].y1 = channels[i];
            }
        }
    }
}

rocalTensor::~rocalTensor() {
    _mem_handle = nullptr;
    if (_vx_handle) vxReleaseTensor(&_vx_handle);
}

rocalTensor::rocalTensor(const rocalTensorInfo &tensor_info)
    : _info(tensor_info) {
    _info._type = rocalTensorInfo::Type::UNKNOWN;
    _mem_handle = nullptr;
}

int rocalTensor::create_virtual(vx_context context, vx_graph graph) {
    if (_vx_handle) {
        WRN("Tensor object create method is already called ")
        return -1;
    }

    _context = context;
    _vx_handle = vxCreateVirtualTensor(graph, _info.num_of_dims(), _info.dims().data(), interpret_tensor_data_type(_info.data_type()), 0);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateVirtualTensor(input:[" + TOSTR(_info.max_dims().at(0)) + "W" + TOSTR(_info.max_dims().at(1)) + "H" + "]): failed " + TOSTR(status))

    _info._type = rocalTensorInfo::Type::VIRTUAL;
    return 0;
}

int rocalTensor::create_from_handle(vx_context context) {
    if (_vx_handle) {
        WRN("Tensor object create method is already called ")
        return -1;
    }

    _context = context;
    vx_enum tensor_data_type = interpret_tensor_data_type(_info.data_type());
    unsigned num_of_dims = _info.num_of_dims();
    vx_size stride[num_of_dims];
    void *ptr[1] = {nullptr};

    stride[0] = tensor_data_size(_info.data_type());
    for (unsigned i = 1; i < num_of_dims; i++)
        stride[i] = stride[i - 1] * _info.dims().at(i - 1);

    _vx_handle = vxCreateTensorFromHandle(_context, _info.num_of_dims(), _info.dims().data(), tensor_data_type, 0, stride, ptr, vx_mem_type(_info._mem_type));
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(input: failed " + TOSTR(status))
    _info._type = rocalTensorInfo::Type::HANDLE;
    return 0;
}

int rocalTensor::create(vx_context context) {
    if (_vx_handle) {
        WRN("Tensor object create method is already called ")
        return -1;
    }

    _context = context;
    vx_status status;
    vx_enum tensor_data_type = interpret_tensor_data_type(_info.data_type());
    _vx_handle = vxCreateTensor(context, _info.num_of_dims(), _info.dims().data(), tensor_data_type, 0);
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensor(input: failed " + TOSTR(status))
    _info._type = rocalTensorInfo::Type::REGULAR;
    return 0;
}

#if ENABLE_OPENCL
unsigned rocalTensor::copy_data(cl_command_queue queue, unsigned char *user_buffer, bool sync) {
    if (_info._type != rocalTensorInfo::Type::HANDLE) return 0;

    if (_info._mem_type == RocalMemType::OCL) {
        cl_int status;
        if ((status = clEnqueueReadBuffer(
                queue, (cl_mem)_mem_handle, sync ? (CL_TRUE) : CL_FALSE, 0,
                _info.data_size(), user_buffer, 0, nullptr, nullptr)) != CL_SUCCESS) {
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
        }
    } else {
        memcpy(user_buffer, _mem_handle, _info.data_size());
    }
    return 0;
}
#elif ENABLE_HIP
unsigned rocalTensor::copy_data(hipStream_t stream, void *host_memory, bool sync) {
    if (_info._type != rocalTensorInfo::Type::HANDLE) return 0;

    if (_info._mem_type == RocalMemType::HIP) {
        // copy from device to host
        hipError_t status;
        if ((status = hipMemcpyDtoHAsync((void *)host_memory, _mem_handle, _info.data_size(), stream)))
            THROW("copy_data::hipMemcpyDtoH failed: " + TOSTR(status))
        if (sync) {
            if ((status = hipStreamSynchronize(stream)))
                THROW("copy_data::hipStreamSynchronize failed: " + TOSTR(status))
        }
    } else {
        memcpy(host_memory, _mem_handle, _info.data_size());
    }
    return 0;
}
#endif

unsigned rocalTensor::copy_data(void *user_buffer) {
    if (_info._type != rocalTensorInfo::Type::HANDLE) return 0;

#if ENABLE_HIP
    if (_info._mem_type == RocalMemType::HIP) {
        // copy from device to host
        hipError_t status;
        if ((status = hipMemcpyDtoH((void *)user_buffer, _mem_handle, _info.data_size())))
            THROW("copy_data::hipMemcpyDtoH failed: " + TOSTR(status))
    } else
#endif
    {
        // copy from host to host
        memcpy(user_buffer, _mem_handle, _info.data_size());
    }
    return 0;
}

unsigned rocalTensor::copy_data(void *user_buffer, uint max_y1, uint max_x1) {
    if (_info._type != rocalTensorInfo::Type::HANDLE) return 0;

    size_t datatype_stride = _info.data_type_size();
    auto src_stride = (_info.max_dims().at(0) * _info.max_dims().at(1) * datatype_stride);
    auto dst_stride = (max_y1 * max_x1 * datatype_stride);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(8)
	for(int batchCount = 0; batchCount < _info._batch_size; batchCount++)
	{
        float *srcPtrTemp = static_cast<float *>(_mem_handle) + batchCount * src_stride;
		float *dstPtrTemp = static_cast<float *>(user_buffer) + batchCount * dst_stride;

        uint srcAudioDims[2], srcReductionDims[2], srcStride[2], paramStride[2];
        srcAudioDims[0] = _info.get_roi()[batchCount].y2;
        srcAudioDims[1] = _info.get_roi()[batchCount].x2;

        srcStride[0] = 1;
        srcStride[1] = _info.max_dims().at(0);
        srcReductionDims[0] = srcAudioDims[0];
        srcReductionDims[1] = srcAudioDims[1];
        paramStride[0] = 0;
        paramStride[1] = 1;

        float* meanTensor = (float *)calloc(srcReductionDims[0], sizeof(float));
        float* stdDevTensor = (float *)calloc(srcReductionDims[0], sizeof(float));

        meanTensor[0] = 0.0;
        stdDevTensor[0] = 1.0;
        compute_2D_mean(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
        compute_2D_inv_std_dev(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);
        normalize_2D_tensor_avx_axis2(srcPtrTemp, _info.max_dims().at(0), dstPtrTemp, max_x1, meanTensor, stdDevTensor, 0, srcAudioDims, paramStride);
    }
}

unsigned rocalTensor::copy_data(void *user_buffer, uint last_batch_size) {
    if (_info._type != rocalTensorInfo::Type::HANDLE) return 0;

#if ENABLE_HIP
    if (_info._mem_type == RocalMemType::HIP) {
        // copy from device to host
        hipError_t status;
        if ((status = hipMemcpyDtoH((void *)user_buffer, _mem_handle, _info.data_size()/_info._batch_size*last_batch_size)))
            THROW("copy_data::hipMemcpyDtoH failed: " + TOSTR(status))
    } else
#endif
    {
        // copy from host to host
        memcpy(user_buffer, _mem_handle, _info.data_size()/_info._batch_size*last_batch_size);
    }
    return 0;
}

int rocalTensor::swap_handle(void *handle) {
    vx_status status;
    if ((status = vxSwapTensorHandle(_vx_handle, handle, nullptr)) != VX_SUCCESS) {
        ERR("Swap handles failed for tensor" + TOSTR(status));
        return -1;
    }

    // Updating the buffer pointer as well,
    // user might want to copy directly using it
    _mem_handle = handle;
    return 0;
}