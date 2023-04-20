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

#include "internal_publishKernels.h"
#include <omp.h>

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

#define NUM_OF_DIMS 5
struct TensorMulScalarLocalData
{
    RPPCommonHandle handle;
    Rpp32u device_type;
    Rpp32u nbatchSize;
    void* roi_tensor_ptr_src;
    RpptROI* roi_ptr_src;
    void* roi_tensor_ptr_dst;
    RpptROI* roi_ptr_dst;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    float scalar_value;
    size_t tensor_size;
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
    size_t in_tensor_dims[NUM_OF_DIMS];
    size_t out_tensor_dims[NUM_OF_DIMS];
#if ENABLE_HIP
    void *pSrc_dev;
    void *pDst_dev;
#endif
};

void update_destination_roi(const vx_reference *parameters, TensorMulScalarLocalData *data)
{
    data->roi_ptr_dst = (RpptROI *)data->roi_tensor_ptr_dst;
    data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
    for (uint i=0; i < data->nbatchSize; i++)
    {
        data->roi_ptr_dst[i].xywhROI.xy.x = data->roi_ptr_src[i].xywhROI.xy.x;
        data->roi_ptr_dst[i].xywhROI.xy.y = data->roi_ptr_src[i].xywhROI.xy.y;
    }
}

static vx_status VX_CALLBACK refreshTensorMulScalar(vx_node node, const vx_reference *parameters, vx_uint32 num, TensorMulScalarLocalData *data)
{
    vx_status status = VX_SUCCESS;
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc_dev, sizeof(data->pSrc_dev)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->pDst_dev, sizeof(data->pDst_dev)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_src, sizeof(data->roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_dst, sizeof(data->roi_tensor_ptr_dst)));
#endif
    }
    else if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_UINT8)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        }
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_src, sizeof(vx_uint32)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_dst, sizeof(vx_uint32)));

        update_destination_roi(parameters, data);
    }
    return status;
}

static vx_status VX_CALLBACK validateTensorMulScalar(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #2 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #3 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[NUM_OF_DIMS];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 1);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processTensorMulScalar(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    TensorMulScalarLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_HIP
        refreshTensorMulScalar(node, parameters, num, data);
        // make the rpp call or add the implementation here
        // hipMemcpy(data->pDst_dev, data->pSrc_dev, data->tensor_size, hipMemcpyDeviceToDevice);
        std::cerr << "process :  TensorMulScalar , HIP backend not supported as of date";
#endif
    }
    else if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshTensorMulScalar(node, parameters, num, data);
        // memcpy(data->pDst, data->pSrc, data->tensor_size);
        // Add the case for UNIT8 datatype
        if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            uint channels = 1; // for audio data
            data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
            __m256 pMul = _mm256_set1_ps(data->scalar_value);
            float scalarValue = data->scalar_value;
            size_t nStride = data->in_tensor_dims[1] * data->in_tensor_dims[2] * channels;

        #pragma omp parallel for num_threads(8)
            for (uint i = 0; i < data->nbatchSize; i++)
            {
                float *srcTemp = (float *)(data->pSrc) + i * nStride;
                float *dstTemp = (float *)(data->pDst) + i * nStride;
                uint height = data->roi_ptr_src[i].xywhROI.xy.y;
                uint width = data->roi_ptr_src[i].xywhROI.xy.x * channels;
                uint alignedWidth = (width / 8) * 8;
                for (uint row = 0; row < height; row++)
                {
                    float *srcPtrRow = srcTemp + row * data->in_tensor_dims[1];
                    float *dstPtrRow = dstTemp + row * data->out_tensor_dims[1];
                    uint vectorLoopCount = 0;
                    for(; vectorLoopCount < alignedWidth; vectorLoopCount += 8)
                    {
                        __m256 pSrc = _mm256_loadu_ps(srcPtrRow);
                        __m256 pDst = _mm256_mul_ps(pSrc, pMul);
                        _mm256_storeu_ps(dstPtrRow, pDst);
                        srcPtrRow += 8;
                        dstPtrRow += 8;
                    }
                    for(; vectorLoopCount < width; vectorLoopCount++)
                        *dstPtrRow++ = *srcPtrRow++ * scalarValue;
                }
            }

            // for (uint i = 0; i < (data->tensor_size)/ sizeof(float); i++)
            // {
            //     // std::cerr << "\n i :: " << i;
            //     // std::cerr << "\n scalar :: "<< data->scalar_value;
            //     // std::cerr << "\n (float *)(data->pSrc))[i]"<< ((float *)(data->pSrc))[i];
            //     ((float *)(data->pDst))[i] = ((float *)(data->pSrc))[i] * data->scalar_value;
            //     // std::cerr << "\n ((float *)(data->pDst))[i]" << ((float *)(data->pDst))[i];
            
            // }
        }
    }
    return status;
}

static vx_status VX_CALLBACK initializeTensorMulScalar(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorMulScalarLocalData *data = new TensorMulScalarLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    THROW("initialize : TensorMulScalar, OpenCL backend is not supported")
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[2], &data->scalar_value));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[5], &data->nbatchSize));
    vx_size num_of_dims;
    // size_t in_tensor_dims[NUM_OF_DIMS], out_tensor_dims[NUM_OF_DIMS];
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims, sizeof(vx_size) * num_of_dims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->out_tensor_dims, sizeof(vx_size) * num_of_dims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));

    data->tensor_size = 1;
    for(int i = 0; i < num_of_dims; i++)
        data->tensor_size *= data->in_tensor_dims[i];

    if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        data->tensor_size *= sizeof(float);

    refreshTensorMulScalar(node, parameters, num, data);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeTensorMulScalar(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorMulScalarLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
)
{
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status TensorMulScalar_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.TensorMulScalar",
                                       VX_KERNEL_RPP_TENSORMULSCALAR,
                                       processTensorMulScalar,
                                       7,
                                       validateTensorMulScalar,
                                       initializeTensorMulScalar,
                                       uninitializeTensorMulScalar);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    if (kernel)
    {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); 
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // New Arg
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // Old
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS)
    {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}