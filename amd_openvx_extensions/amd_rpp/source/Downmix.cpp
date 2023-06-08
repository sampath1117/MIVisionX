/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include "vx_ext_amd.h"

struct DownmixLocalData
{
    RPPCommonHandle *handle;
    Rpp32u deviceType;
    Rpp32u nbatchSize;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_int32 *samples;
    vx_int32 *channels;
    RpptDescPtr srcDescPtr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    RpptDescPtr dstDescPtr;
    size_t inTensorDims[RPP_MAX_TENSOR_AUDIO_DIMS];
    size_t outTensorDims[RPP_MAX_TENSOR_AUDIO_DIMS];
    vx_enum inTensorType;
    vx_enum outTensorType;
};

static vx_status VX_CALLBACK refreshDownmix(vx_node node, const vx_reference *parameters, vx_uint32 num, DownmixLocalData *data)
{
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, data->nbatchSize, sizeof(data->samples), data->samples, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[3], 0, data->nbatchSize, sizeof(data->channels), data->channels, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
    {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
    }
    return status;
}

static vx_status VX_CALLBACK validateDownmix(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #4 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_AUDIO_DIMS];
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

static vx_status VX_CALLBACK processDownmix(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    DownmixLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_HIP
        refreshDownmix(node, parameters, num, data);
        // rpp_status = rppt_Downmix_gpu((void *)data->hip_pSrc, data->srcDescPtr, (void *)data->hip_pDst, data->srcDescPtr,  data->alpha, data->beta, data->hip_roi_tensor_Ptr, data->roiType, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
    {
        refreshDownmix(node, parameters, num, data);
        rpp_status = rppt_down_mixing_host((float *)data->pSrc, data->srcDescPtr, (float *)data->pDst, data->dstDescPtr, data->samples, data->channels, false);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeDownmix(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    DownmixLocalData *data = new DownmixLocalData;
    memset(data, 0, sizeof(*data));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[4], &data->nbatchSize));

    // Querying for input tensor
    data->srcDescPtr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->srcDescPtr->numDims, sizeof(data->srcDescPtr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inTensorDims, sizeof(vx_size) * data->srcDescPtr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->inTensorType, sizeof(data->inTensorType)));
    if (data->inTensorType == vx_type_e::VX_TYPE_FLOAT32)
        data->srcDescPtr->dataType = RpptDataType::F32;
    data->srcDescPtr->offsetInBytes = 0;

    // Querying for output tensor
    data->dstDescPtr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->dstDescPtr->numDims, sizeof(data->dstDescPtr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->outTensorDims, sizeof(vx_size) * data->dstDescPtr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1],VX_TENSOR_DATA_TYPE, &data->outTensorType, sizeof(data->outTensorType)));
    if (data->outTensorType == vx_type_e::VX_TYPE_FLOAT32)
        data->dstDescPtr->dataType = RpptDataType::F32;
    data->dstDescPtr->offsetInBytes = 0;

    // source_description_ptr
    data->srcDescPtr->n = data->inTensorDims[0];
    data->srcDescPtr->h = data->inTensorDims[2];
    data->srcDescPtr->w = data->inTensorDims[1];
    data->srcDescPtr->c = 1;
    data->srcDescPtr->strides.nStride = data->srcDescPtr->c * data->srcDescPtr->w * data->srcDescPtr->h;
    data->srcDescPtr->strides.hStride = data->srcDescPtr->c * data->srcDescPtr->w;
    data->srcDescPtr->strides.wStride = data->srcDescPtr->c;
    data->srcDescPtr->strides.cStride = 1;
    data->srcDescPtr->numDims = 4;

    // source_description_ptr
    data->dstDescPtr->n = data->inTensorDims[0];
    data->dstDescPtr->w = data->inTensorDims[1];
    data->dstDescPtr->h = 1;
    data->dstDescPtr->c = 1;
    data->dstDescPtr->strides.nStride = data->dstDescPtr->c * data->dstDescPtr->w * data->dstDescPtr->h;
    data->dstDescPtr->strides.hStride = data->dstDescPtr->c * data->dstDescPtr->w;
    data->dstDescPtr->strides.wStride = data->dstDescPtr->c;
    data->dstDescPtr->strides.cStride = 1;
    data->dstDescPtr->numDims = 4;
    data->samples = (vx_int32 *)malloc(sizeof(vx_int32) * data->srcDescPtr->n);
    data->channels = (vx_int32 *)malloc(sizeof(vx_int32) * data->srcDescPtr->n);
    refreshDownmix(node, parameters, num, data);
    STATUS_ERROR_CHECK(createGraphHandle(node, &data->handle, data->srcDescPtr->n, data->deviceType));

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeDownmix(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    DownmixLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseGraphHandle(node, data->handle, data->deviceType));
    free(data->samples);
    free(data->channels);
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

// hardcode the affinity to  CPU for OpenCL backend to avoid VerifyGraph failure since there is no codegen callback for amd_rpp nodes
#if ENABLE_OPENCL
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
#endif

    return VX_SUCCESS;
}

vx_status Downmix_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Downmix",
                                       VX_KERNEL_RPP_DOWNMIX,
                                       processDownmix,
                                       6,
                                       validateDownmix,
                                       initializeDownmix,
                                       uninitializeDownmix);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_OPENCL || ENABLE_HIP
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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