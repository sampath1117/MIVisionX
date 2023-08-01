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

struct SliceLocalData {
    vxRppHandle *handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    Rpp32f *pAnchor;
    Rpp32f *pShape;
    Rpp32f *pFillValues;
    int axes;
    bool normalizedAnchor;
    bool normalizedShape;
    uint policy;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    void *roi_tensor_ptr_src;
    RpptROI *roi_ptr_src;
    int *srcDims;
    void *roi_tensor_ptr_dst;
    RpptROI *roi_ptr_dst;
    int *dstDims;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
    Rpp32u numDims;
};

void copy_src_dims_and_update_dst_roi(const vx_reference *parameters, SliceLocalData *data) {
    int RPP_MAX_TENSOR_DIMS_shapes_anchors;
    int dimsTotal = 1;
    data->roi_ptr_dst = (RpptROI *)data->roi_tensor_ptr_dst;
    data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
    for(unsigned numDims=0; numDims < data->pDstDesc->numDims ; numDims++)
        dimsTotal*=data->outputTensorDims[numDims];
    if (dimsTotal == 1000)
        RPP_MAX_TENSOR_DIMS_shapes_anchors = 1;
    else if ((dimsTotal == (1000 * 2)))
        RPP_MAX_TENSOR_DIMS_shapes_anchors = 2;
    else
        RPP_MAX_TENSOR_DIMS_shapes_anchors = 3;
        for(unsigned i = 0; i < 1000; i++) {
        int idx = i * RPP_MAX_TENSOR_DIMS_shapes_anchors;
        for(unsigned d = 0; d < RPP_MAX_TENSOR_DIMS_shapes_anchors; d++) {
            if(RPP_MAX_TENSOR_DIMS_shapes_anchors == 2  ) {
                if (d==0) data->roi_ptr_dst[i].xywhROI.xy.x = (data->pShape[idx + d]);
                if (d==1) data->roi_ptr_dst[i].xywhROI.xy.y = (data->pShape[idx + d]);
            }
            else if (RPP_MAX_TENSOR_DIMS_shapes_anchors == 1) {
                data->roi_ptr_dst[i].xywhROI.xy.x = (data->pShape[i]);
                data->roi_ptr_dst[i].xywhROI.xy.y = data->roi_ptr_src[i].xywhROI.xy.y;
            }
        }

    }
}

static vx_status VX_CALLBACK refreshSlice(vx_node node, const vx_reference *parameters, vx_uint32 num, SliceLocalData *data) {
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &data->pAnchor, sizeof(data->pAnchor)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HOST, &data->pShape, sizeof(data->pShape)));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->numDims, sizeof(float), data->pFillValues, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(&data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_src, sizeof(data->roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_dst, sizeof(data->roi_tensor_ptr_dst)));
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
    {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_src, sizeof(data->roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_dst, sizeof(data->roi_tensor_ptr_dst)));
    }
    data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
    for (uint i = 0, j = 0; i < 1000 * 2, j < 1000; i = i + 2, j = j + 1) {
      data->srcDims[i] = data->roi_ptr_src[j].xywhROI.xy.x;
      data->srcDims[i + 1] = data->roi_ptr_src[j].xywhROI.xy.y;
    }

    // Get the dimensions of the shapes / anchors tensor
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    copy_src_dims_and_update_dst_roi(parameters, data);
    return status;
}

static vx_status VX_CALLBACK validateSlice(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_BOOL)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_BOOL)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #8 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[10], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #9 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
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

static vx_status VX_CALLBACK processSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    SliceLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshSlice(node, parameters, num, data);
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        refreshSlice(node, parameters, num, data);
        // rpp_status = rppt_slice_gpu(data->pSrc, data->pSrcDesc, data->pDst, data->pDstDesc, data->srcDims, data->pAnchor, data->pShape, data->pFillValues, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        rpp_status = rppt_slice_host(data->pSrc, data->pSrcDesc, data->pDst, data->pDstDesc, data->srcDims, data->pAnchor, data->pShape, data->pFillValues, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SliceLocalData *data = new SliceLocalData;
    memset(data, 0, sizeof(*data));

    vx_enum input_tensor_datatype, output_tensor_datatype;
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[7], &data->axes));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[8], &data->normalizedAnchor));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[9], &data->normalizedShape));
    data->normalizedShape = data->normalizedAnchor;
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[10], &data->policy));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[11], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    // Querying for input tensor
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_datatype, sizeof(input_tensor_datatype)));
    data->pSrcDesc->dataType = getRpptDataType(input_tensor_datatype);
    data->pSrcDesc->offsetInBytes = 0;

    // Querying for output tensor
    data->pDstDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1],VX_TENSOR_DATA_TYPE, &output_tensor_datatype, sizeof(output_tensor_datatype)));
    data->pDstDesc->dataType = getRpptDataType(output_tensor_datatype);
    data->pDstDesc->offsetInBytes = 0;

    // source_description_ptr
    data->pSrcDesc->n = data->inputTensorDims[0];
    data->pSrcDesc->h = data->inputTensorDims[2];
    data->pSrcDesc->w = data->inputTensorDims[1];
    data->pSrcDesc->c = 1;
    data->pSrcDesc->strides.nStride = data->pSrcDesc->c * data->pSrcDesc->w * data->pSrcDesc->h;
    data->pSrcDesc->strides.hStride = data->pSrcDesc->c * data->pSrcDesc->w;
    data->pSrcDesc->strides.wStride = data->pSrcDesc->c;
    data->pSrcDesc->strides.cStride = 1;
    data->numDims = data->pSrcDesc->numDims - 1;
    data->pSrcDesc->numDims = 4;

    // source_description_ptr
    data->pDstDesc->n = data->outputTensorDims[0];
    data->pDstDesc->w = data->outputTensorDims[1];
    data->pDstDesc->h = data->outputTensorDims[2];;
    data->pDstDesc->c = 1;
    data->pDstDesc->strides.nStride = data->pDstDesc->c * data->pDstDesc->w * data->pDstDesc->h;
    data->pDstDesc->strides.hStride = data->pDstDesc->c * data->pDstDesc->w;
    data->pDstDesc->strides.wStride = data->pDstDesc->c;
    data->pDstDesc->strides.cStride = 1;
    data->pDstDesc->numDims = 4;

    data->srcDims = static_cast<int *>(calloc(data->pSrcDesc->n * 2, sizeof(int)));
    data->pFillValues = static_cast<float *>(calloc(data->pSrcDesc->n * data->numDims, sizeof(float)));

    refreshSlice(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SliceLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    free(data->pFillValues);
    free(data->srcDims);
    free(data->pSrcDesc);
    free(data->pDstDesc);
    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status Slice_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Slice",
                                       VX_KERNEL_RPP_SLICE,
                                       processSlice,
                                       12,
                                       validateSlice,
                                       initializeSlice,
                                       uninitializeSlice);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel) {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}