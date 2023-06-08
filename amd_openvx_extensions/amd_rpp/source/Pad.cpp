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
#define NUM_OF_DIMS 5

struct PadLocalData
{
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u deviceType;
    Rpp32u nbatchSize;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    float *anchor;
    float *shape;
    float *fill_values;
    int axes;
    bool normalized_anchor;
    bool normalized_shape;
    uint policy;
    RpptDescPtr src_desc_ptr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    RpptDescPtr dst_desc_ptr;
    void *roi_tensor_ptr_src;
    RpptROI *roi_ptr_src;
    int *srcDims;
    void *roi_tensor_ptr_dst;
    RpptROI *roi_ptr_dst;
    int *dstDims;
    size_t in_tensor_dims[NUM_OF_DIMS];
    size_t out_tensor_dims[NUM_OF_DIMS];
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
    Rpp32u numDims;
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};


void update_destination_roi_pad(const vx_reference *parameters, PadLocalData *data)
{
    int num_of_dims_shapes_anchors;
    data->roi_ptr_dst = (RpptROI *)data->roi_tensor_ptr_dst;
    data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
    num_of_dims_shapes_anchors = data->dst_desc_ptr->numDims;

    if (data->out_tensor_dims[2] == 1)
        num_of_dims_shapes_anchors = 1;

        for(unsigned i = 0; i < data->nbatchSize; i++) {
        int idx = i * num_of_dims_shapes_anchors;
        for(unsigned d = 0; d < num_of_dims_shapes_anchors; d++) {
        // std::cerr << "\n Anchor : " << data->anchor[idx + d] << "|\t Shape Array : " << (data->shape[idx + d] - data->anchor[idx + d]);
            if(num_of_dims_shapes_anchors == 2 ) {
            data->anchor[idx + d] = 0;
            data->shape[idx + d] = (d==0) ? data->in_tensor_dims[2] : data->in_tensor_dims[1];
            }
            else if(num_of_dims_shapes_anchors == 1) {
                data->anchor[idx + d] = 0;
                data->shape[idx + d] = data->in_tensor_dims[2] * data->in_tensor_dims[1];
            }
        }
            data->roi_ptr_dst[i].xywhROI.xy.x = data->roi_ptr_src[i].xywhROI.xy.x;
            data->roi_ptr_dst[i].xywhROI.xy.y = data->roi_ptr_src[i].xywhROI.xy.y;

    }
}

static vx_status VX_CALLBACK refreshPad(vx_node node, const vx_reference *parameters, vx_uint32 num, PadLocalData *data)
{
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &data->anchor, sizeof(vx_float32)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HOST, &data->shape, sizeof(vx_float32)));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->nbatchSize * data->numDims, sizeof(float), data->fill_values, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->hip_pDst, sizeof(data->hip_pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_src, sizeof(data->roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_dst, sizeof(data->roi_tensor_ptr_dst)));
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
    {
        if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_src, sizeof(vx_uint32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_dst, sizeof(vx_uint32)));
        }
    }
    data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
    for (uint i = 0, j = 0; i < data->nbatchSize * 2, j < data->nbatchSize; i = i + 2, j = j + 1) {
      data->srcDims[i] = data->roi_ptr_src[j].xywhROI.xy.y;
      data->srcDims[i + 1] = data->roi_ptr_src[j].xywhROI.xy.x;

    }

    // Get the dimensions of the shapes / anchors tensor
    data->dst_desc_ptr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &data->dst_desc_ptr->numDims, sizeof(data->dst_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, &data->out_tensor_dims, sizeof(vx_size) * data->dst_desc_ptr->numDims));
    if (data->policy == 0)
        update_destination_roi_pad(parameters, data);
    return status;
}

static vx_status VX_CALLBACK validatePad(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
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

static vx_status VX_CALLBACK processPad(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    PadLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_HIP
        refreshPad(node, parameters, num, data);
        // rpp_status = rppt_Slice_gpu((void *)data->hip_pSrc, data->src_desc_ptr, (void *)data->hip_pDst, data->src_desc_ptr,  data->alpha, data->beta, data->hip_roi_tensor_ptr, data->roiType, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
    {
        refreshPad(node, parameters, num, data);
//TODO: Swetha : To clean up the debug code
//  float * buffer = (float *)data->anchor;
//             for(int n = 0; n < data->nbatchSize * 2; n++)
//             {
//                 std::cerr <<"slice begin:  "<<(float)buffer[n] << "\n";
//             }
//  float * buffer1 = (float *)data->shape;
//             for(int n = 0; n < data->nbatchSize * 2; n++)
//             {
//                 std::cerr <<"slice length:  "<<(float)buffer1[n] << "\n";
//             }

// int * dimSrc = (int*) data->srcDims;
//  for(int n = 0; n < data->nbatchSize*2; n++)
//             {
//                 std::cerr <<"src length:  "<<(int)dimSrc[n] << "\n";
//             }
// float * psrc = (float*) data->pSrc;
//  for(int n = 0; n < data->nbatchSize; n++)
//             {
//                 for (int j=0; j<(int)dimSrc[n];j++)

//                     std::cerr <<"src psrc:  "<<(float)psrc[(int)dimSrc[n] * n + j] << "\n";
//             }

        rpp_status = rppt_slice_host((float *)data->pSrc, data->src_desc_ptr, (float *)data->pDst, data->dst_desc_ptr, data->srcDims, (float*)data->anchor, (float*)data->shape, data->fill_values);
// float * pdst = (float*) data->pDst;
//  for(int n = 0; n < data->nbatchSize; n++)
//             {
//                 // for (int j=0; j<(int)dimSrc[n];j++)

//                     std::cerr <<"Slice src pdst:  "<<(float)pdst[(int)dimSrc[n] * n + 0] << "\n";
//             }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializePad(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    PadLocalData *data = new PadLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[12], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[7], &data->axes));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[8], &data->normalized_anchor));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[9], &data->normalized_shape));
    data->normalized_shape = data->normalized_anchor; // shobi
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[10], &data->policy));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[11], &data->nbatchSize));

    // Querying for input tensor
    data->src_desc_ptr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->src_desc_ptr->numDims, sizeof(data->src_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims, sizeof(vx_size) * data->src_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        data->src_desc_ptr->dataType = RpptDataType::F32;
    data->src_desc_ptr->offsetInBytes = 0;

    // Querying for output tensor
    data->dst_desc_ptr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->dst_desc_ptr->numDims, sizeof(data->dst_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->out_tensor_dims, sizeof(vx_size) * data->dst_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1],VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));
    if (data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        data->dst_desc_ptr->dataType = RpptDataType::F32;
    data->dst_desc_ptr->offsetInBytes = 0;

    if (data->out_tensor_dims[2] != 1) {
    // source_description_ptr
    data->src_desc_ptr->n = data->in_tensor_dims[0];
    data->src_desc_ptr->h = data->in_tensor_dims[2];
    data->src_desc_ptr->w = 1; // For pad augmentation only
    data->src_desc_ptr->c = data->in_tensor_dims[1];
    data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
    data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
    data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
    data->src_desc_ptr->strides.cStride = 1;
    data->numDims = data->src_desc_ptr->numDims - 1;
    data->src_desc_ptr->numDims = 4;

    // destination_description_ptr
    data->dst_desc_ptr->n = data->out_tensor_dims[0];
    data->dst_desc_ptr->w = 1; // For pad augmentation only
    data->dst_desc_ptr->h = data->out_tensor_dims[2];
    data->dst_desc_ptr->c = data->out_tensor_dims[1]; //maxDstWidth
    data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w;
    data->dst_desc_ptr->strides.wStride = data->dst_desc_ptr->c;
    data->dst_desc_ptr->strides.cStride = 1;
    data->dst_desc_ptr->numDims = 4;
    }
    else if (data->out_tensor_dims[2] == 1)
    {
        // source_description_ptr
    data->src_desc_ptr->n = data->in_tensor_dims[0];
    data->src_desc_ptr->h = data->in_tensor_dims[2];
    data->src_desc_ptr->w = data->in_tensor_dims[1];
    data->src_desc_ptr->c = 1;
    data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
    data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
    data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
    data->src_desc_ptr->strides.cStride = 1;
    data->numDims = data->src_desc_ptr->numDims - 1;
    data->src_desc_ptr->numDims = 4;

    // source_description_ptr
    data->dst_desc_ptr->n = data->out_tensor_dims[0];
    data->dst_desc_ptr->w = data->out_tensor_dims[1];
    data->dst_desc_ptr->h = data->out_tensor_dims[2];;
    data->dst_desc_ptr->c = 1;
    data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
    data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w;
    data->dst_desc_ptr->strides.wStride = data->dst_desc_ptr->c;
    data->dst_desc_ptr->strides.cStride = 1;
    data->dst_desc_ptr->numDims = 4;
    }

    data->srcDims = (int *) calloc(data->src_desc_ptr->n * 2, sizeof(int));
    data->fill_values = (float *) calloc(data->src_desc_ptr->n * data->numDims, sizeof(float));


// #if ENABLE_HIP
//     if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
//         hipMalloc(&data->hip_roi_tensor_ptr, data->src_desc_ptr->n * sizeof(RpptROI));
// #endif
//     data->roi_tensor_ptr = (RpptROI *)calloc(data->src_desc_ptr->n, sizeof(RpptROI));
//TODO: Swetha : To clean up the debug code
    refreshPad(node, parameters, num, data);
#if ENABLE_OPENCL
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.cmdq, data->nbatchSize);
#elif ENABLE_HIP
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.hipstream, data->nbatchSize);
#endif
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
        rppCreateWithBatchSize(&data->rppHandle, data->nbatchSize);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializePad(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    PadLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL || ENABLE_HIP
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
    free(data->fill_values);
    free(data->srcDims);
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

vx_status Pad_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Pad",
                                       VX_KERNEL_RPP_PAD,
                                       processPad,
                                       13,
                                       validatePad,
                                       initializePad,
                                       uninitializePad);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // New - 3
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 12, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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
