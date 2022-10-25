/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#define NUM_OF_DIMS 5

struct ResizeLocalData
{
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u device_type;
    RpptDescPtr src_desc_ptr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    Rpp32u nbatchSize;
    RpptDescPtr dst_desc_ptr;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_uint32 *resize_h;
    vx_uint32 *resize_w;
    size_t in_tensor_dims[NUM_OF_DIMS]; // will have NHWC info
    size_t out_tensor_dims[NUM_OF_DIMS];
    vx_uint32 batch_size;
    RpptROI *roi_tensor_Ptr;
    RpptRoiType roiType;
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
    RpptImagePatch *dstImgsize;
    Rpp32u layout;

#if ENABLE_HIP
    void *pSrc_dev;
    void *pDst_dev;
    RpptImagePatch *hip_dstImgSize;
    RpptROI *hip_roiTensorPtrSrc;
#endif
};

static vx_status VX_CALLBACK refreshResize(vx_node node, const vx_reference *parameters, vx_uint32 num, ResizeLocalData *data) {
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, data->nbatchSize * 4, sizeof(unsigned), data->roi_tensor_Ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->nbatchSize, sizeof(vx_uint32), data->resize_w, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->nbatchSize, sizeof(vx_uint32), data->resize_h, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    for (int i = 0; i < data->nbatchSize; i++) {
        data->dstImgsize[i].width = data->resize_w[i];
        data->dstImgsize[i].height = data->resize_h[i];
    } if(data->layout == 2 || data->layout == 3) {
        unsigned num_of_frames = data->in_tensor_dims[1]; // Num of frames 'F'
        for(int n = data->nbatchSize - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for(int f = 0; f < num_of_frames; f++) {
                data->resize_w[index + f] = data->resize_w[n];
                data->resize_h[index + f] = data->resize_h[n];
                data->roi_tensor_Ptr[index + f].xywhROI.xy.x = data->roi_tensor_Ptr[n].xywhROI.xy.x;
                data->roi_tensor_Ptr[index + f].xywhROI.xy.y = data->roi_tensor_Ptr[n].xywhROI.xy.y;
                data->roi_tensor_Ptr[index + f].xywhROI.roiWidth = data->roi_tensor_Ptr[n].xywhROI.roiWidth;
                data->roi_tensor_Ptr[index + f].xywhROI.roiHeight = data->roi_tensor_Ptr[n].xywhROI.roiHeight;
            }
        }
    }
#if ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc_dev, sizeof(data->pSrc_dev)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst_dev, sizeof(data->pDst_dev)));
        hipMemcpy(data->hip_dstImgSize, data->dstImgsize, data->nbatchSize * sizeof(RpptImagePatch), hipMemcpyHostToDevice);
        hipMemcpy(data->hip_roiTensorPtrSrc, data->roi_tensor_Ptr, data->nbatchSize * sizeof(RpptROI), hipMemcpyHostToDevice);
    } else if (data->device_type == AGO_TARGET_AFFINITY_CPU)
#endif
    {
        if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_UINT8) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        } else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        } else if (data->in_tensor_type == vx_type_e::VX_TYPE_INT8 && data->out_tensor_type == vx_type_e::VX_TYPE_INT8) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_int8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_int8)));
        } else if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        } 
#if defined(AMD_FP16_SUPPORT)
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT16 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT16) {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float16)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float16)));
        }
#endif
    }
    return status;
}

static vx_status VX_CALLBACK validateResize(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be a boolean size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #10 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #11 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[10], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #12 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[NUM_OF_DIMS];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 2);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processResize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    vx_status vxstatus;
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    ResizeLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    Rpp32u N;
    N = data->nbatchSize;
#if ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU) {
        refreshResize(node, parameters, num, data);
        rpp_status = rppt_resize_gpu((void *)data->pSrc_dev, data->src_desc_ptr, (void *)data->pDst_dev, data->dst_desc_ptr, data->hip_dstImgSize, RpptInterpolationType::BILINEAR, data->hip_roiTensorPtrSrc, data->roiType, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;

    } else if (data->device_type == AGO_TARGET_AFFINITY_CPU)
#endif
    {
        refreshResize(node, parameters, num, data);
        rpp_status = rppt_resize_host(data->pSrc, data->src_desc_ptr, data->pDst, data->dst_desc_ptr, data->dstImgsize, RpptInterpolationType::BILINEAR, data->roi_tensor_Ptr, data->roiType, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeResize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ResizeLocalData *data = new ResizeLocalData;
    unsigned roiType;
    memset(data, 0, sizeof(*data));

#if ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[10], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[9], &data->nbatchSize));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &roiType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->roiType = (roiType == 0) ? RpptRoiType::XYWH : RpptRoiType::LTRB;

    // Querying for input tensor
    data->src_desc_ptr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->src_desc_ptr->numDims, sizeof(data->src_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims, sizeof(vx_size) * data->src_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8) {
        data->src_desc_ptr->dataType = RpptDataType::U8;
    } else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
        data->src_desc_ptr->dataType = RpptDataType::F32;
    } else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT16) {
        data->src_desc_ptr->dataType = RpptDataType::F16;
    } else if (data->in_tensor_type == vx_type_e::VX_TYPE_INT8) {
        data->src_desc_ptr->dataType = RpptDataType::I8;
    }
    data->src_desc_ptr->offsetInBytes = 0;

    // Querying for output tensor
    data->dst_desc_ptr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->dst_desc_ptr->numDims, sizeof(data->dst_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->out_tensor_dims, sizeof(vx_size) * data->dst_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));
    if (data->out_tensor_type == vx_type_e::VX_TYPE_UINT8) {
        data->dst_desc_ptr->dataType = RpptDataType::U8;
    } else if (data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32) {
        data->dst_desc_ptr->dataType = RpptDataType::F32;
    } else if (data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT16) {
        data->src_desc_ptr->dataType = RpptDataType::F16;
    } else if (data->out_tensor_type == vx_type_e::VX_TYPE_INT8) {
        data->dst_desc_ptr->dataType = RpptDataType::I8;
    }
    data->dst_desc_ptr->offsetInBytes = 0;
    // declaring and pushing values to roi_tensor_Ptr
    if (data->layout == 0) {    // NHWC
        // source_description_ptr
        data->src_desc_ptr->n = data->in_tensor_dims[0];
        data->src_desc_ptr->h = data->in_tensor_dims[1];
        data->src_desc_ptr->w = data->in_tensor_dims[2];
        data->src_desc_ptr->c = data->in_tensor_dims[3];
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
        data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
        data->src_desc_ptr->strides.cStride = 1;
        data->src_desc_ptr->layout = RpptLayout::NHWC;

        // destination_description_ptr
        data->dst_desc_ptr->n = data->out_tensor_dims[0];
        data->dst_desc_ptr->h = data->out_tensor_dims[1];
        data->dst_desc_ptr->w = data->out_tensor_dims[2];
        data->dst_desc_ptr->c = data->out_tensor_dims[3];
        data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w;
        data->dst_desc_ptr->strides.wStride = data->dst_desc_ptr->c;
        data->dst_desc_ptr->strides.cStride = 1;
        data->dst_desc_ptr->layout = RpptLayout::NHWC;
    } else if(data->layout == 1) {  // NCHW
        data->src_desc_ptr->n = data->in_tensor_dims[0];
        data->src_desc_ptr->h = data->in_tensor_dims[2];
        data->src_desc_ptr->w = data->in_tensor_dims[3];
        data->src_desc_ptr->c = data->in_tensor_dims[1];
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.cStride = data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.hStride = data->src_desc_ptr->w;
        data->src_desc_ptr->strides.wStride = 1;
        data->src_desc_ptr->layout = RpptLayout::NCHW;

        data->dst_desc_ptr->n = data->out_tensor_dims[0];
        data->dst_desc_ptr->h = data->out_tensor_dims[2];
        data->dst_desc_ptr->w = data->out_tensor_dims[3];
        data->dst_desc_ptr->c = data->out_tensor_dims[1];
        data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.cStride = data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->w;
        data->dst_desc_ptr->strides.wStride = 1;
        data->dst_desc_ptr->layout = RpptLayout::NCHW;
    } else if(data->layout == 2) {  // NFHWC
        data->src_desc_ptr->n = data->in_tensor_dims[0] * data->in_tensor_dims[1];
        data->src_desc_ptr->h = data->in_tensor_dims[2];
        data->src_desc_ptr->w = data->in_tensor_dims[3];
        data->src_desc_ptr->c = data->in_tensor_dims[4];
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
        data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
        data->src_desc_ptr->strides.cStride = 1;
        data->src_desc_ptr->layout = RpptLayout::NHWC;

        data->dst_desc_ptr->n = data->out_tensor_dims[0] * data->out_tensor_dims[1];
        data->dst_desc_ptr->h = data->out_tensor_dims[2];
        data->dst_desc_ptr->w = data->out_tensor_dims[3];
        data->dst_desc_ptr->c = data->out_tensor_dims[4];
        data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.cStride = data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->w;
        data->dst_desc_ptr->strides.wStride = 1;
        data->dst_desc_ptr->layout = RpptLayout::NHWC;
    } else if(data->layout == 3) {  // NFCHW
        data->src_desc_ptr->n = data->in_tensor_dims[0] * data->in_tensor_dims[1];
        data->src_desc_ptr->h = data->in_tensor_dims[3];
        data->src_desc_ptr->w = data->in_tensor_dims[4];
        data->src_desc_ptr->c = data->in_tensor_dims[2];
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.cStride = data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.hStride = data->src_desc_ptr->w;
        data->src_desc_ptr->strides.wStride = 1;
        data->src_desc_ptr->layout = RpptLayout::NCHW;

        data->dst_desc_ptr->n = data->out_tensor_dims[0] * data->out_tensor_dims[1];
        data->dst_desc_ptr->h = data->out_tensor_dims[3];
        data->dst_desc_ptr->w = data->out_tensor_dims[4];
        data->dst_desc_ptr->c = data->out_tensor_dims[2];
        data->dst_desc_ptr->strides.nStride = data->dst_desc_ptr->c * data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.cStride = data->dst_desc_ptr->w * data->dst_desc_ptr->h;
        data->dst_desc_ptr->strides.hStride = data->dst_desc_ptr->w;
        data->dst_desc_ptr->strides.wStride = 1;
        data->dst_desc_ptr->layout = RpptLayout::NCHW;
    }
    data->roi_tensor_Ptr = (RpptROI *)calloc(data->nbatchSize, sizeof(RpptROI));
    data->dstImgsize = (RpptImagePatch *)calloc(data->nbatchSize, sizeof(RpptImagePatch));
    data->resize_w = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->src_desc_ptr->n);
    data->resize_h = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->src_desc_ptr->n);
#if ENABLE_HIP
    hipMalloc(&data->hip_dstImgSize, data->nbatchSize * sizeof(RpptImagePatch));
    hipMalloc(&data->hip_roiTensorPtrSrc, data->nbatchSize * sizeof(RpptROI));
#endif
    refreshResize(node, parameters, num, data);
#if ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.hipstream, data->nbatchSize);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppCreateWithBatchSize(&data->rppHandle, data->nbatchSize);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeResize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ResizeLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
#if ENABLE_HIP
    hipFree(data->hip_dstImgSize);
    hipFree(data->hip_roiTensorPtrSrc);
#endif
    free(data->resize_w);
    free(data->resize_h);
    free(data->roi_tensor_Ptr);
    free(data->dstImgsize);
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

vx_status Resize_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Resize",
                                       VX_KERNEL_RPP_RESIZE,
                                       processResize,
                                       11,
                                       validateResize,
                                       initializeResize,
                                       uninitializeResize);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}
