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

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif
#define NUM_OF_DIMS 5

struct TensorAddTensorLocalData
{
    RPPCommonHandle handle;
    Rpp32u device_type;
    Rpp32u nbatchSize;
    RppPtr_t pSrc1;
    RppPtr_t pSrc2;
    RppPtr_t pDst;
    int *srcDims1;
    int *srcDims2;
    void* roi_tensor_ptr_src;
    RpptROI* roi_ptr_src;
    void* roi_tensor_ptr_dst;
    RpptROI* roi_ptr_dst;
    RpptROI *roi_tensor_ptr1;
    RpptROI *roi_tensor_ptr2;
    size_t in_tensor_dims1[NUM_OF_DIMS];
    size_t in_tensor_dims2[NUM_OF_DIMS];
    size_t out_tensor_dims[NUM_OF_DIMS];
    size_t tensor_size;
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
#if ENABLE_HIP
    void *pSrc_dev1;
    void *pSrc_dev2;
    void *pDst_dev;
#endif
};

void update_destination_roi(const vx_reference *parameters, TensorAddTensorLocalData *data)
{
    data->roi_ptr_dst = (RpptROI *)data->roi_tensor_ptr_dst;
    data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
    for (uint i=0; i < data->nbatchSize; i++)
    {
        data->roi_ptr_dst[i].xywhROI.xy.x = data->roi_ptr_src[i].xywhROI.xy.x;
        data->roi_ptr_dst[i].xywhROI.xy.y = data->roi_ptr_src[i].xywhROI.xy.y;
    }
}

static vx_status VX_CALLBACK refreshTensorAddTensor(vx_node node, const vx_reference *parameters, vx_uint32 num, TensorAddTensorLocalData *data)
{

    vx_status status = VX_SUCCESS;

    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc_dev1, sizeof(data->pSrc_dev1)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->pSrc_dev2, sizeof(data->pSrc_dev2)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst_dev, sizeof(data->pDst_dev)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_src, sizeof(data->roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr_dst, sizeof(data->roi_tensor_ptr_dst)));

#endif
    }
    else if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        // std::cerr << "CPU";
        if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_UINT8)
        {
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc1, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pSrc2, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        }
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            // std::cerr << "FLOAT data";
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc1, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pSrc2, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_src, sizeof(vx_uint32)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr_dst, sizeof(vx_uint32)));

        update_destination_roi(parameters, data);
    }
    return status;
}

static vx_status VX_CALLBACK validateTensorAddTensor(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #4 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);

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

static vx_status VX_CALLBACK processTensorAddTensor(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    TensorAddTensorLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_HIP
        refreshTensorAddTensor(node, parameters, num, data);
        // make the rpp call or add the implementation here
        // hipMemcpy(data->pDst_dev, data->pSrc_dev, data->tensor_size, hipMemcpyDeviceToDevice);
        // std::cerr << "process :  TensorAddTensor , HIP backend not supported as of date";
#endif
    }
    else if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshTensorAddTensor(node, parameters, num, data);
        // memcpy(data->pDst, data->pSrc, data->tensor_size);
        // Add the case for UNIT8 datatype // TODO: Swetha
        if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            uint channels = 1; // for audio data
            data->roi_ptr_src = (RpptROI *)data->roi_tensor_ptr_src;
            size_t nStride = data->in_tensor_dims1[1] * data->in_tensor_dims1[2] * channels;

            for (uint i = 0; i < data->nbatchSize; i++)
            {
                float *src1Temp = (float *)(data->pSrc1) + i * nStride;
                float *src2Temp = (float *)(data->pSrc2) + i;
                float *dstTemp = (float *)(data->pDst) + i * nStride;
                uint height = data->roi_ptr_src[i].xywhROI.xy.y;
                uint width = data->roi_ptr_src[i].xywhROI.xy.x;
                uint alignedWidth = (width / 8) * 8;
                float additionFactor = src2Temp[0];
                __m256 pSrc2 = _mm256_set1_ps(additionFactor);
                for (uint row = 0; row < height; row++)
                {
                    float *srcPtr1Row = src1Temp + row * data->in_tensor_dims1[1];
                    float *dstPtrRow = dstTemp + row * data->in_tensor_dims1[1];
                    uint vectorLoopCount = 0;
                    for(; vectorLoopCount < alignedWidth; vectorLoopCount += 8)
                    {
                        __m256 pSrc1 = _mm256_loadu_ps(srcPtr1Row);
                        __m256 pDst = _mm256_add_ps(pSrc1, pSrc2);
                        _mm256_storeu_ps(dstPtrRow, pDst);
                        srcPtr1Row += 8;
                        dstPtrRow += 8;
                    }
                    for(; vectorLoopCount < width; vectorLoopCount++)
                        *dstPtrRow++ = (*srcPtr1Row++) + additionFactor;
                }
            }

            // for (uint i = 0; i < data->nbatchSize; i++)
            // {
            //     //start ptr of the tensor1
            //     // std::cerr << "TAT Batch :: " << i;
            //     size_t size_psrc1_elements = data->in_tensor_dims1[1] * data->in_tensor_dims1[2] * channels;
            //     size_t size_psrc1_roi = data->roi_ptr_src[i].xywhROI.xy.x * data->roi_ptr_src[i].xywhROI.xy.y * channels;
            //     for (uint j = 0; j < size_psrc1_roi ; j++) {
            //         ((float *)(data->pDst))[i * size_psrc1_elements + j] = ((float *)(data->pSrc1))[i * size_psrc1_elements + j] + ((float *)(data->pSrc2))[i] ;
            //         if (j >= 0 && j <  10) {
            //             // std::cerr << "\n i * size_psrc1_elements + j : " << i * size_psrc1_elements + j;
            //             // std::cerr << "\n size_psrc1_elements" << size_psrc1_elements;
            //         //     std::cerr << "\n TAT ((float *)(data->pSrc1))[i * size_psrc1_elements + j] " << ((float *)(data->pSrc1))[i * size_psrc1_elements + j] ;
            //         //     std::cerr << "\n TAT ((float *)(data->pSrc2))[i]  " << ((float *)(data->pSrc2))[i];
            //         //     std::cerr << "\n TAT ((float *)(data->pDst))[i * size_psrc1_elements + j] " << ((float *)(data->pDst))[i * size_psrc1_elements + j] ;
            //         }
            //     }
            // }
        }
    }
    return status;
}

static vx_status VX_CALLBACK initializeTensorAddTensor(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorAddTensorLocalData *data = new TensorAddTensorLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    THROW("initialize : TensorAddTensor, OpenCL backend is not supported")
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[5], &data->nbatchSize));

    vx_size num_of_dims1, num_of_dims2;
    size_t tensor_dims1[NUM_OF_DIMS], tensor_dims2[NUM_OF_DIMS];
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims1, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims1, sizeof(vx_size) * num_of_dims1));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims2, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->in_tensor_dims2, sizeof(vx_size) * num_of_dims2));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));


    refreshTensorAddTensor(node, parameters, num, data);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeTensorAddTensor(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorAddTensorLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    // free(data->roi_tensor_ptr1);
    // free(data->roi_tensor_ptr2);
    // free(data->srcDims1);
    // free(data->srcDims2);
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

vx_status TensorAddTensor_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.TensorAddTensor",
                                       VX_KERNEL_RPP_TENSORADDTENSOR,
                                       processTensorAddTensor,
                                       7,
                                       validateTensorAddTensor,
                                       initializeTensorAddTensor,
                                       uninitializeTensorAddTensor);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // New Arg
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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