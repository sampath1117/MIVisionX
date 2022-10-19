#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_exposure.h"
#include "exception.h"

ExposureNode::ExposureNode(const std::vector<rocalTensor *> &inputs,const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _shift(SHIFT_RANGE[0], SHIFT_RANGE[1])
{
}

void ExposureNode::create_node()
{
    if(_node)
        return;

    _shift.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    if(_inputs[0]->info().layout() == RocalTensorlayout::NCHW)
        _layout = 1;
    else if(_inputs[0]->info().layout() == RocalTensorlayout::NFHWC)
        _layout = 2;
    else if(_inputs[0]->info().layout() == RocalTensorlayout::NFCHW)
        _layout = 3;
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    _node = vxExtrppNode_Exposure(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _shift.default_array(), layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Exposure_batch (vxExtrppNode_Exposure) node failed: "+ TOSTR(status))
}

void ExposureNode::init( float shift)
{
    _shift.set_param(shift);
    _layout = _roi_type = 0;
}

void ExposureNode::init( FloatParam* shift)
{
    _shift.set_param(core(shift));
    _layout = _roi_type = 0;
}


void ExposureNode::update_node()
{
    _shift.update_array();
}