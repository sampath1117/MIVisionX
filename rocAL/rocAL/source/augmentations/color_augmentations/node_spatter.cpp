#include <vx_ext_rpp.h>
#include "node_spatter.h"
#include "exception.h"

SpatterNode::SpatterNode(const std::vector<rocalTensor *> &inputs,const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void SpatterNode::create_node()
{
    if(_node)
        return;

    vx_scalar red_val = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_red);
    vx_scalar green_val = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_green);
    vx_scalar blue_val = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_blue);
    
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

    _node = vxExtrppNode_Spatter(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), red_val, green_val, blue_val, layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the spatter_batch (vxExtrppNode_spatter) node failed: "+ TOSTR(status))
}

void SpatterNode::init( int red, int green, int blue)
{
    _red = red;
    _green = green;
    _blue = blue;
    _layout = _roi_type = 0;
    // _layout = (unsigned) _outputs[0]->layout();

}

void SpatterNode::update_node()
{
}
