#include <vx_ext_rpp.h>
#include "node_noise.h"
#include "exception.h"

NoiseTensorNode::NoiseTensorNode(const std::vector<rocalTensor *> &inputs,const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _noise_prob(NOISE_PROB_RANGE[0], NOISE_PROB_RANGE[1]),
        _salt_prob (SALT_PROB_RANGE[0], SALT_PROB_RANGE[1]),
        _noise_value(NOISE_RANGE[0], NOISE_RANGE[1]),
        _salt_value(SALT_RANGE[0], SALT_RANGE[1])


{
}

void NoiseTensorNode::create_node()
{
    if(_node)
        return;

    _noise_prob.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _salt_prob.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _noise_value.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _salt_value.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);
    vx_scalar seed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_seed);

    _node = vxExtrppNode_Noise(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _noise_prob.default_array(), _salt_prob.default_array(), _noise_value.default_array(), _salt_value.default_array(),seed, layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Noise_batch (vxExtrppNode_Noise) node failed: "+ TOSTR(status))
}

void NoiseTensorNode::init( float noise_prob, float salt_prob, float noise_value , float salt_value,int seed)
{
    _noise_prob.set_param(noise_prob);
    _salt_prob.set_param(salt_prob);
    _noise_value.set_param(noise_value);
    _salt_value.set_param(salt_value);
    _seed=seed;
    _layout = (int)_inputs[0]->info().layout();
    _roi_type = (int)_inputs[0]->info().roi_type();
}

void NoiseTensorNode::init( FloatParam* noise_prob, FloatParam* salt_prob, FloatParam* noise_value, FloatParam* salt_value, int seed)
{
    _noise_prob.set_param(core(noise_prob));
    _salt_prob.set_param(core(salt_prob));
    _noise_value.set_param(core(noise_value));
    _salt_value.set_param(core(salt_value));
    _seed=seed;
    _layout = (int)_inputs[0]->info().layout();
    _roi_type = (int)_inputs[0]->info().roi_type();
}


void NoiseTensorNode::update_node()
{
    _noise_prob.update_array();
    _salt_prob.update_array();
    _noise_value.update_array();
    _salt_value.update_array();
}
