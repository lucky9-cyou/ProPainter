import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8
import numpy as np
import RAFT.trt_utils as trt_utils

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in args._get_kwargs():
            args.dropout = 0

        if 'alternate_corr' not in args._get_kwargs():
            args.alternate_corr = False
        
        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
            
            self.fnet_engine = trt_utils.load_engine(args.fnet_path)
            _, self.fnet_outputs, self.fnet_bindings = trt_utils.allocate_buffers(
                engine=self.fnet_engine)
            for host_device_buffer in self.fnet_outputs:
                print(
                    f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
                    f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
                )
            self.fnet_context = self.fnet_engine.create_execution_context()

            
            self.cnet_engine = trt_utils.load_engine(args.cnet_path)
            _, self.cnet_outputs, self.cnet_bindings = trt_utils.allocate_buffers(
                engine=self.cnet_engine)
            for host_device_buffer in self.cnet_outputs:
                print(
                    f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
                    f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
                )
            self.cnet_context = self.cnet_engine.create_execution_context()
            
            self.update_block_engine = trt_utils.load_engine(args.update_block_path)
            _, self.update_block_outputs, self.update_block_bindings = trt_utils.allocate_buffers(
                engine=self.update_block_engine)
            for host_device_buffer in self.update_block_outputs:
                print(
                    f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
                    f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
                )
            self.update_block_context = self.update_block_engine.create_execution_context()
            

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=True):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            batch_size = image1.shape[0]
            x = torch.cat([image1, image2], dim=0)
            
            # fmap = self.fnet(x)
            self.fnet_context.set_input_shape('x', x.shape)
            trt_utils.do_inference_v2(self.fnet_context, bindings=[int(x.data_ptr())] + self.fnet_bindings, outputs=self.fnet_outputs)
            fmap = trt_utils.ptr_to_tensor(self.fnet_outputs[0].device, self.fnet_outputs[0].nbytes, self.fnet_outputs[0].shape)[:batch_size * 2]
            
            fmap1, fmap2 = torch.split(fmap, [batch_size, batch_size], dim=0)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # cnet = self.cnet(image1)
            self.cnet_context.set_input_shape('x', image1.shape)
            trt_utils.do_inference_v2(self.cnet_context, bindings=[int(image1.data_ptr())] + self.cnet_bindings, outputs=self.cnet_outputs)
            cnet = trt_utils.ptr_to_tensor(self.cnet_outputs[0].device, self.cnet_outputs[0].nbytes, self.cnet_outputs[0].shape)[:batch_size]
            
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                # net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
                self.update_block_context.set_input_shape('net_in', net.shape)
                self.update_block_context.set_input_shape('inp', inp.shape)
                self.update_block_context.set_input_shape('corr', corr.shape)
                self.update_block_context.set_input_shape('flow', flow.shape)
                
                trt_utils.do_inference_v2(self.update_block_context, bindings=[int(net.data_ptr()), int(inp.data_ptr()), int(corr.data_ptr()), int(flow.data_ptr())] + self.update_block_bindings, outputs=self.update_block_outputs)
                net = trt_utils.ptr_to_tensor(self.update_block_outputs[0].device, self.update_block_outputs[0].nbytes, self.update_block_outputs[0].shape)[:batch_size]
                up_mask = trt_utils.ptr_to_tensor(self.update_block_outputs[1].device, self.update_block_outputs[1].nbytes, self.update_block_outputs[1].shape)[:batch_size]
                delta_flow = trt_utils.ptr_to_tensor(self.update_block_outputs[2].device, self.update_block_outputs[2].nbytes, self.update_block_outputs[2].shape)[:batch_size]
                
                up_mask = 0.25 * up_mask

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
    
    def export_quantized_model(self):
        fnet_input = torch.randn(24, 3, 640, 360).cuda()
        cnet_input = torch.randn(12, 3, 640, 360).cuda()
        # net.shape, inp.shape, corr.shape, flow.shape
        update_net = torch.randn(12, 128, 80, 45).cuda()
        update_inp = torch.randn(12, 128, 80, 45).cuda()
        update_corr = torch.randn(12, 324, 80, 45).cuda()
        update_flow = torch.randn(12, 2, 80, 45).cuda()
        
        onnx_program = torch.onnx.export(self.fnet, fnet_input, 'raft_fnet_quan.onnx', input_names=["x"], output_names=["fmap"], dynamic_axes={'x': {0: 'batch_size'}, 'fmap': {0: 'batch_size'}}, opset_version=20)
        onnx_program = torch.onnx.export(self.cnet, cnet_input, 'raft_cnet_quan.onnx', input_names=["x"], output_names=["cnet"], dynamic_axes={'x': {0: 'batch_size'}, 'cnet': {0: 'batch_size'}}, opset_version=20)
        onnx_program = torch.onnx.export(self.update_block, (update_net, update_inp, update_corr, update_flow), 'raft_update_block_quan.onnx', input_names=["net_in", "inp", "corr", "flow"], output_names=["net_out", "up_mask", "delta_flow"], dynamic_axes={'net_in': {0: 'batch_size'}, 'inp': {0: 'batch_size'}, 'corr': {0: 'batch_size'}, 'flow': {0: 'batch_size'}, 'up_mask': {0: 'batch_size'}, 'delta_flow': {0: 'batch_size'}}, opset_version=20)
