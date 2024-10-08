""" Towards An End-to-End Framework for Video Inpainting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from einops import rearrange

from model.modules.base_module import BaseNetwork
from model.modules.sparse_transformer import (
    TemporalSparseTransformerBlock,
    SoftSplit,
    SoftComp,
)
from model.modules.spectral_norm import spectral_norm as _spectral_norm
from model.modules.flow_loss_utils import flow_warp
from model.modules.deformconv import ModulatedDeformConv2d

import tensorrt as trt
from model import trt_utils
import time
import deform_conv2d_onnx_exporter

from .misc import constant_init


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw


class DeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""

    def __init__(self, *args, **kwargs):
        # self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 3)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond_feat, flow):
        out = self.conv_offset(cond_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel, learnable=True, feat=True):
        super(BidirectionalPropagation, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ["backward_1", "forward_1"]
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = DeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16
                )

                # self.backbone[module] = nn.Sequential(
                #     nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                #     nn.Conv2d(channel, channel, 3, 1, 1),
                # )

            # self.fuse = nn.Sequential(
            #     nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
            #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #     nn.Conv2d(channel, channel, 3, 1, 1),
            # )
        if feat:
            # self.deform_align_back = trt_utils.load_engine("/root/ProPainter/weights/inpainter_feat_back_deform_align_best.engine")
            # _, self.deform_align_back_outputs, self.deform_align_back_bindings = trt_utils.allocate_buffers(self.deform_align_back)
            # for host_device_buffer in self.deform_align_back_outputs:
            #     print(
            #             f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
            #             f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
            #     )
            # self.deform_align_back_context = self.deform_align_back.create_execution_context()
            # self.deform_align_back_context.set_input_shape('feat', [1, 128, 160, 90])
            # self.deform_align_back_context.set_input_shape('cond', [1, 261, 160, 90])
            # self.deform_align_back_context.set_input_shape('flow', [1, 2, 160, 90])

            # self.deform_align_forw = trt_utils.load_engine("/root/ProPainter/weights/inpainter_feat_forw_deform_align_best.engine")
            # _, self.deform_align_forw_outputs, self.deform_align_forw_bindings = trt_utils.allocate_buffers(self.deform_align_forw)
            # for host_device_buffer in self.deform_align_forw_outputs:
            #     print(
            #             f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
            #             f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
            #     )
            # self.deform_align_forw_context = self.deform_align_forw.create_execution_context()
            # self.deform_align_forw_context.set_input_shape('feat', [1, 128, 160, 90])
            # self.deform_align_forw_context.set_input_shape('cond', [1, 261, 160, 90])
            # self.deform_align_forw_context.set_input_shape('flow', [1, 2, 160, 90])

            self.backbone_back = trt_utils.load_engine(
                "/root/ProPainter/weights/inpainter_feat_back_backbone_res_best.engine"
            )
            _, self.backbone_back_outputs, self.backbone_back_bindings = (
                trt_utils.allocate_buffers(
                    self.backbone_back, shapes={"ouput": [1, 258, 320, 320]}
                )
            )
            for host_device_buffer in self.backbone_back_outputs:
                print(
                    f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
                    f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
                )
            self.backbone_back_context = self.backbone_back.create_execution_context()

            self.backbone_forw = trt_utils.load_engine(
                "/root/ProPainter/weights/inpainter_feat_forw_backbone_res_best.engine"
            )
            _, self.backbone_forw_outputs, self.backbone_forw_bindings = (
                trt_utils.allocate_buffers(
                    self.backbone_forw, shapes={"ouput": [1, 258, 320, 320]}
                )
            )
            for host_device_buffer in self.backbone_forw_outputs:
                print(
                    f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
                    f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
                )
            self.backbone_forw_context = self.backbone_forw.create_execution_context()

            self.fuse_ = trt_utils.load_engine(
                "/root/ProPainter/weights/inpainter_feat_fuse_res_best.engine"
            )
            _, self.fuse_outputs, self.fuse_bindings = trt_utils.allocate_buffers(
                self.fuse_, shapes={"ouput": [11, 258, 320, 320]}
            )
            for host_device_buffer in self.fuse_outputs:
                print(
                    f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
                    f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
                )
            self.fuse_context = self.fuse_.create_execution_context()

    def binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        # return mask.float()
        return mask.to(mask)

    def forward(self, x, flows_forward, flows_backward, mask, interpolation="bilinear"):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        feats, masks = {}, {}
        feats["input"] = [x[:, i, :, :, :] for i in range(0, t)]
        masks["input"] = [mask[:, i, :, :, :] for i in range(0, t)]

        prop_list = ["backward_1", "forward_1"]
        cache_list = ["input"] + prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if "backward" in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = flow_warp(
                        feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation
                    )

                    if self.learnable:
                        cond = torch.cat(
                            [
                                feat_current,
                                feat_warped,
                                flow_prop,
                                flow_vaild_mask,
                                mask_current,
                            ],
                            dim=1,
                        )
                        # forward_1 feat prop shape: torch.Size([1, 128, 160, 90]) cond shape: torch.Size([1, 261, 160, 90]) flow prop shape: torch.Size([1, 2, 160, 90])
                        # backward_1 feat prop shape: torch.Size([1, 128, 160, 90]) cond shape: torch.Size([1, 261, 160, 90]) flow prop shape: torch.Size([1, 2, 160, 90])
                        feat_prop = self.deform_align[module_name](
                            feat_prop, cond, flow_prop
                        )
                        # if module_name == 'backward_1':
                        #     trt_utils.do_inference_v2(self.deform_align_back_context, bindings=[int(feat_prop.data_ptr()), int(cond.data_ptr()), int(flow_prop.data_ptr())] + self.deform_align_back_bindings, outputs=self.deform_align_back_outputs)
                        #     feat_prop = trt_utils.ptr_to_tensor(self.deform_align_back_outputs[0].device, self.deform_align_back_outputs[0].nbytes, self.deform_align_back_outputs[0].shape)
                        # else:
                        #     trt_utils.do_inference_v2(self.deform_align_forw_context, bindings=[int(feat_prop.data_ptr()), int(cond.data_ptr()), int(flow_prop.data_ptr())] + self.deform_align_forw_bindings, outputs=self.deform_align_forw_outputs)
                        #     feat_prop = trt_utils.ptr_to_tensor(self.deform_align_forw_outputs[0].device, self.deform_align_forw_outputs[0].nbytes, self.deform_align_forw_outputs[0].shape)
                        # feat prop shape: torch.Size([1, 128, 160, 90])
                        # feat prop shape: torch.Size([1, 128, 160, 90])
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(
                            mask_prop, flow_prop.permute(0, 2, 3, 1)
                        )
                        mask_prop_valid = self.binary_mask(mask_prop_valid)

                        union_vaild_mask = self.binary_mask(
                            mask_current * flow_vaild_mask * (1 - mask_prop_valid)
                        )
                        feat_prop = (
                            union_vaild_mask * feat_warped
                            + (1 - union_vaild_mask) * feat_current
                        )
                        # update mask
                        mask_prop = self.binary_mask(
                            mask_current
                            * (1 - (flow_vaild_mask * (1 - mask_prop_valid)))
                        )

                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)

                    # forward_1 feat shape: torch.Size([1, 258, 160, 90])
                    # backward_1 feat shape: torch.Size([1, 258, 160, 90])
                    # feat_prop = feat_prop + self.backbone[module_name](feat)
                    if module_name == "backward_1":
                        self.backbone_back_context.set_input_shape("feat", feat.shape)
                        trt_utils.do_inference_v2(
                            self.backbone_back_context,
                            bindings=[int(feat.data_ptr())]
                            + self.backbone_back_bindings,
                            outputs=self.backbone_back_outputs,
                        )
                        output_shape = [
                            feat.shape[0],
                            128,
                            feat.shape[2],
                            feat.shape[3],
                        ]
                        feat_prop = feat_prop + trt_utils.ptr_to_tensor(
                            self.backbone_back_outputs[0].device,
                            self.backbone_back_outputs[0].nbytes,
                            output_shape,
                        )
                    else:
                        self.backbone_forw_context.set_input_shape("feat", feat.shape)
                        trt_utils.do_inference_v2(
                            self.backbone_forw_context,
                            bindings=[int(feat.data_ptr())]
                            + self.backbone_forw_bindings,
                            outputs=self.backbone_forw_outputs,
                        )
                        output_shape = [
                            feat.shape[0],
                            128,
                            feat.shape[2],
                            feat.shape[3],
                        ]
                        feat_prop = feat_prop + trt_utils.ptr_to_tensor(
                            self.backbone_forw_outputs[0].device,
                            self.backbone_forw_outputs[0].nbytes,
                            output_shape,
                        )
                    # feat prop shape: torch.Size([1, 128, 160, 90])
                    # feat prop shape: torch.Size([1, 128, 160, 90])

                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(feats["backward_1"], dim=1).view(t, c, h, w)
        outputs_f = torch.stack(feats["forward_1"], dim=1).view(t, c, h, w)

        if self.learnable:
            mask_in = mask.view(t, 2, h, w)
            masks_b, masks_f = None, None
            feat = torch.cat([outputs_b, outputs_f, mask_in], dim=1)
            # feat shape: torch.Size([6, 258, 160, 90])
            # feat shape: torch.Size([11, 258, 160, 90])
            # outputs = self.fuse(feat) + x.view(-1, c, h, w)
            self.fuse_context.set_input_shape("feat", feat.shape)
            trt_utils.do_inference_v2(
                self.fuse_context,
                bindings=[int(feat.data_ptr())] + self.fuse_bindings,
                outputs=self.fuse_outputs,
            )
            output_shape = [feat.shape[0], 128, feat.shape[2], feat.shape[3]]
            outputs = trt_utils.ptr_to_tensor(
                self.fuse_outputs[0].device, self.fuse_outputs[0].nbytes, output_shape
            ) + x.view(-1, c, h, w)
            # outputs shape: torch.Size([6, 128, 160, 90])
            # outputs shape: torch.Size([11, 128, 160, 90])
        else:
            masks_b = torch.stack(masks["backward_1"], dim=1)
            masks_f = torch.stack(masks["forward_1"], dim=1)
            outputs = outputs_f

        return (
            outputs_b.view(b, -1, c, h, w),
            outputs_f.view(b, -1, c, h, w),
            outputs.view(b, -1, c, h, w),
            masks_f,
        )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        # torch.Size([18, 5, 640, 360])
        # torch.Size([18, 64, 320, 180])
        # torch.Size([18, 64, 320, 180])
        # torch.Size([18, 64, 320, 180])
        # torch.Size([18, 64, 320, 180])
        # torch.Size([18, 128, 160, 90])
        # torch.Size([18, 128, 160, 90])
        # torch.Size([18, 256, 160, 90])
        # torch.Size([18, 256, 160, 90])
        # torch.Size([18, 384, 160, 90])
        # torch.Size([18, 640, 160, 90])
        # torch.Size([18, 512, 160, 90])
        # torch.Size([18, 768, 160, 90])
        # torch.Size([18, 384, 160, 90])
        # torch.Size([18, 640, 160, 90])
        # torch.Size([18, 256, 160, 90])
        # torch.Size([18, 512, 160, 90])
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i == 10:
                break
            out = layer(out)

        x = x0.view(bt, 2, 128, h, w)
        o = out.view(bt, 2, 192, h, w)
        out = torch.cat([x, o], 2).view(bt, 640, h, w)
        out = self.layers[10](out)
        out = self.layers[11](out)
        x = x0.view(bt, 4, 64, h, w)
        o = out.view(bt, 4, 128, h, w)
        out = torch.cat([x, o], 2).view(bt, 768, h, w)
        out = self.layers[12](out)
        out = self.layers[13](out)
        x = x0.view(bt, 8, 32, h, w)
        o = out.view(bt, 8, 48, h, w)
        out = torch.cat([x, o], 2).view(bt, 640, h, w)
        out = self.layers[14](out)
        out = self.layers[15](out)
        x = x0.view(bt, 1, 256, h, w)
        o = out.view(bt, 1, 256, h, w)
        out = torch.cat([x, o], 2).view(bt, 512, h, w)
        out = self.layers[16](out)
        out = self.layers[17](out)

        return out


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(x)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, model_path=None):
        super(InpaintGenerator, self).__init__()
        channel = 128
        hidden = 512

        # encoder
        # self.encoder = Encoder()
        self.encoder_engine = trt_utils.load_engine(
            "/root/ProPainter/weights/inpainter_encoder_res_best.engine"
        )
        _, self.encoder_outputs, self.encoder_bindings = trt_utils.allocate_buffers(
            self.encoder_engine, shapes={"ouput": [18, 5, 1280, 1280]}
        )
        for host_device_buffer in self.encoder_outputs:
            print(
                f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
                f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
            )
        self.encoder_context = self.encoder_engine.create_execution_context()

        # decoder
        # self.decoder = nn.Sequential(
        #     deconv(channel, 128, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     deconv(64, 64, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        # )

        self.decoder_engine = trt_utils.load_engine(
            "/root/ProPainter/weights/inpainter_decoder_res_best.engine"
        )
        _, self.decoder_outputs, self.decoder_bindings = trt_utils.allocate_buffers(
            self.decoder_engine, shapes={"ouput": [11, 128, 320, 320]}
        )
        for host_device_buffer in self.decoder_outputs:
            print(
                f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
                f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
            )
        self.decoder_context = self.decoder_engine.create_execution_context()

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        # feature propagation module
        self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True, feat=True)

        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlock(
            dim=hidden,
            n_head=num_heads,
            window_size=window_size,
            pool_size=pool_size,
            depths=depths,
            t2t_params=t2t_params,
        )
        if init_weights:
            self.init_weights()

        if model_path is not None:
            print("Pretrained ProPainter has loaded...")
            ckpt = torch.load(model_path, map_location="cpu")
            self.load_state_dict(ckpt, strict=False)

        # print network parameter number
        self.print_network()

    def img_propagation(
        self, masked_frames, completed_flows, masks, interpolation="nearest"
    ):
        _, _, prop_frames, updated_masks = self.img_prop_module(
            masked_frames, completed_flows[0], completed_flows[1], masks, interpolation
        )
        return prop_frames, updated_masks

    def forward(
        self,
        masked_frames,
        completed_flows,
        masks_in,
        masks_updated,
        num_local_frames,
        interpolation="bilinear",
        t_dilation=2,
    ):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """

        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.size()

        # extracting features
        # [9, 5, 640, 360] -> [18, 5, 640, 360]

        # enc_feat = self.encoder(
        #     torch.cat(
        #         [
        #             masked_frames.view(b * t, 3, ori_h, ori_w),
        #             masks_in.view(b * t, 1, ori_h, ori_w),
        #             masks_updated.view(b * t, 1, ori_h, ori_w),
        #         ],
        #         dim=1,
        #     )
        # )

        enc_input = torch.cat(
            [
                masked_frames.view(b * t, 3, ori_h, ori_w),
                masks_in.view(b * t, 1, ori_h, ori_w),
                masks_updated.view(b * t, 1, ori_h, ori_w),
            ],
            dim=1,
        )
        self.encoder_context.set_input_shape("input", enc_input.shape)
        trt_utils.do_inference_v2(
            self.encoder_context,
            bindings=[int(enc_input.data_ptr())] + self.encoder_bindings,
            outputs=self.encoder_outputs,
        )
        enc_output_shape = [
            enc_input.shape[0],
            128,
            enc_input.shape[2] // 4,
            enc_input.shape[3] // 4,
        ]
        enc_feat = trt_utils.ptr_to_tensor(
            self.encoder_outputs[0].device,
            self.encoder_outputs[0].nbytes,
            enc_output_shape,
        )

        _, c, h, w = enc_feat.size()
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        fold_feat_size = (h, w)

        ds_flows_f = (
            F.interpolate(
                completed_flows[0].view(-1, 2, ori_h, ori_w),
                scale_factor=1 / 4,
                mode="bilinear",
                align_corners=False,
            ).view(b, l_t - 1, 2, h, w)
            / 4.0
        )
        ds_flows_b = (
            F.interpolate(
                completed_flows[1].view(-1, 2, ori_h, ori_w),
                scale_factor=1 / 4,
                mode="bilinear",
                align_corners=False,
            ).view(b, l_t - 1, 2, h, w)
            / 4.0
        )
        ds_mask_in = F.interpolate(
            masks_in.reshape(-1, 1, ori_h, ori_w), scale_factor=1 / 4, mode="nearest"
        ).view(b, t, 1, h, w)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local = F.interpolate(
            masks_updated[:, :l_t].reshape(-1, 1, ori_h, ori_w),
            scale_factor=1 / 4,
            mode="nearest",
        ).view(b, l_t, 1, h, w)

        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(
                b, t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1)
            )
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(
                b, l_t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1)
            )

        prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
        # local_feat shape: torch.Size([1, 6, 128, 160, 90]) ds_flows_f shape: torch.Size([1, 5, 2, 160, 90]) ds_flows_b shape: torch.Size([1, 5, 2, 160, 90]) prop_mask_in shape: torch.Size([1, 6, 2, 160, 90]) interpolation: bilinear
        # local_feat shape: torch.Size([1, 11, 128, 160, 90]) ds_flows_f shape: torch.Size([1, 10, 2, 160, 90]) ds_flows_b shape: torch.Size([1, 10, 2, 160, 90]) prop_mask_in shape: torch.Size([1, 11, 2, 160, 90]) interpolation: bilinear
        _, _, local_feat, _ = self.feat_prop_module(
            local_feat, ds_flows_f, ds_flows_b, prop_mask_in
        )
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        mask_pool_l = rearrange(mask_pool_l, "b t c h w -> b t h w c").contiguous()
        # transformers shape: torch.Size([1, 9, 54, 30, 512]) (160, 90) torch.Size([1, 6, 54, 30, 1])
        # transformers shape: torch.Size([1, 18, 54, 30, 512]) (160, 90) torch.Size([1, 11, 54, 30, 1])
        trans_feat = self.transformers(
            trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation
        )
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)

        enc_feat = enc_feat + trans_feat

        if self.training:
            output = self.decoder(enc_feat.view(-1, c, h, w))
            output = torch.tanh(output).view(b, t, 3, ori_h, ori_w)
        else:
            # decoder input shape: torch.Size([6, 128, 160, 90]) -> torch.Size([11, 128, 160, 90])
            # output = self.decoder(enc_feat[:, :l_t].view(-1, c, h, w))

            decoder_input = enc_feat[:, :l_t].view(-1, c, h, w)
            self.decoder_context.set_input_shape("input", decoder_input.shape)
            trt_utils.do_inference_v2(
                self.decoder_context,
                bindings=[int(decoder_input.data_ptr())] + self.decoder_bindings,
                outputs=self.decoder_outputs,
            )
            output_shape = [
                decoder_input.shape[0],
                3,
                decoder_input.shape[2] * 4,
                decoder_input.shape[3] * 4,
            ]
            output = trt_utils.ptr_to_tensor(
                self.decoder_outputs[0].device,
                self.decoder_outputs[0].nbytes,
                output_shape,
            )

            output = torch.tanh(output).view(b, l_t, 3, ori_h, ori_w)

        return output

    def export_quantized_model(self):
        deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
        encoder_input = torch.randn(18, 5, 1280, 1280).to(torch.half).cuda()
        decoder_input = torch.randn(11, 128, 320, 320).to(torch.half).cuda()

        feat_back_deform_align_feat = (
            torch.randn(1, 128, 320, 320).to(torch.half).cuda()
        )
        feat_back_deform_align_cond = (
            torch.randn(1, 261, 320, 320).to(torch.half).cuda()
        )
        feat_back_deform_align_flow = torch.randn(1, 2, 320, 320).to(torch.half).cuda()

        feat_forw_deform_align_feat = (
            torch.randn(1, 128, 320, 320).to(torch.half).cuda()
        )
        feat_forw_deform_align_cond = (
            torch.randn(1, 261, 320, 320).to(torch.half).cuda()
        )
        feat_forw_deform_align_flow = torch.randn(1, 2, 320, 320).to(torch.half).cuda()

        feat_back_backbone_feat = torch.randn(1, 258, 320, 320).to(torch.half).cuda()
        feat_forw_backbone_feat = torch.randn(1, 258, 320, 320).to(torch.half).cuda()

        feat_fuse_feat = torch.randn(11, 258, 320, 320).to(torch.half).cuda()

        onnx_program = torch.onnx.export(
            self.encoder,
            encoder_input,
            "inpainter_encoder_res.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            },
            opset_version=20,
        )
        onnx_program = torch.onnx.export(
            self.decoder,
            decoder_input,
            "inpainter_decoder_res.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            },
            opset_version=20,
        )

        # TODO: deforma conv2d do not support dynamic axes

        onnx_program = torch.onnx.export(
            self.feat_prop_module.backbone["backward_1"],
            feat_back_backbone_feat,
            "inpainter_feat_back_backbone_res.onnx",
            input_names=["feat"],
            output_names=["output"],
            dynamic_axes={
                "feat": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            opset_version=20,
        )
        onnx_program = torch.onnx.export(
            self.feat_prop_module.backbone["forward_1"],
            feat_forw_backbone_feat,
            "inpainter_feat_forw_backbone_res.onnx",
            input_names=["feat"],
            output_names=["output"],
            dynamic_axes={
                "feat": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            opset_version=20,
        )
        onnx_program = torch.onnx.export(
            self.feat_prop_module.fuse,
            feat_fuse_feat,
            "inpainter_feat_fuse_res.onnx",
            input_names=["feat"],
            output_names=["output"],
            dynamic_axes={
                "feat": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            },
            opset_version=20,
        )


# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################
class Discriminator(BaseNetwork):
    def __init__(
        self,
        in_channels=3,
        use_sigmoid=False,
        use_spectral_norm=True,
        init_weights=True,
    ):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=nf * 1,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 1,
                    nf * 2,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 2,
                    nf * 4,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 4,
                    nf * 4,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 4,
                    nf * 4,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                nf * 4,
                nf * 4,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
            ),
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


class Discriminator_2D(BaseNetwork):
    def __init__(
        self,
        in_channels=3,
        use_sigmoid=False,
        use_spectral_norm=True,
        init_weights=True,
    ):
        super(Discriminator_2D, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=nf * 1,
                    kernel_size=(1, 5, 5),
                    stride=(1, 2, 2),
                    padding=(0, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 1,
                    nf * 2,
                    kernel_size=(1, 5, 5),
                    stride=(1, 2, 2),
                    padding=(0, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 2,
                    nf * 4,
                    kernel_size=(1, 5, 5),
                    stride=(1, 2, 2),
                    padding=(0, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 4,
                    nf * 4,
                    kernel_size=(1, 5, 5),
                    stride=(1, 2, 2),
                    padding=(0, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 4,
                    nf * 4,
                    kernel_size=(1, 5, 5),
                    stride=(1, 2, 2),
                    padding=(0, 2, 2),
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                nf * 4,
                nf * 4,
                kernel_size=(1, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
            ),
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
