"""Image path of the released segmentation checkpoint."""

import torch
from torch import nn
from torch.nn import functional as F
from segment_anything import sam_model_registry
from segment_anything.modeling import MaskDecoder, TwoWayTransformer

from medfoe import FEATURE_LAYERS
from medfoe.hardnet import HarDNet


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(CNNBlock, self).__init__()
        P = int((kernel_size-1)/2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in, inx=-1):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x_out = self.conv2(x)
        return x_out


class UpBlockSkip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, func=None, drop=0):
        super(UpBlockSkip, self).__init__()
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in, x_up):
        x = self.Upsample(x_in)
        x_cat = torch.cat((x, x_up), 1)
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_cat))))
        if self.func == 'tanh':
            return F.tanh(self.BN(x1))
        elif self.func == 'relu':
            return F.leaky_relu(self.BN(x1))
        elif self.func == 'sigmoid':
            return F.sigmoid(self.BN(x1))
        else:
            return x1


class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super(SmallDecoder, self).__init__()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0)
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = F.tanh(self.final(z))
        return out


class ModelEmb(nn.Module):
    def __init__(self, pretrained=True):
        super(ModelEmb, self).__init__()
        self.backbone = HarDNet(depth_wise=False, arch=85, pretrained=pretrained)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SmallDecoder(d, out=256)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img, size=None):
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
        return dense_embeddings


class PICAISegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        sam = sam_model_registry['vit_b']()
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.dense_encoder = ModelEmb(pretrained=False)
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=4,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
        )
        self.activations = {}
        self._handles = [self.image_encoder.blocks[layer].register_forward_hook(self._record(layer))
                         for layer in FEATURE_LAYERS]

    def _record(self, layer):
        def hook(module, inputs, output):
            self.activations[layer] = output
        return hook

    def forward(self, images):
        self.activations.clear()
        image_small = F.interpolate(images, (512, 512), mode='bilinear', align_corners=True)
        image_embedding = self.image_encoder(images)
        dense_embedding = self.dense_encoder(image_small)
        sparse, _ = self.prompt_encoder(points=None, boxes=None, masks=None)
        masks, _ = self.mask_decoder(
            image_embeddings=image_embedding, image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense_embedding,
            multimask_output=True,
        )
        return F.interpolate(masks, images.shape[-2:], mode='bilinear', align_corners=False)

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        # Text encoder parameters are not used during segmentation inference.
        state = {key: value for key, value in state.items()
                 if not key.startswith('text_encoder.') and key != 'logit_scale'}
        model = cls()
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()
