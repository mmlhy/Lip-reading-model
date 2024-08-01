#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet
from espnet.nets.pytorch_backend.backbones.mobilenetv3_extractor import mobilenetv3_large
from espnet.nets.pytorch_backend.nets_utils import rename_state_dict

# from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,  # noqa: H301
    PositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.raw_embeddings import (
    AudioEmbedding,
    VideoEmbedding,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Front_Module(torch.nn.Module):

    def __init__(
        self,

        input_layer="conv2d",

        relu_type="prelu",
        a_upsample_ratio=1,
    ):
        """Construct an Encoder object."""
        super(Front_Module, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)


        # -- frontend module.
        if input_layer == "conv1d":
            self.frontend = Conv1dResNet(
                relu_type=relu_type,
                a_upsample_ratio=a_upsample_ratio,
            )
            self.linear = nn.Linear(512,768)
        elif input_layer == "conv3d":
            self.frontend = Conv3dResNet(relu_type=relu_type)
            self.linear = nn.Linear(512, 768)
        elif input_layer == "mobilenetv3":
            self.frontend = mobilenetv3_large()
            self.linear = nn.Linear(960, 768)



    def forward(self, xs):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param str extract_features: the position for feature extraction
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        xs = self.frontend(xs)
        xs = self.linear(xs)

        return xs





