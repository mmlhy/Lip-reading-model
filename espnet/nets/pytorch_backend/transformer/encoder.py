#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
from espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict

# from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    RotaryMultiHeadedAttention,
    LongformerSelfAttention,
    Rotary_Attention,
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,  # noqa: H301
    PositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer,EncoderLayer_S4
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



class Conformer(torch.nn.Module):
    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_adim=512,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        encoder_attn_layer_type="mha",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        is_embed=True,
        backbone_type="conformer"
    ):
        """Construct an Encoder object."""
        super(Conformer, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        self.encoder_attn_layer_type = encoder_attn_layer_type
        self.backbone_type = backbone_type
        if encoder_attn_layer_type == "rel_mha":
            pos_enc_class = RelPositionalEncoding
        elif encoder_attn_layer_type == "legacy_rel_mha":
            pos_enc_class = LegacyRelPositionalEncoding
        # -- backend module.
        self.is_embed = is_embed
        if is_embed:

            self.embed = torch.nn.Sequential(
      #          torch.nn.Linear(input_adim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if encoder_attn_layer_type == "mha":
            encoder_attn_layer = LongformerSelfAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif encoder_attn_layer_type == "legacy_rel_mha":
            encoder_attn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        elif encoder_attn_layer_type == "rope_mha":
            encoder_attn_layer = RotaryMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)
        if self.backbone_type == "conformer":
            self.encoders = repeat(
                num_blocks,
                lambda: EncoderLayer(
                    attention_dim,
                    encoder_attn_layer(*encoder_attn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    macaron_style,
                ),
            )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param str extract_features: the position for feature extraction
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if self.is_embed:
            if isinstance(self.embed, Conv2dSubsampling):
                xs, masks = self.embed(xs, masks)
            else:
                xs = self.embed(xs)


        xs, masks = self.encoders(xs, masks)


        if isinstance(xs, tuple):
            xs = xs[0]
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """

        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
