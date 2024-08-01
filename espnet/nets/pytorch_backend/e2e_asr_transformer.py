# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import math
from argparse import Namespace
from distutils.util import strtobool

import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect, ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos


from espnet.nets.pytorch_backend.transformer.decoder import Decoder

from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer

from espnet.nets.pytorch_backend.transformer.frontLayer import Front_Module
from espnet.nets.pytorch_backend.transformer.Conformer import Conformer


class E2E(torch.nn.Module):
    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, odim, args, modality, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        # Check the relative positional encoding type
        self.rel_pos_type = getattr(args, "rel_pos_type", None)
        if (
                self.rel_pos_type is None
                and args.transformer_encoder_attn_layer_type == "rel_mha"
        ):
            args.transformer_encoder_attn_layer_type = "legacy_rel_mha"
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )

        idim = args.adim
        self.modality = modality
        if self.modality == "audiovisual":
            self.visual_front = Front_Module(
                input_layer="conv3d",
                relu_type=getattr(args, "relu_type", "swish"),
            )
            self.audio_front = Front_Module(
                input_layer="conv1d",
                relu_type=getattr(args, "relu_type", "swish"),
            )
            self.visual_encoder = Conformer(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
                macaron_style=args.macaron_style,
                use_cnn_module=args.use_cnn_module,
                cnn_module_kernel=args.cnn_module_kernel,
                zero_triu=getattr(args, "zero_triu", False),

                backbone_type="conformer_s4",
            )

        elif self.modality == "audio":
            self.audio_front = Front_Module(
                input_layer="conv1d",
                relu_type=getattr(args, "relu_type", "swish"),
            )
            self.audio_encoder = Conformer(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,

                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
                macaron_style=args.macaron_style,
                use_cnn_module=args.use_cnn_module,
                cnn_module_kernel=args.cnn_module_kernel,
                zero_triu=getattr(args, "zero_triu", False),
            )
        elif self.modality == "video":
            self.visual_front = Front_Module(
                input_layer="conv3d",
                relu_type=getattr(args, "relu_type", "swish"),
            )
            self.visual_encoder = Conformer(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
                macaron_style=args.macaron_style,
                use_cnn_module=args.use_cnn_module,
                cnn_module_kernel=args.cnn_module_kernel,
                zero_triu=getattr(args, "zero_triu", False),
            )




        self.decoder_v = Decoder(
            odim=odim,
            attention_dim=args.ddim,
            attention_heads=args.dheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.r_decoder = Decoder(
            odim=odim,
            attention_dim=args.ddim,
            attention_heads=args.dheads,
            linear_units=args.dunits,
            num_blocks=args.r_dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        )

        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")

        # self.lsm_weight = a
        self.criterion_v = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.ctc_v = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
        )



    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder_v, ctc=CTCPrefixScorer(self.ctc_v, self.eos))

    def forward(self, x_v, x_a, lengths, uid, label):
        padding_mask = make_non_pad_mask(lengths).to(x_v.device).unsqueeze(-2)

        if self.modality == "audiovisual":
            x_v = self.visual_front(x_v)

            if x_a is not None:
                x_a = self.audio_front(x_a)
                x_v = torch.cat((x_v, x_a), dim=0)
                padding_mask = torch.cat((padding_mask, padding_mask), dim=0)
                label = torch.cat((label, label), dim=0)
                lengths = torch.cat((lengths, lengths), dim=0)
            x_v, _ = self.visual_encoder(x_v, padding_mask)


        elif self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
            padding_mask = make_non_pad_mask(lengths).to(x_v.device).unsqueeze(-2)
            x_v = self.audio_front(x_v)
            x_v, _ = self.audio_encoder(x_v, padding_mask)
        elif self.modality == "video":
            x_v = self.visual_front(x_v)
            x_v, _ = self.visual_encoder(x_v, padding_mask)

        # ctc loss
        loss_ctc_v, y_hat = self.ctc_v(x_v, lengths, label)


        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)

        pred_pad_v, _ = self.decoder_v(ys_in_pad, ys_mask, x_v, padding_mask)
        loss_v = self.criterion_v(pred_pad_v, ys_out_pad)

        reversed_label = torch.flip(label, dims=[2])
        ys_in_pad_reversed, ys_out_pad_reversed = add_sos_eos(reversed_label, self.sos, self.eos, self.ignore_id)
        pred_pad_reversed, _ = self.r_decoder(ys_in_pad_reversed, ys_mask, x_v, padding_mask)
        loss_att_reversed = self.criterion_v(pred_pad_reversed, ys_out_pad_reversed)


        loss = 0.6 * loss_v + 0.1 * loss_ctc_v + 0.3 * loss_att_reversed

        acc_v = th_accuracy(
            pred_pad_v.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )
        acc_reversed = th_accuracy(
            pred_pad_reversed.view(-1, self.odim), ys_out_pad_reversed, ignore_label=self.ignore_id
        )

        return loss, loss_v, loss_ctc_v,acc_v, loss_att_reversed, acc_reversed

    def get_feature(self, x_v, device):
        x_v = x_v.unsqueeze(0).to(device)
        if self.modality == "audiovisual":
            x_v = self.visual_front(x_v)
            x_v, _ = self.visual_encoder(x_v, None)
        elif self.modality == "video":
            x_v = self.visual_front(x_v)
            x_v, _ = self.visual_encoder(x_v, None)
        else:
            x_v = self.audio_front(x_v)
            x_v, _ = self.audio_encoder(x_v, None)
        return x_v

