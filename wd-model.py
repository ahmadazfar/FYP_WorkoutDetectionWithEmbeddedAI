###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Keyword spotting network for AI85/AI86
"""
from torch import nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import ai8x


class WdModel(nn.Module):
    """
    Compound KWS20 v3 Audio net, all with Conv1Ds
    """

    # num_classes = n keywords + 1 unknown
    def __init__(
            self,
            num_classes=3,
            num_channels=6,
            dimensions=(6, 200),  # pylint: disable=unused-argument
            bias=False,
            **kwargs

    ):
        super().__init__()

        self.voice_conv1 = ai8x.FusedConv1dReLU(num_channels, 32, 3, stride=1, padding=0,
                                                bias=bias, **kwargs)

        self.voice_conv2 = ai8x.FusedConv1dReLU(32, 64, 3, stride=1, padding=1,
                                                bias=bias, **kwargs)
        
        self.drop = nn.Dropout(p=0.2)

        self.voice_conv3 = ai8x.FusedConv1dReLU(64, 4, 3, stride=1, padding=1,
                                        bias=bias, **kwargs)
                                

        self.fc = ai8x.Linear(792, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        x = x.float()
        x = self.voice_conv1(x)
        x = self.voice_conv2(x)
        x = self.drop(x)
        x = self.voice_conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def wdmodel(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    rn AI85KWS20Net(**kwargs)
    """
    assert not pretrained
    return WdModel(**kwargs)


models = [
    {
        'name': 'wdmodel',
        'min_input': 1,
        'dim': 1,
    },
]
