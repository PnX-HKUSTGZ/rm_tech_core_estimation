"""@brief Minimal non-registry adapters for the unchanged MMPose LiteHRNet tensor operations."""

from mmengine.model import BaseModule

BaseBackbone = BaseModule


def channel_shuffle(x, groups):
    """@brief Shuffle grouped channels as in MMPose. @param x NCHW tensor. @param groups Divisor of C."""
    batch, channels, height, width = x.shape
    if channels % groups:
        raise ValueError("Channels must be divisible by groups")
    return x.reshape(batch, groups, channels // groups, height, width).transpose(1, 2).contiguous().reshape(
        batch, channels, height, width)
