# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class RecurrentEncoder(nn.Module):
    """A GRU or LSTM encoder for inputs of shape (T, N, num_features).

    Unlike `TDSConvEncoder`, a recurrent layer does not shorten the time axis.
    The first ``context_frames`` outputs are therefore dropped so that the output
    has T - context_frames steps, exactly as for `TDSConvEncoder`. With the default
    ``context_frames=124`` (4 TDS conv blocks, each losing kernel_width - 1 = 31
    frames), the emissions stay aligned with the labels, which only cover the
    un-padded window, and the CTC length logic in ``lightning.py`` is unchanged.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        hidden_size (int): Hidden size of the recurrent layers per direction.
        num_layers (int): Number of stacked recurrent layers.
        bidirectional (bool): Whether the recurrent layers are bidirectional.
        dropout (float): Dropout between stacked recurrent layers (only applied
            when ``num_layers > 1``).
        rnn_type (str): Either "gru" or "lstm".
        context_frames (int): Number of leading output frames to drop.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1,
        rnn_type: str = "gru",
        context_frames: int = 124,
    ) -> None:
        super().__init__()

        assert rnn_type in {"gru", "lstm"}, f"Unsupported rnn_type: {rnn_type}"
        assert context_frames >= 0
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        # batch_first defaults to False, so inputs and outputs are (T, N, ...)
        self.rnn = rnn_cls(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.context_frames = context_frames
        self.output_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(inputs)  # (T, N, output_size)
        return x[self.context_frames :]  # (T - context_frames, N, output_size)


class ConvRecurrentEncoder(nn.Module):
    """A CNN + RNN hybrid encoder for inputs of shape (T, N, num_features):
    a (shallow) `TDSConvEncoder` followed by a `RecurrentEncoder`.

    The convolutions shorten the time axis by ``len(block_channels) *
    (kernel_width - 1)`` frames and the recurrent encoder drops the remaining
    ``context_frames`` minus that amount, so that the output has exactly
    T - context_frames steps, as for `TDSConvEncoder`.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): Number of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
        hidden_size (int): Hidden size of the recurrent layers per direction.
        num_layers (int): Number of stacked recurrent layers.
        bidirectional (bool): Whether the recurrent layers are bidirectional.
        dropout (float): Dropout between stacked recurrent layers.
        rnn_type (str): Either "gru" or "lstm".
        context_frames (int): Total number of leading frames to drop (conv
            receptive field plus recurrent warm-up). Must be at least the
            frames already lost to the convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24),
        kernel_width: int = 32,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1,
        rnn_type: str = "gru",
        context_frames: int = 124,
    ) -> None:
        super().__init__()

        conv_frames_lost = len(block_channels) * (kernel_width - 1)
        assert conv_frames_lost <= context_frames, (
            f"Convolutions already lose {conv_frames_lost} frames, "
            f"which exceeds context_frames={context_frames}"
        )
        self.conv = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )
        self.rnn = RecurrentEncoder(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            context_frames=context_frames - conv_frames_lost,
        )
        self.output_size = self.rnn.output_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rnn(self.conv(inputs))  # (T - context_frames, N, output_size)


class ChunkedTransformerEncoder(nn.Module):
    """A Transformer encoder for inputs of shape (T, N, num_features) that applies
    self-attention within non-overlapping blocks of ``chunk_size`` frames.

    Full self-attention over a whole test session (tens of thousands of frames)
    does not fit in memory, so attention is block-local, which works for any
    sequence length. Sinusoidal positional encodings are relative to the start of
    each block. As in `RecurrentEncoder`, the first ``context_frames`` outputs are
    dropped so the output has T - context_frames steps, as for `TDSConvEncoder`.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        d_model (int): Transformer width. Must be even and divisible by
            ``num_heads``.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Hidden size of the feedforward blocks.
        dropout (float): Dropout inside the Transformer layers.
        chunk_size (int): Number of frames attended to jointly (125 frames = 1 s).
        context_frames (int): Number of leading output frames to drop.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        chunk_size: int = 128,
        context_frames: int = 124,
    ) -> None:
        super().__init__()

        assert d_model % 2 == 0 and d_model % num_heads == 0
        assert chunk_size > 0 and context_frames >= 0
        self.chunk_size = chunk_size
        self.context_frames = context_frames
        self.output_size = d_model

        self.input_proj = nn.Linear(num_features, d_model)

        # Sinusoidal positional encoding within a chunk: (chunk_size, 1, d_model)
        position = torch.arange(chunk_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(chunk_size, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, _ = inputs.shape
        x = self.input_proj(inputs)  # (T, N, d_model)
        d_model = x.shape[-1]

        # Pad time to a multiple of chunk_size, then fold chunks into the batch:
        # (T, N, d) -> (chunk_size, num_chunks * N, d)
        pad = (-T) % self.chunk_size
        if pad > 0:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
        num_chunks = (T + pad) // self.chunk_size
        x = x.reshape(num_chunks, self.chunk_size, N, d_model)
        x = x.permute(1, 0, 2, 3).reshape(self.chunk_size, num_chunks * N, d_model)
        x = x + self.pe

        # Mask the padded frames of the last chunk as attention keys
        key_padding_mask = None
        if pad > 0:
            frame_idx = torch.arange(T + pad, device=x.device).reshape(
                num_chunks, self.chunk_size
            )
            mask = (frame_idx >= T)[:, None, :].expand(num_chunks, N, self.chunk_size)
            key_padding_mask = mask.reshape(num_chunks * N, self.chunk_size)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Unfold chunks: (chunk_size, num_chunks * N, d) -> (T, N, d)
        x = x.reshape(self.chunk_size, num_chunks, N, d_model).permute(1, 0, 2, 3)
        x = x.reshape(num_chunks * self.chunk_size, N, d_model)[:T]
        return x[self.context_frames :]  # (T - context_frames, N, d_model)
