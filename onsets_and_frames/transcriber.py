"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from x_transformers import Encoder

from .lstm import BiLSTM
from .mel import melspectrogram


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

class SinusPosEncoding(nn.Module):
  def __init__(self, emb_size, max_t):
    super().__init__()
    self.emb_size =emb_size
    self.max_t = max_t
    self.register_buffer('encoding', self._prepare_emb())

  def _prepare_emb(self):
    dim_axis = 10000**(torch.arange(self.emb_size//2) * 2 / self.emb_size) # 10000 ** (normalized values between 0~1 num_emb_dim)
    timesteps = torch.arange(self.max_t)
    pos_enc_in = timesteps.unsqueeze(1) / dim_axis.unsqueeze(0)
    pos_enc_sin = torch.sin(pos_enc_in) # x values for sin are between 0 ~ 1 so the values could never be the same
    pos_enc_cos = torch.cos(pos_enc_in)

    pos_enc = torch.stack([pos_enc_sin, pos_enc_cos], dim=-1).reshape([self.max_t, self.emb_size])
    return pos_enc

  def forward(self, x):
    x_t = x.shape[1]
    return x + self.encoding[:x_t, ...]

class OnsetsAndFramesTF(OnsetsAndFrames):
    def __init__(self, input_features, output_features, depth=2, heads=16, attn_dropout=0.1, ff_dropout=0.1, model_complexity=64):
        super().__init__(input_features, output_features, model_complexity)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, depth, heads: Encoder(dim=input_size, depth=depth, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout, attn_flash=True)
        
        self.conv = ConvStack(input_features, model_size)
        self.enc_pos_enc = SinusPosEncoding(model_size, 3000)
        self.transformer = sequence_model(model_size, depth, heads)
        self.fc = nn.Linear(model_size, output_features * 4)
        
        del self.onset_stack, self.offset_stack, self.frame_stack, self.combined_stack, self.velocity_stack

    def forward(self, mel):
        x = self.conv(mel)
        x = self.enc_pos_enc(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = x.reshape(-1, x.size(-1) // 4, 4)
        onset_pred, offset_pred, frame_pred, velocity_pred = x.split(1, dim=-1)
        onset_pred = onset_pred.squeeze(-1)
        offset_pred = offset_pred.squeeze(-1)
        velocity_pred = velocity_pred.squeeze(-1)
        frame_pred = frame_pred.squeeze(-1)
        
        onset_pred = onset_pred.sigmoid()
        offset_pred = offset_pred.sigmoid()
        frame_pred = frame_pred.sigmoid()
        
        return onset_pred, offset_pred, frame_pred, frame_pred, velocity_pred