import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, mlp_mul=3, norm=False, negative_slope=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, kernel_size//2)
        self.norm = ChannelNorm(channels) if norm else nn.Identity()
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)
        self.negative_slope = negative_slope

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.c3(x)
        return x + res


class Classifier(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=2048, internal_channels=256, num_layers=4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = n_fft // 4
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate, self.n_fft, n_mels=128)
        
        self.input_layer = nn.Conv1d(128, internal_channels, 1)
        self.stack = nn.Sequential(*[ResBlock(internal_channels) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, 1, 1)

    def forward(self, wave):
        x = self.mel(wave)
        x = self.input_layer(x)
        x = self.stack(x)
        x = self.output_layer(x)
        x = F.sigmoid(x)
        return x

    @torch.inference_mode()
    def extract_voice_parts(self,
                            wave,
                            sample_rate,
                            classify_chunk=16000,
                            extract_chunk=1024,
                            threshold=0.75):
        wave_16k = resample(wave, sample_rate, 16000)
        L = wave.shape[1]
        waves = wave_16k.split(classify_chunk, dim=1)
        labels = torch.cat([self.forward(c) for c in waves], dim=2)
        labels = F.interpolate(labels, L).squeeze(1)
        parts = []
        buffer = []
        for chunk, label in zip(wave.split(extract_chunk, dim=1), labels.split(extract_chunk, dim=1)):
            label = label.mean().item()
            flag = label >= threshold
            if flag == True:
                buffer.append(chunk)
            elif len(buffer) > 0 and flag == False:
                parts.append(torch.cat(buffer, dim=1))
                buffer = []
        if len(buffer) > 0:
            parts.append(torch.cat(buffer, dim=1))
        return parts
