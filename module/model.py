import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, mlp_mul=3, dilation=1, norm=False, negative_slope=0.1):
        super().__init__()
        pad_size = int((kernel_size*dilation - dilation)/2)
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, pad_size, dilation=dilation)
        self.norm = ChannelNorm(channels) if norm else nn.Identity()
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.gelu(x)
        x = self.c3(x)
        return x + res


class Classifier(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=2048, internal_channels=512, dilations=[1, 3, 9, 27]):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = n_fft // 4
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate, self.n_fft, n_mels=128)
        
        self.input_layer = nn.Conv1d(128, internal_channels, 1)
        self.stack = nn.Sequential(*[ResBlock(internal_channels, dilation=d) for d in dilations])
        self.output_layer = nn.Conv1d(internal_channels, 1, 1)

    def forward(self, wave):
        wave = torch.cat([wave, torch.zeros(wave.shape[0], self.n_fft, device=wave.device)], dim=1)
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
                            extract_chunk=8000,
                            threshold=0.4,
                            min_length=32000):
        wave_16k = resample(wave, sample_rate, 16000)
        L = wave.shape[1]
        waves = wave_16k.split(classify_chunk, dim=1)
        labels = torch.cat([self.forward(c) for c in waves], dim=2)
        labels = F.interpolate(labels, L // (extract_chunk * 4))
        labels = F.interpolate(labels, L).squeeze(1)
        parts = []
        buffer = []
        for chunk, label in zip(wave.split(extract_chunk, dim=1), labels.split(extract_chunk, dim=1)):
            if chunk.abs().max() < 0.05:
                continue
            label = label.mean().item()
            flag = label >= threshold
            if flag == True:
                buffer.append(chunk)
            elif len(buffer) > 0 and flag == False:
                if len(buffer) > min_length // extract_chunk:
                    parts.append(torch.cat(buffer, dim=1))
                buffer = []
        if len(buffer) > min_length // extract_chunk:
            parts.append(torch.cat(buffer, dim=1))
        return parts
