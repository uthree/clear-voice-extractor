import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from module.model import Classifier
from module.dataset import WaveFileDirectory

parser = argparse.ArgumentParser(description="train")

parser.add_argument('music_dir')
parser.add_argument('voice_dir')
parser.add_argument('-model-path', default='model.pt')
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-len', '--length', default=64000, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    model = Classifier().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    return model

def save_models(model):
    print("Saving models...")
    torch.save(model.state_dict(), args.model_path)
    print("Complete!")

device = torch.device(args.device)

music_ds = WaveFileDirectory(
        [args.music_dir],
        length=args.length,
        max_files=args.max_data
        )
voice_ds = WaveFileDirectory(
        [args.voice_dir],
        length=args.length,
        max_files=args.max_data
        )

music_dl = torch.utils.data.DataLoader(music_ds, batch_size=args.batch_size, shuffle=True)
voice_dl = torch.utils.data.DataLoader(voice_ds, batch_size=args.batch_size, shuffle=True)

step_count = 0

model = load_or_init_models(device)
Opt = optim.RAdam(model.parameters(), lr=1e-4)
BCE = nn.BCELoss().to(device)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=min(len(music_ds), len(voice_ds)))
    for batch, (music, voice) in enumerate(zip(music_dl, voice_dl)):
        N = music.shape[0]
        music = music.to(device)
        voice = voice.to(device)
        if music.shape != voice.shape:
            continue

        Opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logit_m = model(music)
            logit_v = model(voice)
            logit_mv = model(music + voice)

            zero = torch.zeros_like(logit_m)
            one = torch.ones_like(logit_v)

            loss = (BCE(logit_m, zero) + BCE(logit_mv, zero)) / 2 + BCE(logit_v, one)

        scaler.scale(loss).backward()
        scaler.step(Opt)

        scaler.update()

        step_count += 1

        tqdm.write(f"Step {step_count}, loss: {loss.item()}")

        bar.update(N)

        if batch % 500 == 0:
            save_models(model)

print("Training Complete!")
save_models(model)



