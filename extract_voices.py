import argparse
import os
import glob

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.model import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('--model-path', default="model.pt")
parser.add_argument('-d', '--device', default="cpu")

args = parser.parse_args()

device = torch.device(args.device)
model = Classifier()
model.load_state_dict(torch.load(args.model_path))
model.to(device)

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

counter = 0
paths = glob.glob(os.path.join(args.inputs, "*"))
for i, path in enumerate(paths):
    print(f"extracting from {path}...")
    wf, sr = torchaudio.load(path)
    wf = wf.to(device)
    parts = model.extract_voice_parts(wf, sr)
    print(f"extracted {len(parts)} parts.")
    for part in parts:
        part = part.cpu()
        torchaudio.save(os.path.join(args.outputs, f"{counter}.wav"), src=part, sample_rate=sr)
        counter += 1

