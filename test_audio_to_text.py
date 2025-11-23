import torch
import os
from clip_model import CLIP
from AudioMNIST_CLIPDataset import AudioMNIST_CLIPDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(text, max_seq_length=32):
    out = chr(2) + text + chr(3)
    if len(out) > max_seq_length:
        out = out[:max_seq_length]
    out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])
    out = torch.IntTensor(list(out.encode("utf-8")))
    mask_1d = torch.ones(len(out.nonzero()), dtype=torch.int32)
    mask_1d = torch.cat((mask_1d, torch.zeros(max_seq_length - len(mask_1d), dtype=torch.int32)))
    return out, mask_1d


dataset = AudioMNIST_CLIPDataset("./Dataset")

prompts = []
for i in range(len(dataset)):
    _, p = dataset[i]
    prompts.append(p)

tok_list = []
mask_list = []
for p in prompts:
    tok, m = tokenizer(p, 32)
    tok_list.append(tok)
    mask_list.append(m)

text_tokens = torch.stack(tok_list).long().to(device)
mask_1d = torch.stack(mask_list).to(device)
L = 32
mask = mask_1d.unsqueeze(1).repeat(1, L, 1).to(device)


model = CLIP().to(device)
model.load_state_dict(torch.load("checkpoints/clip_audio_epoch250.pth", map_location=device))
model.eval()

with torch.no_grad():
    text_features = model.text_encoder(text_tokens, mask)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

while True:
    idx_raw = input("\nEnter audio index (0–937) or quit: ")
    if idx_raw.lower() == "quit":
        break

    idx = int(idx_raw)
    audio, _ = dataset[idx]
    audio = audio.unsqueeze(0).to(device)

    B, C, F, T = audio.shape
    audio_patches = audio.permute(0, 3, 1, 2).reshape(1, T, 256)

    with torch.no_grad():
        afeat = model.audio_encoder(audio_patches)
        afeat = afeat / afeat.norm(dim=-1, keepdim=True)
        sim = (afeat @ text_features.t()).squeeze()

    topk = torch.topk(sim, 5)

    print("\nTop 5 text matches:")
    for score, i in zip(topk.values.cpu(), topk.indices.cpu()):
        print(f"{score:.4f} | {prompts[i]}")
    print()
