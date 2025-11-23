import torch
import os
from clip_model import CLIP
from AudioMNIST_CLIPDataset import AudioMNIST_CLIPDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tokenizers import Tokenizer

bpe_tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

def tokenizer(text: str, max_seq_length: int = 32):
    encoded = bpe_tokenizer.encode(text)

    ids = encoded.ids[:max_seq_length]

    while len(ids) < max_seq_length:
        ids.append(0)  # pad with <pad> id (0)

    mask = [1 if x != 0 else 0 for x in ids]

    return torch.tensor(ids, dtype=torch.int32), torch.tensor(mask, dtype=torch.int32)


dataset = AudioMNIST_CLIPDataset("./Dataset")
all_audios = []
all_prompts = []

for i in range(len(dataset)):
    a, p = dataset[i]
    all_audios.append(a)
    all_prompts.append(p)

all_audios = torch.stack(all_audios).to(device)


B, C, F, T = all_audios.shape
audio_patches = all_audios.permute(0, 3, 1, 2).reshape(B, T, 256)


model = CLIP().to(device)
model.load_state_dict(torch.load("checkpoints/clip_audio_epoch250.pth", map_location=device))
model.eval()

print("\nIndexing Audio...\n")
with torch.no_grad():
    audio_features = model.audio_encoder(audio_patches)
    audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

while True:
    text = input("Enter prompt (or quit): ")
    if text.strip().lower() == "quit":
        break

    tok, mask1d = tokenizer(text, 32)
    tok = tok.unsqueeze(0).to(device).long()
    L = 32
    mask = mask1d.unsqueeze(0).unsqueeze(1).repeat(1, L, 1).to(device)

    with torch.no_grad():
        tfeat = model.text_encoder(tok, mask)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        sim = (audio_features @ tfeat.t()).squeeze()

    topk = torch.topk(sim, 5)
    print("\nTop 5 matches:")
    for score, idx in zip(topk.values.cpu(), topk.indices.cpu()):
        file_path, _ = dataset.samples[idx]
        print(f"{score:.4f} | {file_path}")
print()

