import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from AudioMNIST_CLIPDataset import AudioMNIST_CLIPDataset
from clip_model import CLIP



def tokenizer(text: str, encode: bool = True, max_seq_length: int = 32):
    if not encode:
        raise NotImplementedError("Decoding not implemented.")

    out = chr(2) + text + chr(3)

    if len(out) > max_seq_length:
        out = out[:max_seq_length]

    out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])
    out = torch.IntTensor(list(out.encode("utf-8")))

    mask_1d = torch.ones(len(out.nonzero()), dtype=torch.int32)
    mask_1d = torch.cat(
        (mask_1d,
         torch.zeros(max_seq_length - len(mask_1d), dtype=torch.int32))
    )

    return out, mask_1d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset = AudioMNIST_CLIPDataset(root="./Dataset")

train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)



emb_dim = 512           
vit_width = 256
n_patches = 44
patch_dim = 256
vit_layers = 6
vit_heads = 4

vocab_size = 256        
text_width = 512        
max_seq_length = 32
text_heads = 8          
text_layers = 8         



model = CLIP(
    emb_dim=emb_dim,
    vit_width=vit_width,
    n_patches=n_patches,
    patch_dim=patch_dim,
    vit_layers=vit_layers,
    vit_heads=vit_heads,
    vocab_size=vocab_size,
    text_width=text_width,
    max_seq_length=max_seq_length,
    text_heads=text_heads,
    text_layers=text_layers,
).to(device)



optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


os.makedirs("checkpoints", exist_ok=True)
num_epochs = 150

print("\nStarting upgraded CLIP training...\n")

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

    for audios, prompts in pbar:
        audios = audios.to(device)

        B, C, F, T = audios.shape
        assert C == 2 and F == 128 and T == n_patches

        audio_patches = audios.permute(0, 3, 1, 2).reshape(B, T, 256)

        token_list = []
        mask_list = []

        for s in prompts:
            tok, mask1d = tokenizer(s, max_seq_length=max_seq_length)
            token_list.append(tok)
            mask_list.append(mask1d)

        text_tokens = torch.stack(token_list).long().to(device)
        mask_1d = torch.stack(mask_list).to(device)
        L = max_seq_length
        mask = mask_1d.unsqueeze(1).repeat(1, L, 1)

        optimizer.zero_grad()
        loss = model(audio_patches, text_tokens, mask)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch} | Avg Loss = {avg_loss:.4f}\n")

    ckpt_path = f"checkpoints/clip_audio_epoch{epoch}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint at: {ckpt_path}\n")

print("Training completed successfully!")
