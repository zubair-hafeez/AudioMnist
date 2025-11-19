import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from AudioMNIST_CLIPDataset import AudioMNIST_CLIPDataset
from clip_model import CLIP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/clip_audio_epoch150.pth"



def tokenizer(text: str, max_len=32):
    txt = chr(2) + text + chr(3)

    if len(txt) > max_len:
        txt = txt[:max_len]

    txt = txt + "".join([chr(0) for _ in range(max_len - len(txt))])

    tokens = torch.tensor(list(txt.encode("utf-8")), dtype=torch.long)

    mask_1d = torch.ones(len(tokens.nonzero()), dtype=torch.int32)
    mask_1d = torch.cat(
        (mask_1d,
         torch.zeros(max_len - len(mask_1d), dtype=torch.int32))
    )

    return tokens, mask_1d



def load_model():
    model = CLIP(
        emb_dim=512,          
        vit_width=256,
        n_patches=44,
        patch_dim=256,
        vit_layers=6,
        vit_heads=4,
        vocab_size=256,
        text_width=512,       
        max_seq_length=32,
        text_heads=8,         
        text_layers=8         
    )

    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def build_audio_index(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_embs = []
    all_prompts = []
    all_paths = []

    with torch.no_grad():
        for batch_audio, batch_prompts in tqdm(loader, desc="Indexing Audio"):
            B = batch_audio.shape[0]

            patches = batch_audio.permute(0, 3, 1, 2).reshape(B, 44, 256).to(DEVICE)

            emb = model.audio_encoder(patches)
            emb = F.normalize(emb, dim=-1)

            all_embs.append(emb.cpu())
            all_prompts.extend(batch_prompts)

            for _ in range(B):
                all_paths.append(dataset.samples[len(all_paths)][0])

    return torch.cat(all_embs, dim=0), all_prompts, all_paths


def retrieve_audio(model, audio_embs, paths, prompts):
    while True:
        query = input("\nEnter prompt (or quit): ").strip()
        if query.lower() == "quit":
            break

        tokens, m1 = tokenizer(query)
        tokens = tokens.unsqueeze(0).to(DEVICE)
        m1 = m1.to(DEVICE)

        L = 32
        mask = m1.unsqueeze(0).unsqueeze(1).repeat(1, L, 1)

        with torch.no_grad():
            txt_emb = model.text_encoder(tokens, mask=mask)
            txt_emb = F.normalize(txt_emb, dim=-1)

            sims = txt_emb @ audio_embs.T
            sims = sims.squeeze(0)

        vals, idx = torch.topk(sims, k=5)

        print("\nTop 5 matches:")
        for v, j in zip(vals, idx):
            print(f"{v:.4f} | {paths[j]} | {prompts[j]}")



def main():
    dataset = AudioMNIST_CLIPDataset(root="./Dataset")
    model = load_model()

    audio_embs, prompts, paths = build_audio_index(model, dataset)
    audio_embs = audio_embs.to(DEVICE)

    retrieve_audio(model, audio_embs, paths, prompts)


if __name__ == "__main__":
    main()
