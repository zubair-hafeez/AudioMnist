import torch
import torch.nn.functional as F
import numpy as np
import librosa
import scipy.signal
from tqdm import tqdm

from AudioMNIST_CLIPDataset import AudioMNIST_CLIPDataset
from clip_model import CLIP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/clip_audio_epoch150.pth"
TARGET_SR = 12000
NFFT = 512


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



def make_patches_from_wav(path):
    waveform, _ = librosa.load(path, sr=TARGET_SR)

    f, t, Z = scipy.signal.stft(
        waveform,
        TARGET_SR,
        nperseg=NFFT//2,
        noverlap=NFFT//4,
        window="hann"
    )

    Z = Z[0:128, :-1]
    amp = np.abs(Z)
    phase = np.angle(Z)

    eps = 1e-10
    norm = (amp - amp.min()) / (amp.max() - amp.min() + eps)
    Zc = norm * np.exp(1j * phase)

    real = np.real(Zc)
    imag = np.imag(Zc)
    spec = np.stack([real, imag], axis=0)

    if spec.shape[-1] < 44:
        spec = np.pad(spec, ((0,0),(0,0),(0,44 - spec.shape[-1])))
    else:
        spec = spec[:, :, :44]

    patches = torch.tensor(spec, dtype=torch.float32)
    patches = patches.permute(2, 0, 1).reshape(44, 256)

    return patches.unsqueeze(0)  # shape (1, 44, 256)


def build_text_index(model, dataset):
    text_embs = []
    prompts = []

    with torch.no_grad():
        for _, prompt in tqdm(dataset, desc="Indexing Text"):
            tok, m1 = tokenizer(prompt)
            tok = tok.unsqueeze(0).to(DEVICE)
            m1 = m1.to(DEVICE)

            L = 32
            mask = m1.unsqueeze(0).unsqueeze(1).repeat(1, L, 1)

            emb = model.text_encoder(tok, mask)
            emb = F.normalize(emb, dim=-1)

            text_embs.append(emb.cpu())
            prompts.append(prompt)

    return torch.cat(text_embs, dim=0), prompts



def retrieve_text(model, text_embs, prompts):
    while True:
        path = input("\nEnter .wav file path (or quit): ")
        if path.lower() == "quit":
            break

        patches = make_patches_from_wav(path).to(DEVICE)

        with torch.no_grad():
            audio_emb = model.audio_encoder(patches)
            audio_emb = F.normalize(audio_emb, dim=-1)

            sims = audio_emb @ text_embs.T
            sims = sims.squeeze(0)

        vals, idx = torch.topk(sims, k=5)

        print("\nTop 5 matching prompts:")
        for v, j in zip(vals, idx):
            print(f"{v:.4f} | {prompts[j]}")



def main():
    dataset = AudioMNIST_CLIPDataset(root="./Dataset")
    model = load_model()

    text_embs, prompts = build_text_index(model, dataset)
    text_embs = text_embs.to(DEVICE)

    retrieve_text(model, text_embs, prompts)


if __name__ == "__main__":
    main()
