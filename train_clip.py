import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from AudioMNIST_CLIPDataset import AudioMNIST_CLIPDataset
from clip_model import CLIP

from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def train_bpe_tokenizer(dataset_root="./Dataset",
                        tokenizer_path="bpe_tokenizer.json",
                        vocab_size=3000):

    print("\n===== TRAINING BPE TOKENIZER =====")

    dataset = AudioMNIST_CLIPDataset(root=dataset_root)
    all_prompts = []

    for i in range(len(dataset)):
        _, prompt = dataset[i]
        all_prompts.append(prompt)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>"]
    )

    tokenizer.train_from_iterator(all_prompts, trainer=trainer)
    tokenizer.save(tokenizer_path)

    print(f"Saved tokenizer to {tokenizer_path}")
    return tokenizer


def train_clip_model(tokenizer_path="bpe_tokenizer.json",
                     dataset_root="./Dataset",
                     num_epochs=100):

    print("\n===== LOADING TOKENIZER =====")
    bpe_tokenizer = Tokenizer.from_file(tokenizer_path)

    def tokenizer_fn(text: str, max_seq_length: int = 32):
        encoded = bpe_tokenizer.encode(text)
        ids = encoded.ids[:max_seq_length]
        while len(ids) < max_seq_length:
            ids.append(0)
        mask = [1 if x != 0 else 0 for x in ids]
        return torch.tensor(ids, dtype=torch.int32), torch.tensor(mask, dtype=torch.int32)

    dataset = AudioMNIST_CLIPDataset(root=dataset_root)
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    emb_dim = 512
    vit_width = 512
    n_patches = 44
    patch_dim = 256
    vit_layers = 8
    vit_heads = 8

    vocab_size = bpe_tokenizer.get_vocab_size()
    text_width = 512
    max_seq_length = 32
    text_heads = 8
    text_layers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        text_layers=text_layers
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for audios, prompts in pbar:
            audios = audios.to(device)
            B, C, F, T = audios.shape

            audio_patches = audios.permute(0, 3, 1, 2).reshape(B, T, 256)

            token_list = []
            mask_list = []

            for s in prompts:
                tok, mask1d = tokenizer_fn(s, max_seq_length=max_seq_length)
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
        ckpt_path = f"checkpoints/clip_audio_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("Training completed successfully!")


if __name__ == "__main__":
    train_bpe_tokenizer()
    train_clip_model()
