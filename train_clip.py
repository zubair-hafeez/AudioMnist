import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from AudioMNIST_CLIPDataset import AudioMNIST_CLIPDataset
from clip_model import CLIPModel


def simple_tokenizer(prompts, word2idx):
    tokenized = []
    for sentence in prompts:
        tokens = [word2idx.get(word.lower(), 0) for word in sentence.split()]
        tokenized.append(torch.tensor(tokens))
    # Pad to equal length
    max_len = max(len(t) for t in tokenized)
    padded = torch.zeros(len(tokenized), max_len, dtype=torch.long)
    for i, t in enumerate(tokenized):
        padded[i, :len(t)] = t
    return padded



def clip_loss(audio_features, text_features, temperature):
    logits = (audio_features @ text_features.T) * torch.exp(temperature)
    labels = torch.arange(len(logits), device=logits.device)
    loss_a = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_a + loss_t) / 2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = AudioMNIST_CLIPDataset(root="./Dataset")
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = CLIPModel(embed_dim=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
vocab = {}
for _, prompt in dataset.samples:
    for w in prompt.lower().split():
        if w not in vocab:
            vocab[w] = len(vocab) + 1


print("Starting CLIP training...")
model.train()
for audios, prompts in tqdm(train_loader, desc="Training"):
    audios = audios.to(device)
    tokens = simple_tokenizer(prompts, vocab).to(device)

    optimizer.zero_grad()
    audio_features, text_features = model(audios, tokens)
    loss = clip_loss(audio_features, text_features, model.temperature)
    loss.backward()
    optimizer.step()

print("Training completed successfully!")


torch.save(model.state_dict(), "clip_audio_epoch1.pth")
print("Model saved as clip_audio_epoch1.pth")
