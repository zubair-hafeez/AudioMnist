"""
from torchinfo import summary
import torch
from clip_model import CLIP

model = CLIP().to("cpu")

audio_patches = torch.randn(1, 44, 256)

summary(model.audio_encoder, input_data=audio_patches, depth=3)
"""

from torchinfo import summary
import torch
from clip_model import CLIP

model = CLIP().to("cpu")

audio_patches = torch.randn(1, 44, 256)        # (B, n_patches, patch_dim)
text = torch.randint(0, 255, (1, 32))          # (B, L)
mask = torch.ones((1, 32, 32), dtype=torch.long)   # <-- FIX

summary(
    model,
    input_data=(audio_patches, text, mask),
    depth=3,
    device="cpu"
)
