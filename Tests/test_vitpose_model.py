import pytest
import torch
from src.annotator_dashboard.vitpose_model import (
    PatchEmbed,
    Attention,
    FFN,
    Block,
    VisionTransformer,
    TopdownHeatmapSimpleHead,
    ViTPose,
)

def test_patch_embed_forward():
    batch_size = 2
    embed = PatchEmbed(img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=384)
    x = torch.randn(batch_size, 3, 256, 192)
    out = embed(x)
    
    expected_patches = (256 // 16) * (192 // 16)  # 16 * 12 = 192 patches
    assert out.shape == (batch_size, expected_patches, 384)

def test_attention_forward():
    batch_size = 2
    seq_len = 192
    embed_dim = 384
    attn = Attention(dim=embed_dim, num_heads=12)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = attn(x)
    assert out.shape == (batch_size, seq_len, embed_dim)

def test_ffn_forward():
    batch_size = 2
    seq_len = 192
    embed_dim = 384
    ffn = FFN(dim=embed_dim, hidden_dim=embed_dim * 4)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = ffn(x)
    assert out.shape == (batch_size, seq_len, embed_dim)

def test_block_forward():
    batch_size = 2
    seq_len = 192
    embed_dim = 384
    block = Block(dim=embed_dim, num_heads=12)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = block(x)
    assert out.shape == (batch_size, seq_len, embed_dim)

def test_vision_transformer_forward():
    batch_size = 1
    vit = VisionTransformer(img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=384, depth=2, num_heads=4)
    x = torch.randn(batch_size, 3, 256, 192)
    out = vit(x)
    assert out.shape == (batch_size, 192, 384)

def test_heatmap_head_forward():
    batch_size = 2
    head = TopdownHeatmapSimpleHead(in_channels=384, out_channels=17)
    x = torch.randn(batch_size, 192, 384)
    out = head(x)
    # Output heatmap shape after 2 deconv layers (stride 2 each): 16x12 -> 32x24 -> 64x48
    assert out.shape == (batch_size, 17, 64, 48)

def test_vitpose_full_model_forward():
    batch_size = 1
    model = ViTPose(img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=384, depth=2, num_heads=4, out_channels=17)
    model.eval()
    with torch.no_grad():
        x = torch.randn(batch_size, 3, 256, 192)
        heatmaps = model(x)
    assert heatmaps.shape == (batch_size, 17, 64, 48)
