# Unified Multimodal Models List

## Table of Contents
- [Diffusion Model](#diffusion-model)
- [Autoregressive Model](#autoregressive-model)
  - [Type b-1: Single Token Space](#type-b-1-single-token-space)
  - [Type b-2: Separate Encoders with Diffusion](#type-b-2-separate-encoders-with-diffusion)
  - [Type b-3: Learnable Query](#type-b-3-learnable-query)
  - [Type b-4: Dual Path (Janus-like)](#type-b-4-dual-path-janus-like)
  - [Type b-5: Hybrid Approaches](#type-b-5-hybrid-approaches)
- [Fused Autoregressive and Diffusion Model](#fused-autoregressive-and-diffusion-model)

---

## Diffusion Model

| Model | Type | Architecture | Date | Backbone | Und. Enc. | Gen. Enc. | Gen. Dec. | Mask |
|-------|------|--------------|------|----------|-----------|-----------|-----------|------|
| Dual Diffusion [127] | a | D-DiT | 2024-12 | - | SD-VAE | SD-VAE | Bidirect. | ✓ |
| UniDisc [128] | a | DiT | 2025-03 | - | MAGVIT-v2 | MAGVIT-v2 | Bidirect. | ✓ |
| MMaDA [129] | a | LLaDA | 2025-05 | - | MAGVIT-v2 | MAGVIT-v2 | Bidirect. | ✓ |
| FUDOKI [130] | a | DeepSeek-LLM | 2025-05 | - | SigLIP VQGAN | VQGAN | Bidirect. | ✓ |
| Muddit [131] | a | Meissonic (MM-DiT) | 2025-05 | - | VQGAN | VQGAN | Bidirect. | ✓ |

---

## Autoregressive Model

### Type b-1: Single Token Space

| Model | Type | Architecture | Date | Backbone | Und. Enc. | Gen. Enc. | Gen. Dec. | Mask |
|-------|------|--------------|------|----------|-----------|-----------|-----------|------|
| LWM [29] | b-1 | LLaMa-2 | 2024-02 | LLaMa-2 | VQGAN | VQGAN | Causal | ✓ |
| Chameleon [30] | b-1 | LLaMa-2 | 2024-05 | LLaMa-2 | VQ-IMG | VQ-IMG | Causal | ✓ |
| ANOLE [132] | b-1 | LLaMa-2 | 2024-07 | LLaMa-2 | VQ-IMG | VQ-IMG | Causal | ✓ |
| Emu3 [133] | b-1 | LLaMA-2 | 2024-09 | LLaMA-2 | SBER-MoVQGAN | SBER-MoVQGAN | Causal | ✓ |
| MMAR [134] | b-1 | Qwen2 | 2024-10 | Qwen2 | SD-VAE + EmbeddingViT | Diffusion MLP | Bidirect. | ✓ |
| Orthus [135] | b-1 | Chameleon | 2024-11 | Chameleon | VQ-IMG+Vision embed. | Diffusion MLP | Causal | ✓ |
| SynerGen-VL [136] | b-1 | InterLM2 | 2024-12 | InterLM2 | SBER-MoVQGAN | SBER-MoVQGAN | Causal | ✓ |
| Liquid [137] | b-1 | GEMMA | 2024-12 | GEMMA | VQGAN | VQGAN | Causal | ✓ |
| UGen [138] | b-1 | TinyLlama | 2025-03 | TinyLlama | SBER-MoVQGAN | SBER-MoVQGAN | Causal | ✓ |
| Harmon [139] | b-1 | Qwen2.5 | 2025-03 | Qwen2.5 | MAR | MAR | Bidirect. | ✓ |
| TokLIP [140] | b-1 | Qwen2.5 | 2025-05 | Qwen2.5 | VQGAN+SigLIP | VQGAN | Causal | ✓ |
| Selftok [141] | b-1 | LLaMA3.1 | 2025-05 | LLaMA3.1 | SD3-VAE+MMDiT | SD3 | Causal | ✓ |

### Type b-2: Separate Encoders with Diffusion

| Model | Type | Architecture | Date | Backbone | Und. Enc. | Gen. Enc. | Gen. Dec. | Mask |
|-------|------|--------------|------|----------|-----------|-----------|-----------|------|
| Emu [142] | b-2 | LLaMA | 2023-07 | LLaMA | EVA-CLIP | SD | Causal | ✓ |
| LaVIT [143] | b-2 | LLaMA | 2023-09 | LLaMA | EVA-CLIP | SD-1.5 | Causal | ✓ |
| DreamLLM [34] | b-2 | LLaMA | 2023-09 | LLaMA | OpenAI-CLIP | SD-2.1 | Causal | ✓ |
| Emu2 [33] | b-2 | LLaMA | 2023-12 | LLaMA | EVA-CLIP | SDXL | Causal | ✓ |
| VL-GPT [35] | b-2 | LLaMA | 2023-12 | LLaMA | OpenAI-CLIP | IP-Adapter | Causal | ✓ |
| MM-Interleaved [144] | b-2 | Vicuna | 2024-01 | Vicuna | OpenAI-CLIP | SD-v2.1 | Causal | ✓ |
| Mini-Gemini [145] | b-2 | Gemma&Vicuna | 2024-03 | Gemma&Vicuna | OpenAI-CLIP+ConvNext | SDXL | Causal | ✓ |
| VILA-U [146] | b-2 | LLaMA-2 | 2024-09 | LLaMA-2 | SigLIP+RQ | RQ-VAE | Causal | ✓ |
| PUMA [147] | b-2 | LLaMA-3 | 2024-10 | LLaMA-3 | OpenAI-CLIP | SDXL | Bidirect. | ✓ |
| MetaMorph [148] | b-2 | LLaMA | 2024-12 | LLaMA | SigLIP | SD-1.5 | Causal | ✓ |
| ILLUME [149] | b-2 | Vicuna | 2024-12 | Vicuna | UNIT | SDXL | Causal | ✓ |
| UniTok [150] | b-2 | LLaMa-2 | 2025-02 | LLaMa-2 | ViTamin | ViTamin | Causal | ✓ |
| QLIP [151] | b-2 | LlaMa-3 | 2025-02 | LlaMa-3 | QLIP-ViT+BSQ | BSQ-AE | Causal | ✓ |
| DualToken [152] | b-2 | Qwen2.5 | 2025-03 | Qwen2.5 | SigLIP | RQVAE | Causal | ✓ |
| UniFork [153] | b-2 | Qwen2.5 | 2025-06 | Qwen2.5 | SigLIP+RQ | RQ-VAE | Causal | ✓ |
| UniCode2 [154] | b-2 | Qwen2.5 | 2025-06 | Qwen2.5 | SigLIP+RQ | FLUX.1-dev / SD-1.5 | Causal | ✓ |
| UniWorld [155] | b-2 | Qwen2.5-VL | 2025-06 | Qwen2.5-VL | SigLIP2 | DiT | Bidrect. | ✓ |
| Pisces [156] | b-2 | LLaMA-3.1 | 2025-06 | LLaMA-3.1 | SigLIP EVA-CLIP | Diffusion | Causal | ✓ |
| Tar [157] | b-2 | Qwen2.5 | 2025-06 | Qwen2.5 | SigLIP2+VQ | VQGAN / SANA | Causal | ✓ |
| OmniGen2 [158] | b-2 | Qwen2.5-VL | 2025-06 | Qwen2.5-VL | SigLIP | OmniGen | Causal | ✓ |
| Ovis-U1 [159] | b-2 | Ovis | 2025-06 | Ovis | AimV2 | MMDiT | Causal | ✓ |
| X-Omni [160] | b-2 | Qwen2.5-VL | 2025-07 | Qwen2.5-VL | QwenViT Siglip | FLUX | Causal | ✓ |
| Qwen-Image [161] | b-2 | Qwen2.5-VL | 2025-08 | Qwen2.5-VL | QwenViT | MMDiT | Causal | ✓ |
| Bifrost-1 [162] | b-2 | Qwen2.5-VL | 2025-08 | Qwen2.5-VL | QwenViT ViT | FLUX | Causal | ✓ |

### Type b-3: Learnable Query

| Model | Type | Architecture | Date | Backbone | Und. Enc. | Gen. Enc. | Gen. Dec. | Mask |
|-------|------|--------------|------|----------|-----------|-----------|-----------|------|
| SEED [163] | b-3 | OPT | 2023-07 | OPT | SEED Tokenizer | Learnable Query | SD | Causal | ✓ |
| SEED-LLaMA [164] | b-3 | LLaMa-2 &Vicuna | 2023-10 | LLaMa-2 &Vicuna | SEED Tokenizer | Learnable Query | unCLIP-SD | Causal | ✓ |
| SEED-X [165] | b-3 | LLaMa-2 | 2024-04 | LLaMa-2 | SEED Tokenizer | Learnable Query | SDXL | Causal | ✓ |
| MetaQueries [166] | b-3 | LLaVA&Qwen2.5-VL | 2025-04 | LLaVA&Qwen2.5-VL | SigLIP | Learnable Query | Sana | Causal | ✓ |
| Nexus-Gen [167] | b-3 | Qwen2.5-VL | 2025-04 | Qwen2.5-VL | QwenViT | Learnable Query | FLUX | Causal | ✓ |
| Ming-Lite-Uni [168] | b-3 | M2-omni | 2025-05 | M2-omni | NaViT | Learnable Query | Sana | Causal | ✓ |
| BLIP3-o [169] | b-3 | Qwen2.5-VL | 2025-05 | Qwen2.5-VL | OpenAI-CLIP | Learnable Query | Lumina-Next | Causal | ✓ |
| OpenUni [170] | b-3 | InternVL3 | 2025-05 | InternVL3 | InternViT | Learnable Query | Sana | Causal | ✓ |
| Ming-Omni [171] | b-3 | Ling | 2025-06 | Ling | QwenViT | Learnable Query | Multi-scale DiT | Causal | ✓ |
| UniLIP [172] | b-3 | InternVL3 | 2025-07 | InternVL3 | InternViT | Learnable Query | Sana | Causal | ✓ |
| TBAC-UniImage [173] | b-3 | Qwen2.5-VL | 2025-08 | Qwen2.5-VL | QwenViT | Learnable Query | Sana | Causal | ✓ |

### Type b-4: Dual Path (Janus-like)

| Model | Type | Architecture | Date | Backbone | Und. Enc. | Gen. Enc. | Gen. Dec. | Mask |
|-------|------|--------------|------|----------|-----------|-----------|-----------|------|
| **Janus [174]** | **b-4** | **DeepSeek-LLM** | **2024-10** | **DeepSeek-LLM** | **SigLIP** | **VQGAN** | **VQGAN** | **Casual** |
| **Janus-Pro [175]** | **b-4** | **DeepSeek-LLM** | **2025-01** | **DeepSeek-LLM** | **SigLIP** | **VQGAN** | **VQGAN** | **Casual** |
| OmniMamba [176] | b-4 | Mamba-2 | 2025-03 | Mamba-2 | DINO-v2+SigLIP | VQGAN | VQGAN | Causal |
| Unifluid [177] | b-4 | Gemma-2 | 2025-03 | Gemma-2 | SigLIP | SD-VAE | Diffusion MLP | Causal |
| MindOmni [178] | b-4 | Qwen2.5-VL | 2025-06 | Qwen2.5-VL | QwenViT | VAE | OmniGen | Causal |
| Skywork UniPic [179] | b-4 | Qwen2.5 | 2025-08 | Qwen2.5 | SigLIP2 | SDXL-VAE | SDXL-VAE | Causal |

### Type b-5: Hybrid Approaches

| Model | Type | Architecture | Date | Backbone | Und. Enc. | Gen. Enc. | Gen. Dec. | Mask |
|-------|------|--------------|------|----------|-----------|-----------|-----------|------|
| MUSE-VL [180] | b-5 | Qwen-2.5&Yi-1.5 | 2024-11 | Qwen-2.5&Yi-1.5 | SigLIP | VQGAN | VQGAN | Causal |
| Tokenflow [181] | b-5 | Vicuna&Qwen-2.5 | 2024-12 | Vicuna&Qwen-2.5 | OpenAI-CLIP | MSVQ | MSVQ | Causal |
| VARGPT [182] | b-5 | Vicuna-1.5 | 2025-01 | Vicuna-1.5 | OpenAI-CLIP | MSVQ | VAR-d30 | Causal |
| SemHiTok [183] | b-5 | Qwen2.5 | 2025-03 | Qwen2.5 | SigLIP ViT | ViT | Causal | ✓ |
| VARGPT-1.1 [184] | b-5 | Qwen2 | 2025-04 | Qwen2 | SigLIP | MSVQ | Infinity | Causal |
| ILLUME+ [185] | b-5 | Qwen2.5 | 2025-04 | Qwen2.5 | QwenViT | MoVQGAN | SDXL | Causal |
| UniToken [186] | b-5 | Chameleon | 2025-04 | Chameleon | SigLIP | VQ-IMG | VQGAN | Causal |
| Show-o2 [187] | b-5 | Qwen2.5 | 2025-06 | Qwen2.5 | Wan-3DVAE + SigLIP | Wan-3DVAE | Wan-3DVAE | Causal |

---

## Fused Autoregressive and Diffusion Model

| Model | Type | Architecture | Date | Backbone | Und. Enc. | Gen. Enc. | Gen. Dec. | Mask |
|-------|------|--------------|------|----------|-----------|-----------|-----------|------|
| Transfusion [38] | c-1 | LLaMA-2 | 2024-08 | LLaMA-2 | SD-VAE | SD-VAE | Bidirect. | ✓ |
| Show-o [39] | c-1 | LLaVA-v1.5-Phi | 2024-08 | LLaVA-v1.5-Phi | MAGVIT-v2 | MAGVIT-v2 | Bidirect. | ✓ |
| MonoFormer [37] | c-1 | TinyLLaMA | 2024-09 | TinyLLaMA | SD-VAE | SD-VAE | Bidirect. | ✓ |
| LMFusion [188] | c-1 | LLaMA | 2024-12 | LLaMA | SD-VAE+UNet down. | SD-VAE+UNet up. | Bidirect. | ✓ |
| Janus-flow [189] | c-2 | DeepSeek-LLM | 2024-11 | DeepSeek-LLM | SigLIP | SDXL-VAE | SDXL-VAE | Causal |
| Mogao [190] | c-2 | Qwen2.5 | 2025-05 | Qwen2.5 | SigLIP+SDXL-VAE | SDXL-VAE | SDXL-VAE | Bidirect. |
| BAGEL [191] | c-2 | Qwen2.5 | 2025-05 | Qwen2.5 | SigLIP | FLUX-VAE | FLUX-VAE | Bidirect. |

---

## Notes

- **Type a**: Pure Diffusion Models
- **Type b-1**: Single unified token space (true unified models like Chameleon)
- **Type b-2**: Separate encoders with diffusion-based generation
- **Type b-3**: Learnable query-based approaches
- **Type b-4**: Dual-path architecture (Janus-like) - LLM only shared
- **Type b-5**: Hybrid approaches with various tokenization strategies
- **Type c-1**: Fused autoregressive and diffusion (single token space)
- **Type c-2**: Fused autoregressive and diffusion (dual path)

**Janus** and **Janus-Pro** are highlighted as they represent the dual-path architecture analyzed in ARCHITECTURE_ANALYSIS.md.
