# UniToken - Type B5 Hybrid (Discrete + Continuous)

## 목차
1. [개요](#개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [Hybrid 토큰 시스템](#hybrid-토큰-시스템)
4. [BOI 토큰과 이미지 생성](#boi-토큰과-이미지-생성)
5. [MultiModalLogitsProcessor](#multimodallogitsprocessor)
6. [Interleaved 생성 가능성](#interleaved-생성-가능성)
7. [핵심 특징](#핵심-특징)
8. [결론](#결론)

---

## 개요

**UniToken**은 Fudan University와 Meituan이 개발한 "Harmonizing Multimodal Understanding and Generation through Unified Visual Encoding" 모델로, **Discrete와 Continuous 표현을 동시에 사용**하는 Type B5 (Hybrid) 모델입니다.

**핵심 특징**:
- ✅ Hybrid visual encoding: Discrete (VQ-VAE) + Continuous (SigLIP)
- ✅ 단일 모델에서 이해와 생성 통합
- ⚠️ BOI 토큰 수동 삽입 (이미지 생성 시)
- ⚠️ 제한적 Interleaved 생성
- ✅ Any-resolution 지원

**저장소**: `d:\Check_\janus\analysis\repos\type_b5_hybrid\UniToken`

**논문**: [UniToken: Harmonizing Multimodal Understanding and Generation through Unified Visual Encoding](https://arxiv.org/abs/2504.04423)

**모델 크기**: 7B (기반 LLM: Lumina-mGPT)

**CVPR 2025 Workshop 채택**

---

## 아키텍처 분석

### 1. Type B5 - Hybrid 방식

**특징**:
- 이미지를 **두 가지 방식**으로 표현
  - **Discrete tokens**: 생성 시 사용 (Chameleon VQ-VAE)
  - **Continuous tokens**: 이해 시 사용 (SigLIP + Adapter)
- 두 표현을 상황에 따라 선택적 사용

**비교표**:

| 모델 | 이미지 표현 | 생성 방식 | 이해 방식 |
|------|-----------|----------|----------|
| Chameleon | Discrete only | VQ tokens | VQ tokens |
| Janus | Discrete (separate) | VQ tokens | CLIP features |
| SEED-X | Learnable Query | Resampler → SDXL | Resampler |
| **UniToken** | **Hybrid** | **VQ tokens** | **SigLIP features** |

### 2. 전체 구조

**파일**: `unitoken/model/modeling_xllmx_chameleon_anyres.py`

```
Image Input
    ↓
┌─────────────────────────────────────┐
│  Understanding Path (Continuous)    │
│  ├─ SigLIP ViT                     │
│  ├─ Adapter (Linear + GELU)        │
│  └─ Continuous Tokens (729)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Generation Path (Discrete)         │
│  ├─ Chameleon VQ-VAE               │
│  └─ Discrete Tokens (32x32=1024)   │
└─────────────────────────────────────┘
    ↓
Unified Sequence: <SOI> [Discrete] <SEP> [Continuous] <EOI>
    ↓
Lumina-mGPT (7B Transformer)
    ↓
Autoregressive Generation
```

### 3. ChameleonXLLMXForConditionalGenerationAnyRes

**라인 57-81**: 모델 초기화

```python
class ChameleonXLLMXForConditionalGenerationAnyRes(ChameleonForConditionalGeneration):
    config_class = ChameleonXLLMXConfig

    def __init__(self, config):
        super().__init__(config)
        self._init_proj()   # Adapter 초기화
        self._init_vit()    # SigLIP ViT 초기화
        self.image_grid_pinpoints = [
            [384, 768],
            [768, 384],
            [768, 768],
            [384, 1152],
            [1152, 384]
        ]

    def _init_vit(self, vit_root="./ckpts/SigLIP"):
        self.vit = AutoModel.from_pretrained(vit_root).vision_model

    def _init_proj(self):
        # SigLIP (1152) → LLM hidden size (4096)
        self.adapter = nn.Sequential(
            nn.Linear(1152, self.model.config.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=True),
        )
```

**핵심 컴포넌트**:

| 컴포넌트 | 역할 | 입력 | 출력 |
|---------|------|------|------|
| SigLIP ViT | 이미지 인코딩 | Image (384~1152) | ViT Features (1152D) |
| Adapter | 차원 변환 | ViT Features | LLM Embeddings (4096D) |
| VQ-VAE | Discrete 토큰화 | Image | VQ Tokens (8192 codebook) |
| Lumina-mGPT | 통합 추론 | Hybrid Tokens | Text/Image Generation |

---

## Hybrid 토큰 시스템

### 1. 토큰 정의

**파일**: `unitoken/data/item_processor.py`

**라인 62-67**: FlexARItemProcessor 클래스

```python
class FlexARItemProcessor(MMConvItemProcessor):
    image_start_token = "<racm3:break>"  # BOI (8197)
    image_end_token = "<eoss>"           # EOI (8196)
    full_sub_sep_token = "<reserved08796>"
    sub_sub_sep_token = "<reserved08797>"
    sub_skip_token = "<reserved08798>"
    new_line_token = "<reserved08799>"   # 8803
```

**특수 토큰**:

| 토큰 | ID | 용도 |
|------|-----|------|
| `<racm3:break>` | 8197 | Start of Image (BOI) |
| `<eoss>` | 8196 | End of Image (EOI) |
| `<sentinel:0>` | 8198 | Separator between discrete and continuous |
| `<reserved08799>` | 8803 | New line in image grid |
| `<reserved08820>` | 8820 | Resolution token (512x512) |

### 2. Discrete 토큰 생성

**라인 112-147**: `process_image()` 메서드

```python
@torch.no_grad()
def process_image(self, image) -> Dict:
    # Center crop
    image = var_center_crop(image, crop_size_list=self.crop_size_list)

    # Calculate grid size
    w_grids, h_grids = image.size[0] // self.patch_size, image.size[1] // self.patch_size

    # VQ-VAE tokenization
    image_toks = self.chameleon_ori_translation.convert_img2bp2(
        self.chameleon_ori_image_tokenizer.img_tokens_from_pil(image)
    ).view(-1)

    # Reshape to 2D grid
    full_image_toks = image_toks.reshape(image.size[1] // 16, image.size[0] // 16)
    new_line_id = self.token2id(self.new_line_token)

    # Add new line tokens
    full_image_toks = torch.cat(
        (
            full_image_toks,
            torch.ones(image.size[1] // 16, 1, device=full_image_toks.device, dtype=full_image_toks.dtype)
            * new_line_id,
        ),
        dim=1,
    ).flatten()

    # Final token sequence: [BOI] [h_grids] [w_grids] [image_tokens] [EOI]
    result_toks = [
        self.token2id(self.image_start_token),  # 8197
        self.token2id(self.get_n_grids_token(h_grids)),  # e.g., 8820 for 16x16
        self.token2id(self.get_n_grids_token(w_grids)),
        *full_image_toks.tolist(),
        self.token2id(self.image_end_token),  # 8196
    ]

    return {"input_ids": result_toks, "labels": result_toks}
```

**토큰 시퀀스 예시** (512x512 이미지):
```
[8197] [8820] [8820] [4] [123] [456] ... [8803] [789] ... [8196]
  ↑      ↑      ↑     ↑___________________↑_____↑___________↑
 BOI   h_grids w_grids   VQ tokens (32x32)  newline      EOI
```

### 3. Continuous 토큰 생성

**파일**: `unitoken/model/modeling_xllmx_chameleon_anyres.py`

**라인 88-121**: Forward pass의 continuous token 생성

```python
# Generate continuous visual tokens with 'anyres' configuration
continuous_tokens = []
eol_token = self.model.embed_tokens(torch.tensor(8803, dtype=torch.int64, device=self.device))

for batch_id in range(len(discrete_ids)):
    # SigLIP ViT encoding
    vit_feat = self.vit(images[batch_id], interpolate_pos_encoding=True).last_hidden_state

    # Adapter projection
    image_feature = self.adapter(vit_feat)

    # Any-resolution handling
    if image_feature.shape[0] > 1:
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]

        # Recover 2D grid pinpoints
        num_patch_width, num_patch_height = get_anyres_image_grid_shape(
            image_sizes[batch_id], self.image_grid_pinpoints, self.vit.config.image_size
        )

        # Reshape and unpad
        height = width = self.vit.config.image_size // self.vit.config.patch_size
        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_sizes[batch_id])

        # Add EOL token
        image_feature = torch.cat((
            image_feature,
            eol_token[:, None, None].expand(*image_feature.shape[:-1], 1).to(self.device)
        ), dim=-1)

        # Flatten and combine with base
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    else:
        image_feature = image_feature[0]
        image_feature = torch.cat((
            image_feature,
            eol_token[None].to(self.device)
        ), dim=0)

    continuous_tokens.append(image_feature)
```

**Any-Resolution 처리**:
- Grid pinpoints: `[384, 768]`, `[768, 384]`, `[768, 768]`, `[384, 1152]`, `[1152, 384]`
- 이미지를 여러 패치로 분할
- Base patch + High-resolution patches
- Unpadding으로 원본 종횡비 유지

### 4. Hybrid 토큰 결합

**라인 144-200**: Discrete와 Continuous 결합

```python
# Insert continuous tokens into discrete tokens
# Format: "<soi>[discrete_tokens]<sep>[continuous_tokens]<eoi>"
sep_token = self.model.embed_tokens(torch.tensor(8198, dtype=torch.int64, device=self.device))

for batch_id in range(len(discrete_ids)):
    image_end_pos = torch.where(discrete_ids[batch_id]==8196)[0]  # EOI position

    if len(image_end_pos) == 0:    # Text-only samples
        # Pad to match multimodal sequence length
        uni_token = torch.cat([
            discrete_token,
            pad_token.unsqueeze(0).repeat(pad_num,1)
        ], dim=0)
    else:   # Multimodal samples
        discrete_token = discrete_tokens[batch_id]
        continuous_token = continuous_tokens[batch_id]

        # Hybrid sequence: [discrete_before_EOI] [SEP] [continuous] [discrete_after_EOI]
        uni_token = torch.cat([
            discrete_token[:image_end_pos],  # Up to EOI
            sep_token.unsqueeze(0),          # Separator
            continuous_token,                # Continuous tokens
            discrete_token[image_end_pos:]   # From EOI onwards
        ], dim=0)
```

**Unified Sequence 구조**:
```
Text: "Describe this image: "
    ↓
[text tokens] [8197] [8820] [8820] [VQ tokens...] [8196]
                ↑                                    ↑
               BOI                                  EOI (image_end_pos)
    ↓
[text tokens] [8197] [8820] [8820] [VQ tokens...] [8196] [8198] [SigLIP features...] [remaining text]
                                                            ↑            ↑
                                                          SEP      Continuous tokens
```

---

## BOI 토큰과 이미지 생성

### 1. BOI 토큰 수동 삽입

**파일**: `unitoken/inference_solver_anyres.py`

**라인 357-422**: `generate_img()` 메서드

```python
@torch.no_grad()
def generate_img(
    self,
    images: Image.Image | str | List[Union[Image.Image, str]],
    qas,
    max_gen_len,
    temperature,
    logits_processor=None,
    streamer=None,
    num_return_sequences=1,
):
    conversations = []
    for q, a in qas:
        conversations.append({"from": "human", "value": q})
        conversations.append({"from": "gpt", "value": a})

    item = {"image": images, "conversations": conversations}

    # Process item to get prompt
    _prompt = self.item_processor.process_item(item)
    prompt = []
    for value in _prompt:
        if isinstance(value, int):
            prompt.append(value)
        else:
            prompt += value["input_ids"]

    prompt_len = len(prompt)

    # ⭐ Manually Add <soi> token to guarantee image generation success rate
    # These added part should be contained in the answer part
    prompt += [16853, 8197]  # [16853] is unknown, [8197] is BOI
    prompt = torch.tensor(prompt, dtype=torch.int64, device=self.model.device).unsqueeze(0)

    # ... generation code ...
```

**핵심 발견**:
- BOI 토큰 (8197)이 **코드에서 명시적으로 추가됨**
- 주석: "Manually Add `<soi>` token to guarantee image generation success rate"
- 모델이 자동으로 예측하는 것이 아님

**결론**: ❌ **BOI 토큰 자동 예측 불가능** (Show-o와 유사)

### 2. 이미지 생성 프롬프트 예시

**README.md 라인 77-100**:

```python
# Image Generation
q1 = f"Generate an image according to the following prompt:\n" \
     f"A majestic phoenix with fiery wings soaring above a tranquil mountain lake..."

generated = inference_solver.generate_img(
    images=[],
    qas=[[q1, None]],
    max_gen_len=1536,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(cfg=3.0, image_top_k=4000),
)

a1, new_image = generated[0], generated[1][0]
```

**생성 과정**:
1. 텍스트 프롬프트 입력
2. **BOI 토큰 수동 추가** (`prompt += [16853, 8197]`)
3. Autoregressive 생성
4. MultiModalLogitsProcessor가 이미지 토큰 생성 제어

---

## MultiModalLogitsProcessor

### 1. LogitsProcessor 역할

**파일**: `unitoken/inference_solver_anyres.py`

**라인 169-261**: MultiModalLogitsProcessor 클래스

```python
class MultiModalLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
    ):
        self.image_start_token_id = image_start_token_id  # 8197
        self.image_end_token_id = image_end_token_id      # 8196
        self.image_next_line_token_id = image_next_line_token_id  # 8803
        self.patch_size = patch_size  # 32

        # Image token list: [4, 5, ..., 8195]
        self.image_token_list = [i for i in range(4, 8195 + 1)]

        # Suppress non-image tokens during image generation
        self.suppress_tokens = torch.tensor(
            [x for x in self.vocab_list if x not in self.image_token_list], device="cuda"
        )
```

**핵심**:
- 이미지 생성 중에는 이미지 토큰만 허용
- 텍스트 토큰은 억제 (suppress)

### 2. 생성 제어 메커니즘

**라인 207-261**: `__call__()` 메서드

```python
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

    self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
    self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

    # Case 1: Not in image generation mode
    if self.num_image_start_tokens == self.num_image_end_tokens:
        self.h_latent_dim, self.w_latent_dim = None, None
        self.image_start_token_id_index = None
        return scores  # No modification

    # Case 2: In image generation mode
    elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
        if self.image_start_token_id_index is None:
            self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()

        new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])

        # Step 1: Force resolution token (8820 for 512x512)
        if new_token_num < 2:
            resolution_constrained_scores = torch.full_like(scores, -math.inf)
            resolution_constrained_scores[:, 8820] = 0  # Force 512x512
            return resolution_constrained_scores

        # Step 2: Decode grid size from resolution tokens
        if new_token_num >= 2:
            if self.h_latent_dim is None or self.w_latent_dim is None:
                h_grids, w_grids = (
                    input_ids[0][self.image_start_token_id_index + 1] - 8804,
                    input_ids[0][self.image_start_token_id_index + 2] - 8804,
                )
                self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2

            tokens = input_ids[0][self.image_start_token_id_index + 3 :]

            # Step 3: Force new line token at row end
            if (len(tokens) + 1) % (self.w_latent_dim + 1) == 0:
                new_line_constrained_scores = torch.full_like(scores, -math.inf)
                new_line_constrained_scores[:, self.image_next_line_token_id] = 0
                return new_line_constrained_scores

            # Step 4: Force EOI token at image end
            elif (len(tokens) + 1) == (self.w_latent_dim + 1) * self.h_latent_dim + 1:
                eos_image_constrained_scores = torch.full_like(scores, -math.inf)
                eos_image_constrained_scores[:, self.image_end_token_id] = 0
                return eos_image_constrained_scores

            # Step 5: Allow only image tokens
            elif (len(tokens) + 1) % (self.w_latent_dim + 1) != 0:
                image_constrained_scores = torch.where(self.suppress_token_mask, -float("inf"), scores)
                return image_constrained_scores

    return scores
```

**생성 단계**:

```
Input: [text...] [8197]
                   ↑
                  BOI detected
    ↓
Step 1: Force resolution tokens
Output: [text...] [8197] [8820] [8820]
                           ↑      ↑
                         h_grids w_grids (16x16 for 512x512)
    ↓
Step 2-5: Autoregressive generation with constraints
Output: [text...] [8197] [8820] [8820] [VQ token] [VQ token] ... [8803] ... [8196]
                                         ↑___________↑_____________↑         ↑
                                      Image tokens (32x32)    New line    EOI
```

### 3. InterleavedTopKLogitsWarper

**라인 264-308**: 이미지/텍스트별 top-k 적용

```python
class InterleavedTopKLogitsWarper(LogitsWarper):
    def __init__(
        self,
        image_top_k: int,  # e.g., 4000
        text_top_k: int,   # e.g., 10
        image_start_token_id=None,
        image_end_token_id=None,
    ):
        self.image_top_k = max(image_top_k, min_tokens_to_keep)
        self.text_top_k = max(text_top_k, min_tokens_to_keep)
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # In image generation mode
        if self.num_image_start_tokens == self.num_image_end_tokens + 1:
            top_k = min(self.image_top_k, scores.size(-1))  # Use image_top_k
        else:
            top_k = min(self.text_top_k, scores.size(-1))   # Use text_top_k

        # Apply top-k filtering
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
```

**핵심**:
- 이미지 생성 시: `image_top_k=4000` (높은 다양성)
- 텍스트 생성 시: `text_top_k=10` (집중된 샘플링)

---

## Interleaved 생성 가능성

### 1. 현재 구현 분석

**BOI 토큰 삽입 위치** (라인 394):
```python
# Manually Add <soi> token to guarantee image generation success rate
prompt += [16853, 8197]
```

**핵심 발견**:
- BOI 토큰이 **생성 전에 명시적으로 추가됨**
- `generate_img()` 메서드는 이미지 생성 전용
- `generate()` 메서드는 이해 전용

**이미지 이해 예시** (README 라인 103-127):
```python
# Image Understanding
q1 = "<|image|>Please describe the details of the image as much as possible."

images = [Image.open("../assets/1.png").convert('RGB')]
qas = [[q1, None]]

generated = inference_solver.generate(
    images=images,
    qas=qas,
    max_gen_len=512,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
)

a1 = generated[0]
# generated[1], namely the list of newly generated images, should typically be empty in this case.
```

**메서드 분리**:
- `generate_img()`: 이미지 생성 (BOI 자동 추가)
- `generate()`: 텍스트 생성 및 이미지 이해

### 2. Interleaved 시나리오

**시나리오: 텍스트 → 이미지 → 텍스트**

```python
# Step 1: Generate image
prompt_1 = "Generate an image of a cat"
result_1 = inference_solver.generate_img(
    images=[],
    qas=[[prompt_1, None]],
    ...
)
text_1, image_1 = result_1[0], result_1[1][0]

# Step 2: Describe generated image
prompt_2 = "<|image|>Describe this generated image."
result_2 = inference_solver.generate(
    images=[image_1],
    qas=[[prompt_2, None]],
    ...
)
text_2 = result_2[0]
```

**문제점**:
1. **수동 orchestration**: 각 단계를 명시적으로 호출
2. **메서드 분리**: `generate_img()`와 `generate()` 별도 호출
3. **BOI 자동 예측 불가**: 모델이 스스로 이미지 생성 시작 불가
4. **컨텍스트 유지 어려움**: 이전 출력을 다음 입력으로 수동 전달

### 3. 자동 Interleaved 가능성

**MultiModalLogitsProcessor 분석**:

```python
if self.num_image_start_tokens == self.num_image_end_tokens:
    return scores  # 이미지 모드 아님 - 모든 토큰 허용
```

**핵심**:
- 이미지 생성 모드가 아니면 **모든 토큰 허용**
- 이론적으로 BOI 토큰 (8197) 예측 가능
- **하지만**, `generate_img()`에서 BOI를 명시적으로 추가하므로 실제로는 사용하지 않음

**추론**:
- ⚠️ **기술적으로 가능하지만, 실용적으로 제한적**
- 모델이 BOI를 예측하도록 훈련되었는지 불확실
- 공식 예제에서는 모두 수동 BOI 삽입

**결론**: ❌ **완전 자동 Interleaved 생성 불가능** (Chameleon과 달리)

### 4. 비교표

| 특징 | UniToken | Chameleon | Show-o | SEED-X |
|------|----------|-----------|--------|--------|
| BOI 예측 | ❌ 수동 삽입 | ✅ 자동 예측 | ❌ 수동 삽입 | ⚠️ 제한적 |
| Interleaved | ⚠️ 다중 추론 | ✅ 완전 자동 | ❌ 다중 추론 | ⚠️ 제한적 |
| 이미지 표현 | Hybrid (D+C) | Discrete | Discrete | Learnable Query |
| 제어성 | ✅ 메서드 분리 | ⚠️ 암묵적 | ✅ 태스크 토큰 | ⚠️ Processor |

---

## 핵심 특징

### 1. Hybrid Visual Encoding

**장점**:
- **이해**: Continuous tokens (SigLIP) → 풍부한 semantic features
- **생성**: Discrete tokens (VQ-VAE) → 효율적인 autoregressive generation
- Best of both worlds

**구조**:
```
Image
  ├─ Understanding → SigLIP → Adapter → Continuous (729 tokens)
  └─ Generation → VQ-VAE → Discrete (1024 tokens)
```

**Trade-off**:
- ✅ 높은 이해 성능 (continuous features)
- ✅ 빠른 생성 속도 (discrete tokens)
- ❌ 두 경로 필요 (메모리 증가)
- ❌ VQ-VAE 품질 제한 (discrete generation)

### 2. Any-Resolution 지원

**Grid Pinpoints**:
```python
self.image_grid_pinpoints = [
    [384, 768],   # 1:2 ratio
    [768, 384],   # 2:1 ratio
    [768, 768],   # 1:1 ratio
    [384, 1152],  # 1:3 ratio
    [1152, 384]   # 3:1 ratio
]
```

**처리 과정**:
1. 이미지를 여러 grid로 분할
2. 각 grid를 SigLIP으로 인코딩
3. Base patch + High-res patches 결합
4. Unpadding으로 원본 종횡비 유지

**장점**:
- 다양한 해상도 이미지 효율적 처리
- 고해상도 디테일 보존
- SEED-X와 유사한 접근

### 3. Classifier-Free Guidance

**파일**: `unitoken/inference_solver_anyres.py`

**라인 50-166**: LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor

```python
class LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        guidance_scale: float,  # e.g., 3.0
        model,
        image_start_token_id,
        image_end_token_id,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": transformers.DynamicCache() if use_cache else None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids, image_start_token_id_index):
        # Unconditional forward pass
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, image_start_token_id_index:]
            # ...
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        return out.logits

    def __call__(self, input_ids, scores):
        num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # Only apply CFG during image generation
        if num_image_start_tokens == num_image_end_tokens + 1:
            if self.guidance_scale == 1.0:
                return scores

            unconditional_logits = self.get_unconditional_logits(input_ids, self.image_start_token_id_index)[:, -1]

            # CFG formula
            scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
            return scores_processed

        return scores
```

**CFG Formula**:
```
scores_cfg = guidance_scale * (scores_cond - scores_uncond) + scores_uncond
```

**특징**:
- 이미지 생성 시만 적용
- BOI 토큰 감지 후 활성화
- KV-cache 사용으로 효율적

### 4. Recaptioned Prompts

**기여**:
- GenEval 및 T2I-Compbench 프롬프트 재작성
- 짧은 프롬프트 → 상세한 설명으로 개선
- Hugging Face에 공개: [OceanJay/rewrite_geneval_t2icompbench](https://huggingface.co/datasets/OceanJay/rewrite_geneval_t2icompbench)

**예시**:
```
원본: "A cat"
재작성: "A fluffy orange tabby cat with green eyes sitting on a windowsill,
         looking outside at a sunny garden with colorful flowers."
```

### 5. 성능

**벤치마크** (논문 기준):
- GenEval: 향상된 성능 (recaptioned prompts 사용)
- T2I-Compbench++: 경쟁력 있는 결과
- Multimodal Understanding: LLaVA 벤치마크에서 강력한 성능

**Trade-off**:
- ✅ Hybrid encoding으로 이해 성능 우수
- ⚠️ VQ-VAE 기반 생성의 품질 제한
- ✅ Any-resolution으로 유연성 확보

---

## 결론

### Type B5 (Hybrid)의 대표 모델

UniToken은 **Discrete와 Continuous를 결합**한 Type B5 방식의 대표적인 구현입니다:

| 특징 | UniToken | Janus | Show-o | Chameleon |
|------|----------|-------|--------|-----------|
| 타입 | B5 Hybrid | B2 Separate | C Fused | B1 Single Token |
| 이해 표현 | Continuous (SigLIP) | Continuous (CLIP) | Discrete | Discrete |
| 생성 표현 | Discrete (VQ) | Discrete (VQ) | Discrete (VQ) | Discrete (VQ) |
| BOI 예측 | ❌ 수동 | ❌ 불가능 | ❌ 수동 | ✅ 자동 |
| Interleaved | ⚠️ 다중 추론 | ❌ 불가능 | ❌ 다중 추론 | ✅ 완전 자동 |
| 통합 방식 | Single LLM | Dual-Path | Single Transformer | Single Transformer |
| Any-Res | ✅ 지원 | ❌ 고정 | ❌ 고정 | ❌ 고정 |

### 핵심 설계 원칙

1. **Dual-Path Visual Encoding**
   - Understanding: SigLIP (1152D) → Adapter → Continuous
   - Generation: VQ-VAE (8192 codebook) → Discrete
   - Separator token (8198)로 두 표현 구분

2. **Hybrid Token Sequence**
   ```
   [text] [8197] [8820] [8820] [VQ tokens] [8196] [8198] [SigLIP features] [text]
            ↑      ↑      ↑        ↑          ↑      ↑           ↑
           BOI  h_grids w_grids  Discrete   EOI    SEP     Continuous
   ```

3. **Structured Generation**
   - MultiModalLogitsProcessor로 강제 구조
   - Resolution token → Grid tokens → VQ tokens → Newline → EOI
   - Deterministic structure, probabilistic content

4. **Conditional CFG**
   - BOI 감지 후 활성화
   - Unconditional context caching
   - Guidance scale로 품질 조절

### 장점

1. **최상의 양쪽 세계**
   - 이해: Continuous features로 rich semantics
   - 생성: Discrete tokens로 efficient autoregressive

2. **Any-Resolution**
   - 다양한 종횡비 지원
   - High-resolution 디테일 보존
   - Dynamic padding

3. **명시적 제어**
   - `generate_img()` vs `generate()` 메서드 분리
   - BOI 수동 삽입으로 예측 가능한 동작

4. **커뮤니티 기여**
   - Recaptioned prompts 공개
   - T2I 벤치마크 개선

### 한계점

1. **자동 Interleaved 불가능**
   - BOI 토큰 수동 삽입 필요
   - 다중 추론 orchestration 필요
   - Chameleon처럼 자연스러운 전환 불가

2. **Dual-Path 오버헤드**
   - SigLIP + VQ-VAE 모두 필요
   - 메모리 사용량 증가
   - 두 인코더 학습 필요

3. **VQ-VAE 품질 제한**
   - Discrete tokenization의 정보 손실
   - SDXL 같은 diffusion model보다 품질 낮음

4. **복잡한 토큰 구조**
   - Hybrid sequence 구성 복잡
   - Separator, resolution tokens 등 관리 필요

### 의의

UniToken은 **"실용적 Hybrid" 접근법**을 제시합니다:

- Janus: Dual-path, 하지만 interleaved 불가
- Chameleon: Single path, 완전 자동, 하지만 이해 성능 제한
- **UniToken**: Hybrid path, 명시적 제어, 우수한 이해 성능

**Type B5의 벤치마크**로서, Discrete와 Continuous의 장점을 모두 활용하는 설계 패러다임을 보여줍니다. 특히 Any-resolution 지원과 recaptioned prompts 기여는 멀티모달 커뮤니티에 중요한 자산입니다.
