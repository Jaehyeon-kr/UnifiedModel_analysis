# Emu3 - Type B1 Single Token

## 목차
1. [개요](#개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [VisionTokenizer](#visiontokenizer)
4. [토큰 체계와 프롬프트 구조](#토큰-체계와-프롬프트-구조)
5. [BOI 토큰과 이미지 생성](#boi-토큰과-이미지-생성)
6. [PrefixConstrainedLogitsProcessor](#prefixconstrainedlogitsprocessor)
7. [Interleaved 생성 가능성](#interleaved-생성-가능성)
8. [핵심 특징](#핵심-특징)
9. [결론](#결론)

---

## 개요

**Emu3**는 BAAI (Beijing Academy of Artificial Intelligence)에서 개발한 "Next-Token Prediction is All You Need" 모델로, **순수하게 next-token prediction만으로** 멀티모달 이해와 생성을 수행하는 Type B1 (Single Token) 모델입니다.

**핵심 특징**:
- ✅ Next-token prediction만 사용 (diffusion 없음)
- ✅ 단일 Transformer (8B parameters)
- ✅ Discrete visual tokenization (VisionVQ)
- ❌ BOI 토큰 자동 예측 불가능
- ❌ 완전 자동 Interleaved 생성 불가능
- ✅ 이미지, 텍스트, 비디오 지원
- ✅ Two-stage model: Emu3-Gen (생성) / Emu3-Chat (이해)

**저장소**: `d:\Check_\janus\analysis\repos\type_b1_single_token\Emu3`

**논문**: [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869)

**주요 성과**:
- SDXL, LLaVA-1.6, OpenSora-1.2 등을 능가
- Diffusion이나 compositional architecture 불필요
- 유연한 해상도와 스타일 지원

---

## 아키텍처 분석

### 1. Type B1 - Single Token 방식

**특징**:
- 이미지, 텍스트, 비디오를 **단일 discrete token space**로 통합
- Transformer from scratch 학습
- Next-token prediction objective만 사용

**비교표**:

| 모델 | 이미지 표현 | 생성 방식 | Diffusion 사용 |
|------|-----------|----------|--------------|
| SDXL | Continuous | Diffusion | ✅ Yes |
| Chameleon | Discrete VQ | Autoregressive | ❌ No |
| **Emu3** | **Discrete VQ** | **Autoregressive** | **❌ No** |
| Show-o | Discrete VQ | MaskGIT (Discrete Diffusion) | ✅ Yes (Discrete) |

### 2. 전체 구조

```
Input (Image/Text/Video)
    ↓
VisionVQ Tokenizer
    ├─ Image: 8192 codebook
    ├─ Video: 3D causal VQ
    └─ Spatial downsample: 8x
    ↓
Discrete Visual Tokens
    ↓
Unified Token Sequence
    ├─ Text tokens (Tiktoken-based)
    └─ Visual tokens (<|visual token XXXXXX|>)
    ↓
Single Transformer (8B)
    ├─ LLaMA-like architecture
    ├─ RMSNorm
    └─ Rotary Position Embedding
    ↓
Next-Token Prediction
    ├─ Text: BPE tokens
    └─ Image: Visual tokens
    ↓
VisionVQ Decoder (이미지 생성 시)
    ↓
Generated Image/Video
```

### 3. Two-Stage Approach

**Emu3-Stage1**:
- Pre-training 모델
- Image captioning, 512x512 이미지 생성

**Emu3-Gen**:
- 이미지/비디오 생성 특화
- SFT (Supervised Fine-Tuning)
- High-quality generation

**Emu3-Chat**:
- 멀티모달 이해 특화
- Vision-language understanding
- VQA, image captioning, etc.

---

## VisionTokenizer

### 1. 구조

**파일**: `emu3/tokenizer/modeling_emu3visionvq.py`

**특징**:
- **Codebook size**: 8192 (Chameleon과 동일)
- **Spatial downsample factor**: 8 (vs Chameleon의 16)
- **3D Causal VQ**: 비디오 지원
- **Temporal downsample factor**: 4 (video)

**Image Tokenization**:
```
Input Image (720 x 720)
    ↓
Encoder (8x downsample)
    ↓
Latent (90 x 90)
    ↓
VQ Quantization (8192 codebook)
    ↓
Visual Tokens (90 x 90 = 8100 tokens)
```

**Video Tokenization**:
```
Input Video (T frames, H x W)
    ↓
3D Encoder (8x spatial, 4x temporal)
    ↓
Latent (T/4 x H/8 x W/8)
    ↓
VQ Quantization
    ↓
Visual Tokens
```

### 2. Chameleon과의 차이

| 특징 | Emu3 | Chameleon |
|------|------|-----------|
| Codebook Size | 8192 | 8192 |
| Spatial Downsample | 8x | 16x |
| Token Count (720x720) | 8100 (90x90) | 2025 (45x45) |
| Video Support | ✅ 3D Causal VQ | ❌ Image only |
| Temporal Downsample | 4x | N/A |

**장점**:
- 더 많은 토큰 (8100 vs 2025)으로 더 세밀한 표현
- 비디오 지원
- Causal structure로 autoregressive generation 용이

---

## 토큰 체계와 프롬프트 구조

### 1. 특수 토큰 정의

**파일**: `emu3/mllm/tokenization_emu3.py`

**라인 60-74**: Emu3Tokenizer 초기화

```python
def __init__(
    self,
    vocab_file,
    special_tokens_file,
    errors="replace",
    bos_token = "<|extra_203|>",
    eos_token = "<|extra_204|>",
    pad_token = "<|endoftext|>",
    img_token = "<|image token|>",
    boi_token = "<|image start|>",  # 151852
    eoi_token = "<|image end|>",    # 151853
    eol_token = "<|extra_200|>",    # End of Line
    eof_token = "<|extra_201|>",    # End of Frame
    **kwargs,
):
```

**특수 토큰**:

| 토큰 | ID | 용도 |
|------|-----|------|
| `<|image start|>` | 151852 | Begin of Image (BOI) |
| `<|image end|>` | 151853 | End of Image (EOI) |
| `<|image token|>` | - | Image token placeholder |
| `<|extra_200|>` | - | End of Line (EOL) |
| `<|extra_201|>` | - | End of Frame (EOF, video) |
| `<|extra_203|>` | - | BOS |
| `<|extra_204|>` | - | EOS |

**Visual Token Template**:
```python
visual_template=("<|visual token {token_id:0>6d}|>", r"<\|visual token (\d+)\|>")
```

**예시**:
- `<|visual token 000000|>` (codebook ID 0)
- `<|visual token 008191|>` (codebook ID 8191)

### 2. 프롬프트 구조

**파일**: `emu3/mllm/processing_emu3.py`

**Generation Mode (mode='G')** 라인 177-183:
```python
h, w = self.calculate_generate_size(ratio[idx], image_area, self.vision_tokenizer.spatial_scale_factor)
image_prompt = (
    self.tokenizer.boi_token +
    self.prefix_template.format(H=h, W=w) +
    self.tokenizer.img_token
)
prompt += (text_prompt + image_prompt)
```

**생성 프롬프트 예시**:
```
<|extra_203|>a portrait of young girl. masterpiece, film grained, best quality.<|image start|>90*90<|image token|>
    ↑           ↑                                                              ↑           ↑      ↑
   BOS      Text prompt                                                      BOI      Resolution IMG
```

**Understanding Mode (mode='U')** 라인 163-175:
```python
h, w = image_tokens[idx].shape
imgstr = self.to_imgstr(image_tokens[idx])
image_prompt = (
    self.tokenizer.boi_token +
    self.prefix_template.format(H=h, W=w) +
    self.tokenizer.img_token +
    imgstr +
    self.tokenizer.eol_token +
    self.tokenizer.eof_token +
    self.tokenizer.eoi_token
)
prompt += self.chat_template.format(image_prompt=image_prompt, text_prompt=text_prompt)
```

**이해 프롬프트 예시**:
```
<|extra_203|>You are a helpful assistant. USER: <|image start|>90*90<|image token|><|visual token 000123|><|visual token 004567|>...<|extra_200|><|extra_201|><|image end|>Please describe the image. ASSISTANT:
    ↑                                          ↑           ↑      ↑            ↑                     ↑           ↑            ↑            ↑
   BOS                                        BOI      Resolution IMG       Visual tokens (90x90)  EOL         EOF         EOI       Question
```

**prefix_template**: `{H}*{W}`
- 해상도 정보를 명시적으로 토큰에 포함
- 예: `90*90` (720x720 image → 90x90 tokens)

---

## BOI 토큰과 이미지 생성

### 1. BOI 토큰 명시적 삽입

**파일**: `emu3/mllm/processing_emu3.py`

**라인 177-183**: Generation mode 프롬프트 생성

```python
else:  # mode == 'G'
    h, w = self.calculate_generate_size(ratio[idx], image_area, self.vision_tokenizer.spatial_scale_factor)
    image_prompt = (
        self.tokenizer.boi_token +  # ⭐ BOI 토큰 명시적 추가
        self.prefix_template.format(H=h, W=w) +
        self.tokenizer.img_token
    )
    prompt += (text_prompt + image_prompt)
```

**핵심 발견**:
- BOI 토큰이 **processor에서 명시적으로 추가**
- 모델이 예측하는 것이 아님
- 사용자 텍스트 프롬프트 뒤에 자동 추가

**결론**: ❌ **BOI 토큰 자동 예측 불가능** (Show-o, UniToken과 유사)

### 2. 이미지 생성 과정

**파일**: `image_generation.py`

**라인 29-46**: 프롬프트 준비

```python
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error..."

classifier_free_guidance = 3.0
prompt = ["a portrait of young girl.", "a shiba inu"]
prompt = [p + POSITIVE_PROMPT for p in prompt]

kwargs = dict(
    mode='G',  # Generation mode
    ratio=["1:1", "16:9"],
    image_area=model.config.image_area,
    return_tensors="pt",
    padding="longest",
)
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=[NEGATIVE_PROMPT] * len(prompt), **kwargs)
```

**라인 61-71**: LogitsProcessor 설정

```python
h = pos_inputs.image_size[:, 0]
w = pos_inputs.image_size[:, 1]
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList([
    UnbatchedClassifierFreeGuidanceLogitsProcessor(
        classifier_free_guidance,
        model,
        unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
    ),
    PrefixConstrainedLogitsProcessor(
        constrained_fn,  # ⭐ 구조 제약
        num_beams=1,
    ),
])
```

**라인 73-86**: 생성 및 디코딩

```python
outputs = model.generate(
    pos_inputs.input_ids.to("cuda:0"),
    GENERATION_CONFIG,
    logits_processor=logits_processor,
    attention_mask=pos_inputs.attention_mask.to("cuda:0"),
)

for idx_i, out in enumerate(outputs):
    mm_list = processor.decode(out)
    for idx_j, im in enumerate(mm_list):
        if not isinstance(im, Image.Image):
            continue
        im.save(f"result_{idx_i}_{idx_j}.png")
```

---

## PrefixConstrainedLogitsProcessor

### 1. 구조 제약 메커니즘

**파일**: `emu3/mllm/utils_emu3.py`

**Emu3PrefixConstrainedLogitsHelper 클래스**:

```python
class Emu3PrefixConstrainedLogitsHelper:
    def __init__(
        self,
        tokenizer,
        prefix_template,
        visual_template,
        boi_token,
        eoi_token,
        visual_tokens,
        height,
        width,
    ):
        self.tokenizer = tokenizer
        self.prefix_template = prefix_template
        self.visual_template = visual_template
        self.boi_token = boi_token
        self.eoi_token = eoi_token
        self.visual_tokens = visual_tokens
        self.height = height
        self.width = width

        # Expected sequence structure
        self.seq = [
            boi_token,
            prefix_template.format(H=height, W=width),
            *[visual_template[0].format(token_id=0)] * (height * width),
            eoi_token,
        ]
```

**핵심**:
- 이미지 생성 시퀀스 구조를 **미리 정의**
- `[BOI] [H*W] [visual tokens...] [EOI]` 형태 강제
- Prefix-constrained decoding으로 구조 보장

### 2. 동작 과정

**생성 시퀀스**:

```
Input: "a cat<|image start|>90*90<|image token|>"
    ↓
Step 1: BOI 이후 해상도 토큰 검증
    ↓
Step 2: Visual token 생성 (90x90 = 8100 tokens)
    ├─ <|visual token 000123|>
    ├─ <|visual token 004567|>
    ├─ ...
    └─ <|visual token 001234|>
    ↓
Step 3: EOI 토큰 생성
Output: "a cat<|image start|>90*90<|image token|><|visual token ...>...<|image end|>"
```

**제약 사항**:
1. BOI 다음 반드시 해상도 정보 (`{H}*{W}`)
2. Visual token 정확히 `H * W` 개 생성
3. 마지막 EOI 토큰으로 종료

### 3. Classifier-Free Guidance

**UnbatchedClassifierFreeGuidanceLogitsProcessor**:

```python
UnbatchedClassifierFreeGuidanceLogitsProcessor(
    classifier_free_guidance=3.0,  # Guidance scale
    model=model,
    unconditional_ids=neg_inputs.input_ids,
)
```

**CFG Formula**:
```
logits = unconditional_logits + guidance_scale * (conditional_logits - unconditional_logits)
```

**특징**:
- Positive prompt + Negative prompt
- Guidance scale 조절로 품질 향상
- Unbatched 방식 (메모리 효율)

---

## Interleaved 생성 가능성

### 1. 현재 구현 분석

**Mode 분리**:
```python
# Generation mode
kwargs = dict(mode='G', ...)
pos_inputs = processor(text=prompt, **kwargs)

# Understanding mode
kwargs = dict(mode='U', ...)
inputs = processor(text=text, image=image, **kwargs)
```

**핵심 발견**:
- Generation (mode='G')과 Understanding (mode='U') 완전 분리
- BOI 토큰이 processor에서 자동 추가
- 두 모델 분리: Emu3-Gen, Emu3-Chat

**결론**: ❌ **단일 모델로 interleaved 생성 불가능**

### 2. Two-Model Approach

**시나리오: 텍스트 → 이미지 → 텍스트**

```python
# Step 1: Image generation (Emu3-Gen)
gen_model = AutoModelForCausalLM.from_pretrained("BAAI/Emu3-Gen")
prompt = "a cat" + POSITIVE_PROMPT
pos_inputs = processor(text=prompt, mode='G', ratio="1:1", ...)
outputs = gen_model.generate(pos_inputs.input_ids, ...)
generated_images = processor.decode(outputs[0])

# Step 2: Image understanding (Emu3-Chat)
chat_model = AutoModelForCausalLM.from_pretrained("BAAI/Emu3-Chat")
inputs = processor(
    text="Describe this image",
    image=generated_images[0],
    mode='U',
)
response = chat_model.generate(inputs.input_ids, ...)
description = processor.decode(response[0])
```

**문제점**:
1. **두 개의 별도 모델** 필요 (Emu3-Gen + Emu3-Chat)
2. **수동 orchestration**: 각 단계를 명시적으로 실행
3. **BOI 자동 예측 불가**: Processor가 항상 추가
4. **컨텍스트 유지 어려움**: 모델 간 전환 시 컨텍스트 손실

### 3. Chameleon과의 비교

| 특징 | Emu3 | Chameleon |
|------|------|-----------|
| BOI 예측 | ❌ Processor 추가 | ✅ 모델 예측 |
| Interleaved | ❌ Two-model | ✅ Single model |
| 모드 전환 | ❌ 수동 (mode='G'/'U') | ✅ 자동 (디코더 전환) |
| 모델 수 | 2개 (Gen/Chat) | 1개 |
| 제어성 | ✅ 명시적 모드 | ⚠️ 암묵적 |

### 4. 왜 Interleaved가 불가능한가?

**설계 철학**:
- Emu3-Gen: **생성 전문화** (CFG, high-quality output)
- Emu3-Chat: **이해 전문화** (VQA, captioning)
- Task-specific optimization

**Trade-off**:
- ✅ 각 태스크에서 최고 성능
- ❌ 통합 interleaved 생성 불가

**결론**: ⚠️ **기술적으로 단일 모델로 가능하지만, 실제로는 two-model approach**

---

## 핵심 특징

### 1. Next-Token Prediction Only

**파일**: README.md

**라인 16**:
> We train a single transformer from scratch on a mixture of multimodal sequences solely with **<i>next-token prediction</i>**!

**장점**:
- ✅ 단순한 objective (next-token prediction)
- ✅ Diffusion 불필요
- ✅ Unified training
- ✅ 확장성 (scaling laws 적용 가능)

**비교**:

| 모델 | Objective | Diffusion |
|------|-----------|-----------|
| SDXL | Diffusion | ✅ Yes |
| Stable Diffusion 3 | Flow matching | ✅ Yes |
| Show-o | Autoregressive + MaskGIT | ✅ Yes (Discrete) |
| **Emu3** | **Next-token prediction** | **❌ No** |

### 2. Flexible Resolution

**Resolution 지원**:
```python
ratio=["1:1", "16:9", "9:16", "4:3", "3:4", ...]
image_area=518400  # 720 x 720
```

**계산 방식**:
```python
def calculate_generate_size(ratio, image_area, spatial_scale_factor):
    w_ratio, h_ratio = map(int, ratio.split(':'))
    aspect_ratio = w_ratio / h_ratio

    # Calculate pixel size
    h_pixels = int((image_area / aspect_ratio) ** 0.5)
    w_pixels = int(h_pixels * aspect_ratio)

    # Calculate token size (8x downsample)
    h_tokens = h_pixels // spatial_scale_factor
    w_tokens = w_pixels // spatial_scale_factor

    return h_tokens, w_tokens
```

**예시**:
- `1:1` → 720x720 pixels → 90x90 tokens
- `16:9` → 960x540 pixels → 120x68 tokens

**장점**:
- 다양한 종횡비 지원
- 고정된 이미지 영역 (image_area)
- Token count 조절 가능

### 3. Video Support

**3D Causal VQ**:
```python
# Video autoencode
images = images.view(
    -1,
    model.config.temporal_downsample_factor,  # 4
    *images.shape[2:],
)

with torch.no_grad():
    codes = model.encode(images)  # 3D encoding
    recon = model.decode(codes)   # 3D decoding
```

**특징**:
- **Temporal downsample**: 4x
- **Spatial downsample**: 8x
- **Causal structure**: Autoregressive 생성 지원
- **Video extension**: 기존 비디오에서 다음 프레임 예측

**README 라인 29**:
> Emu3 simply generates a video causally by predicting the next token in a video sequence, unlike the video diffusion model as in Sora.

### 4. High-Quality Generation

**GenEval 벤치마크**:
- Emu3-Gen: State-of-the-art on multiple metrics
- SDXL 능가

**성능 비교표** (README.md):
```
Comparison with SDXL, LLaVA-1.6, OpenSora-1.2
Emu3 outperforms several well-established task-specific models
```

**품질 향상 기법**:
1. **Classifier-Free Guidance** (CFG = 3.0)
2. **Prefix-constrained decoding** (구조 보장)
3. **Positive/Negative prompts**
4. **Top-k sampling** (k=2048)

### 5. Training Pipeline

**Three-stage training**:

**Stage 1**: Pre-training
- 데이터: Image-text pairs
- 목적: Basic multimodal understanding & generation
- 출력: Emu3-Stage1 (512x512 generation, image captioning)

**Stage 2**: SFT (Supervised Fine-Tuning)
- Two separate models:
  - **Emu3-Gen**: Generation tasks (T2I, I2I, etc.)
  - **Emu3-Chat**: Understanding tasks (VQA, captioning, etc.)

**Stage 3** (TODO): DPO (Direct Preference Optimization)
- 품질 향상
- Human preference alignment

---

## 결론

### Type B1 (Single Token)의 순수 구현

Emu3는 **Next-token prediction만으로** 멀티모달을 통합한 Type B1의 순수한 구현입니다:

| 특징 | Emu3 | Chameleon | Show-o | UniToken |
|------|------|-----------|--------|----------|
| 타입 | B1 Single Token | B1 Single Token | C Fused | B5 Hybrid |
| 이미지 생성 | Autoregressive | Autoregressive | Discrete Diffusion | Autoregressive |
| Diffusion 사용 | ❌ No | ❌ No | ✅ Yes (Discrete) | ❌ No |
| BOI 예측 | ❌ Processor 추가 | ✅ 모델 예측 | ❌ 수동 삽입 | ❌ 수동 삽입 |
| Interleaved | ❌ Two-model | ✅ Single model | ❌ 다중 추론 | ⚠️ 다중 추론 |
| 모델 분리 | ✅ Gen/Chat | ❌ Unified | ❌ Unified | ❌ Unified |
| 비디오 지원 | ✅ 3D Causal VQ | ❌ Image only | ❌ Image only | ❌ Image only |
| Token count (720x720) | 8100 (90x90) | 2025 (45x45) | 1024 (32x32) | 1024 (32x32) |

### 핵심 설계 원칙

1. **Simplicity First**
   - Next-token prediction만 사용
   - 복잡한 diffusion 불필요
   - Unified objective

2. **Task-Specific Models**
   - Emu3-Gen: Generation 최적화
   - Emu3-Chat: Understanding 최적화
   - Trade-off: 각 태스크 성능 vs 통합성

3. **Fine-Grained Tokenization**
   - 8x downsample (vs Chameleon 16x)
   - 더 많은 토큰으로 세밀한 표현
   - Token count: 8100 vs 2025

4. **Structured Generation**
   - Prefix-constrained decoding
   - BOI → Resolution → Visual tokens → EOI
   - 구조 보장으로 안정성 확보

5. **Multimodal Extension**
   - 3D Causal VQ for video
   - Temporal + Spatial modeling
   - Causal autoregressive generation

### 장점

1. **순수한 Simplicity**
   - Next-token prediction만
   - 추가 학습 목표 불필요
   - Scaling laws 적용 가능

2. **고품질 생성**
   - SDXL 능가
   - CFG + Prefix-constrained decoding
   - Fine-grained tokenization

3. **Video 지원**
   - 3D Causal VQ
   - Video understanding & generation
   - Video extension (next-frame prediction)

4. **유연한 해상도**
   - 다양한 aspect ratio
   - 고정 image area
   - Token count 조절

5. **Task-Specific 최적화**
   - Emu3-Gen: 최고 품질 생성
   - Emu3-Chat: 우수한 이해 성능

### 한계점

1. **자동 Interleaved 불가능**
   - Two-model approach 필요
   - BOI 토큰 자동 예측 불가
   - Chameleon처럼 seamless 전환 불가

2. **모델 분리**
   - Gen과 Chat 별도 로드 필요
   - 메모리 2배 필요 (동시 사용 시)
   - 통합 interleaved 생성 어려움

3. **Processor 의존성**
   - BOI, EOI 수동 삽입
   - Mode 명시적 지정 필요
   - Prefix template 관리

4. **Token count 증가**
   - 8100 tokens (90x90)
   - Chameleon 2025 대비 4배
   - 생성 시간 증가

### 의의

Emu3는 **"Next-Token Prediction is All You Need"** 명제를 증명합니다:

- **No Diffusion**: Autoregressive만으로 SDXL 능가
- **No CLIP**: Vision encoder 없이 우수한 이해 성능
- **No Pretrained LLM**: From scratch 학습

**Type B1의 순수 구현**으로서, 단순한 objective만으로도 멀티모달 통합이 가능함을 보여줍니다.

**Trade-off**:
- Chameleon: Seamless interleaved, 하지만 품질 제한
- **Emu3**: 최고 품질, 하지만 two-model approach

향후 **Emu3-Unified** 같은 단일 모델이 나온다면, Chameleon의 seamless함과 Emu3의 품질을 모두 가질 수 있을 것입니다.
