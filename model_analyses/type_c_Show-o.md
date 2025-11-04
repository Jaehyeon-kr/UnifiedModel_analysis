# Show-o - Type C Fused Approach

## 목차
1. [개요](#개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [특수 토큰 체계](#특수-토큰-체계)
4. [Discrete Diffusion + Autoregressive](#discrete-diffusion--autoregressive)
5. [BOI 토큰과 태스크 토큰](#boi-토큰과-태스크-토큰)
6. [Interleaved 생성 가능성](#interleaved-생성-가능성)
7. [핵심 특징](#핵심-특징)
8. [결론](#결론)

---

## 개요

**Show-o**는 NUS Show Lab에서 개발한 "One Single Transformer to Unify Multimodal Understanding and Generation" 모델로, **단일 Transformer**에서 텍스트는 autoregressive, 이미지는 discrete diffusion 방식으로 처리하는 Type C (Fused) 모델입니다.

**핵심 특징**:
- ✅ 단일 Transformer에서 텍스트와 이미지 통합
- ✅ Discrete Diffusion (MaskGIT) + Autoregressive 결합
- ✅ 태스크 토큰으로 모드 전환
- ❌ BOI 토큰 자동 예측 불가능
- ❌ 완전 자동 Interleaved 생성 불가능

**저장소**: `d:\Check_\janus\analysis\repos\type_c_fused\Show-o`

**논문**: [Show-o: One Single Transformer to Unify Multimodal Understanding and Generation](https://arxiv.org/abs/2408.12528)

**모델 크기**: 1.3B (기반 LLM: Phi-1.5)

---

## 아키텍처 분석

### 1. Type C - Fused Approach

**특징 비교표** (Show-o README에서 발췌):

| Model Type | Vision Representation | Language Representation | Diffusion |
|-----------|---------------------|----------------------|-----------|
| Understanding Only | Vision Encoder | LLM | ❌ |
| Generation Only | - | LLM | ✅ |
| **Unified (Show-o)** | **VQ Tokenizer** | **LLM** | **✅** |

**핵심**:
- 텍스트와 이미지를 모두 discrete tokens로 표현
- **단일 Transformer**에서 통합 처리
- Attention mechanism 구분:
  - 텍스트: Causal attention (autoregressive)
  - 이미지: Full attention (diffusion)

### 2. 전체 구조

**파일**: `models/modeling_showo.py`

```
Input (Text + Image)
    ↓
VQ Tokenizer (MAGVITv2) → Image Tokens (256)
Text Tokenizer → Text Tokens
    ↓
Unified Sequence: [Task Token] [Text] [SOI] [Image] [EOI]
    ↓
Phi-1.5 Transformer (Unified Backbone)
    ├── Causal Attention (for text tokens)
    └── Full Attention (for image tokens)
    ↓
Output Generation
    ├── Text: Autoregressive sampling
    └── Image: MaskGIT discrete diffusion
```

### 3. Showo 클래스

**라인 23-55**: 모델 초기화

```python
class Showo(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            w_clip_vit,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_showo=True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)

        # Phi-1.5 기반 Transformer
        if load_from_showo:
            config = AutoConfig.from_pretrained(llm_model_path)
            self.showo = PhiForCausalLM(config)
        else:
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')

        # Vocabulary 확장: LLM vocab + VQ codes + special tokens
        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size

        # Optional: CLIP-ViT 통합
        if self.w_clip_vit:
            self.mm_projector = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )
```

**핵심**:
- Phi-1.5 (1.3B) 기반
- Vocabulary = LLM tokens + VQ image tokens + special tokens
- CLIP-ViT를 선택적으로 통합 가능

---

## 특수 토큰 체계

### 1. 토큰 정의

**파일**: `training/prompting_utils.py`

**라인 20-32**: UniversalPrompting 클래스

```python
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                                 "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                 max_text_len=8000, max_seq_len=377, ignore_id=-100, cond_dropout_prob=0.1):
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_tokenizer.add_tokens(list(special_tokens))

        # Special token ID 매핑
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token]))
                           for token in special_tokens}
        self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
```

**특수 토큰 분류**:

| 토큰 | 용도 | 비고 |
|------|------|------|
| `<|sot|>` | Start of Text | = BOS token |
| `<|eot|>` | End of Text | = EOS token |
| `<|soi|>` | Start of Image | Begin of Image ⭐ |
| `<|eoi|>` | End of Image | End of Image |
| `<|sov|>` | Start of Video | Video 지원 (Show-o2) |
| `<|eov|>` | End of Video | Video 지원 (Show-o2) |
| `<|t2i|>` | Text-to-Image Task | 태스크 토큰 |
| `<|mmu|>` | Multimodal Understanding Task | 태스크 토큰 |
| `<|t2v|>` | Text-to-Video Task | Show-o2 |
| `<|v2v|>` | Video-to-Video Task | Show-o2 |
| `<|lvg|>` | Long Video Generation Task | Show-o2 |

### 2. 프롬프트 형식

**Text-to-Image (T2I)**:

**라인 39-90**: `t2i_prompt()` 메서드

```python
def t2i_prompt(self, text_ids, image_ids, labels):
    # ...
    # Prompt 형식: [task] [text] [SOI] [image] [EOI]
    temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

    # Full sequence:
    temp_ids = torch.cat([
        torch.tensor(temp_ids).to(device),         # [<|t2i|>] [text] [<|eot|>]
        self.sptids_dict['<|soi|>'].to(device),    # [<|soi|>]
        image_ids[i],                               # [image tokens]
        self.sptids_dict['<|eoi|>'].to(device)     # [<|eoi|>]
    ], dim=0)
```

**Multimodal Understanding (MMU)**:

**라인 162-200**: `mmu_prompt()` 메서드

```python
def mmu_prompt(self, image_ids, text_ids):
    # ...
    # Prompt 형식: [task] [SOI] [image] [EOI] [text]
    temp_ids = torch.cat([
        self.sptids_dict['<|mmu|>'].to(device),    # [<|mmu|>]
        self.sptids_dict['<|soi|>'].to(device),    # [<|soi|>]
        image_ids[i],                               # [image tokens]
        self.sptids_dict['<|eoi|>'].to(device),    # [<|eoi|>]
        torch.tensor(temp_ids).to(device),         # [text tokens]
    ], dim=0)
```

**형식 비교**:

| Task | Sequence Format |
|------|----------------|
| Text-to-Image | `[<|t2i|>] [text] [<|eot|>] [<|soi|>] [image] [<|eoi|>]` |
| Multimodal Understanding | `[<|mmu|>] [<|soi|>] [image] [<|eoi|>] [text]` |
| Language Modeling | `[text]` (태스크 토큰 없음) |

---

## Discrete Diffusion + Autoregressive

### 1. Forward Pass

**파일**: `models/modeling_showo.py`

**라인 59-102**: `forward()` 메서드

```python
def forward(
        self,
        input_ids,
        input_embeddings=None,
        attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        max_seq_length=128,
        labels_mask_text=None,
        labels_mask_image=None,
        **kwargs,
):
    # Unified forward pass
    if input_embeddings is None:
        logits = self.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
    else:
        logits = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

    if labels is not None:
        # 1. Mask token prediction (discrete diffusion) for T2I
        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
        )

        # 2. Next token prediction for Language Modeling
        loss_lm = F.cross_entropy(
            logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
            labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
        )

        # 3. Next token prediction for MMU
        loss_mmu = F.cross_entropy(
            logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
            labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
        )

        return logits, loss_t2i, loss_lm, loss_mmu

    return logits
```

**핵심**:
- **단일 forward pass**에서 3가지 손실 계산
- Batch를 task별로 분할 (`batch_size_t2i`, `batch_size_lm`, `batch_size_mmu`)
- T2I: Mask token prediction (diffusion)
- LM/MMU: Next token prediction (autoregressive)

### 2. Text-to-Image 생성 (MaskGIT)

**라인 104-181**: `t2i_generate()` 메서드

```python
def t2i_generate(
        self,
        input_ids: torch.LongTensor = None,
        uncond_input_ids: torch.LongTensor = None,
        attention_mask=None,
        temperature=1.0,
        timesteps=18,  # MaskGIT paper의 권장값
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        generator: torch.Generator = None,
        config=None,
        **kwargs,
):
    """
    MaskGIT 방식의 discrete diffusion 생성
    https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
    """
    mask_token_id = self.config.mask_token_id
    num_vq_tokens = config.model.showo.num_vq_tokens  # 256

    # 초기: 모든 이미지 토큰이 [MASK]
    input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()

    # Iterative decoding
    for step in range(timesteps):
        # Classifier-Free Guidance
        if uncond_input_ids is not None and guidance_scale > 0:
            uncond_input_ids = torch.cat(
                [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
            model_input = torch.cat([input_ids, uncond_input_ids])
            cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
            # CFG formula (Muse 스타일)
            logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
        else:
            logits = self(input_ids, attention_mask=attention_mask)

        # Multinomial sampling
        probs = logits.softmax(dim=-1)
        sampled_ids = torch.multinomial(probs.reshape(-1, logits.size(-1)), 1, generator=generator)

        # Mask scheduling
        ratio = 1.0 * (step + 1) / timesteps
        mask_ratio = noise_schedule(torch.tensor(ratio))

        # Confidence-based masking (낮은 확률의 토큰 다시 마스킹)
        selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)
        mask_len = (num_vq_tokens * mask_ratio).floor()
        masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

        # Update input for next iteration
        input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(
            masking, mask_token_id, sampled_ids + llm_vocab_size + num_new_special_tokens
        )

    return sampled_ids
```

**MaskGIT 프로세스**:

```
Step 0: [MASK] [MASK] [MASK] ... [MASK] (256개 모두 마스크)
    ↓
Step 1: [123] [MASK] [456] ... [MASK] (일부 샘플링, 낮은 확률은 다시 마스크)
    ↓
Step 2: [123] [789] [456] ... [MASK] (점진적으로 마스크 감소)
    ↓
    ...
    ↓
Step 18: [123] [789] [456] ... [012] (모든 토큰 확정)
```

### 3. Multimodal Understanding 생성

**라인 183-240**: `mmu_generate()` 메서드

```python
@torch.no_grad()
def mmu_generate(self, idx=None, input_embeddings=None, attention_mask=None,
                 max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
    """
    Autoregressive text generation for MMU
    """
    result = []
    for _ in range(max_new_tokens):
        # Forward pass
        logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)

        # Attention mask 업데이트 (새 토큰 추가)
        L = attention_mask.shape[-1]
        attention_mask = attention_mask.squeeze()
        attention_mask_a = torch.hstack([
            attention_mask,  # L, L
            torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
        ])
        attention_mask_b = torch.vstack([
            attention_mask_a,  # L, L+1
            torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
        ])
        attention_mask = attention_mask_b

        # Next token sampling
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        result.append(idx_next[0][0])

        # Update input
        if self.config.w_clip_vit:
            idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
            input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
        else:
            idx = torch.cat((idx, idx_next), dim=1)

        # EOT 검사
        if eot_token is not None and idx_next.cpu() == eot_token:
            break

    return result
```

**핵심**:
- 일반적인 autoregressive LM과 동일
- Temperature, top-k sampling 지원
- EOT 토큰으로 종료

---

## BOI 토큰과 태스크 토큰

### 1. BOI 토큰은 자동 예측되는가?

**분석**:

**T2I 프롬프트 형식** (라인 53, 78-83):
```python
# 사용자가 명시적으로 제공하는 형식
temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

temp_ids = torch.cat([
    torch.tensor(temp_ids).to(device),         # [<|t2i|>] [text] [<|eot|>]
    self.sptids_dict['<|soi|>'].to(device),    # 항상 추가됨
    image_ids[i],
    self.sptids_dict['<|eoi|>'].to(device)
], dim=0)
```

**핵심 발견**:
- `<|soi|>` 토큰이 **항상 코드에서 추가됨**
- 모델이 예측하는 것이 아니라, **입력 시퀀스 구성 시 자동 삽입**
- T2I 태스크 토큰 `<|t2i|>`도 마찬가지

**결론**: ❌ **BOI 토큰 자동 예측 불가능**

### 2. 태스크 토큰의 역할

**태스크 전환 메커니즘**:

```python
# T2I 모드
input_ids = "[<|t2i|>] [prompt text] [<|soi|>] [MASK...MASK] [<|eoi|>]"
→ model.t2i_generate() 호출 → MaskGIT 방식으로 이미지 생성

# MMU 모드
input_ids = "[<|mmu|>] [<|soi|>] [image tokens] [<|eoi|>] [question]"
→ model.mmu_generate() 호출 → Autoregressive 텍스트 생성
```

**핵심**:
- 태스크 토큰은 **메타 정보**로 사용
- 실제 생성 메서드 선택은 **코드 레벨에서 결정**
- 모델이 태스크를 "선택"하는 것이 아님

---

## Interleaved 생성 가능성

### 1. 단일 태스크 제약

**파일**: `inference_t2i.py`, `inference_mmu.py`

**T2I 추론** (라인 115):
```python
input_ids, _ = uni_prompting((prompt, inpainting_image_tokens), 't2i_gen')
# → 단일 T2I 태스크만 수행
```

**MMU 추론** (라인 100-104):
```python
conv = conversation_lib.default_conversation.copy()
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
# → 단일 MMU 태스크만 수행
```

**핵심**:
- 한 번의 `generate()` 호출로 하나의 태스크만 수행
- Interleaved 생성을 위해서는 **여러 번의 추론 필요**

### 2. Interleaved 시나리오 분석

**시나리오: 텍스트 → 이미지 → 텍스트**

```python
# Step 1: Text generation (LM mode)
input_1 = "[text_prefix]"
output_1 = model.generate(...) → "Here is a cat"

# Step 2: Image generation (T2I mode)
prompt_2 = "[<|t2i|>] Here is a cat [<|soi|>] [MASK...] [<|eoi|>]"
output_2 = model.t2i_generate(...) → image_tokens

# Step 3: Text generation (MMU mode)
prompt_3 = "[<|mmu|>] [<|soi|>] [image_tokens] [<|eoi|>] What is this?"
output_3 = model.mmu_generate(...) → "This is a cat"
```

**문제점**:
1. **수동 orchestration 필요**: 사용자가 각 단계를 명시적으로 호출
2. **태스크 토큰 수동 삽입**: `<|t2i|>`, `<|mmu|>` 등을 코드에서 추가
3. **컨텍스트 유지 어려움**: 이전 출력을 다음 입력으로 수동 연결
4. **모델이 자발적으로 전환 불가**: Chameleon처럼 BOI 예측으로 전환되지 않음

**결론**: ❌ **완전 자동 Interleaved 생성 불가능**

### 3. Chameleon과의 비교

| 특징 | Show-o | Chameleon |
|------|--------|-----------|
| BOI 토큰 | 코드에서 삽입 | 모델이 예측 |
| 태스크 전환 | 수동 (태스크 토큰) | 자동 (디코더 전환) |
| Interleaved | 다중 추론 필요 | 단일 추론 가능 |
| 생성 방식 | Discrete Diffusion (이미지) | VQ-VAE (이미지) |
| 제어성 | ✅ 높음 (태스크 명시) | ⚠️ 낮음 (모델 의존) |

---

## 핵심 특징

### 1. Unified Single Transformer

**장점**:
- 텍스트와 이미지를 단일 모델에서 처리
- 파라미터 공유로 효율성 향상
- Multi-task learning으로 상호 보완

**설계 원칙**:
- **Fused Representation**: 텍스트와 이미지 모두 discrete tokens
- **Heterogeneous Attention**: Causal (text) + Full (image)
- **Hybrid Objective**: Autoregressive + Diffusion

### 2. Discrete Diffusion (MaskGIT)

**파일**: `models/sampling.py`

**Cosine Schedule**:
```python
def cosine_schedule(t):
    """
    Cosine noise schedule for MaskGIT
    t: timestep ratio (0 ~ 1)
    """
    return torch.cos(t * torch.pi * 0.5)
```

**장점**:
- VQ-VAE 기반이므로 이미지 품질 제한적
- BUT, 병렬 생성 가능 (vs autoregressive)
- 18 steps만으로 512x512 이미지 생성

**비교**:

| 모델 | 이미지 생성 방식 | Steps | 품질 |
|------|----------------|-------|------|
| Show-o | Discrete Diffusion (MaskGIT) | 18 | ⚠️ VQ-VAE 제한 |
| SEED-X | Diffusion (SDXL) | 50 | ✅ 고품질 |
| Chameleon | Autoregressive | 1024 | ⚠️ VQ-VAE 제한 |

### 3. MAGVITv2 Tokenizer

**특징**:
- VQ-VAE 기반 image tokenizer
- Codebook size: 8192
- Image resolution: 256x256 → 256 tokens (16x16)
- 512x512도 지원 (업데이트된 버전)

**Trade-off**:
- ✅ Discrete representation으로 LLM과 통합 용이
- ❌ VQ-VAE의 재구성 품질 제한
- ❌ SDXL 같은 diffusion model보다 품질 낮음

### 4. Classifier-Free Guidance

**라인 136-143**:
```python
if uncond_input_ids is not None and guidance_scale > 0:
    uncond_input_ids = torch.cat(
        [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
    model_input = torch.cat([input_ids, uncond_input_ids])
    cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
    # Muse 스타일 CFG
    logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
```

**특징**:
- T2I 생성 시 CFG 적용
- Muse 스타일 formula: `(1 + s) * cond - s * uncond`
- 권장 guidance_scale: 1.75 ~ 5.0

### 5. Show-o2 (후속 모델)

**개선 사항**:
- 3D Causal VAE 도입 (텍스트, 이미지, **비디오** 통합)
- Dual-path spatial(-temporal) fusion
- Flow matching 추가
- 1.5B, 7B 모델 제공
- Video understanding & generation 지원

**구조**:
```
Show-o: 2D (Text + Image)
    ↓
Show-o2: 3D (Text + Image + Video)
- Autoregressive modeling (text/image understanding)
- Flow matching (image/video generation)
```

---

## 결론

### Type C (Fused)의 대표 모델

Show-o는 **단일 Transformer로 통합**하는 Type C 방식의 대표적인 구현입니다:

| 특징 | Show-o | Chameleon | SEED-X | Janus |
|------|--------|-----------|--------|-------|
| 타입 | C Fused | B1 Single Token | B3 Learnable Query | B2 Separate |
| 통합 방식 | Single Transformer | Single Transformer | LLM + Resampler | Dual-Path |
| 이미지 생성 | Discrete Diffusion | Autoregressive | SDXL Diffusion | VQ-VAE |
| BOI 예측 | ❌ 수동 삽입 | ✅ 자동 예측 | ⚠️ 제한적 | ❌ 불가능 |
| Interleaved | ❌ 다중 추론 | ✅ 완전 자동 | ⚠️ 제한적 | ❌ 불가능 |
| 태스크 제어 | ✅ 명시적 | ❌ 암묵적 | ⚠️ Processor | ✅ 모드 선택 |

### 핵심 설계 원칙

1. **Unified Token Space**
   - LLM vocabulary + VQ codes + special tokens
   - 단일 embedding layer에서 처리
   - Vocabulary size ≈ 51,200 (32,000 LLM + 8,192 VQ + special)

2. **Task Token Routing**
   - `<|t2i|>`, `<|mmu|>` 등의 태스크 토큰으로 모드 명시
   - 코드 레벨에서 생성 메서드 선택
   - 명시적 제어 가능

3. **Hybrid Attention**
   - 텍스트: Causal attention (autoregressive dependencies)
   - 이미지: Full attention (bidirectional for diffusion)
   - Attention mask로 구분

4. **Discrete Diffusion**
   - MaskGIT 방식의 iterative refinement
   - 병렬 생성 가능 (vs 1024 steps autoregressive)
   - CFG로 품질 향상

### 장점

1. **명시적 제어**
   - 태스크 토큰으로 명확한 모드 선택
   - Janus보다 사용자 친화적

2. **효율성**
   - 단일 모델로 다양한 태스크 수행
   - MaskGIT 방식으로 빠른 이미지 생성 (18 steps)

3. **확장성**
   - Show-o2에서 비디오까지 확장
   - 3D VAE로 temporal 차원 추가

4. **학습 효율**
   - Multi-task learning으로 상호 보완
   - 단일 loss로 통합 학습

### 한계점

1. **자동 Interleaved 불가능**
   - 태스크 토큰을 수동으로 삽입해야 함
   - Chameleon처럼 자연스러운 모달리티 전환 불가

2. **VQ-VAE 품질 제한**
   - Discrete tokenization의 정보 손실
   - SDXL 같은 diffusion model보다 품질 낮음

3. **복잡한 프롬프트 구성**
   - 특수 토큰 시퀀스를 정확히 구성해야 함
   - 사용자 실수 가능성

4. **고정된 태스크 세트**
   - 사전 정의된 태스크만 지원
   - 새로운 태스크 추가 시 재학습 필요

### 의의

Show-o는 **"진정한 통합"과 "실용적 제어" 사이의 균형**을 추구합니다:

- Chameleon: 완전 자동, 하지만 제어 어려움
- Janus: 명시적 모드 선택, 하지만 interleaved 불가
- **Show-o**: 태스크 토큰으로 명시적 제어, 단일 모델로 통합

**Type C Fused 방식의 벤치마크**로서, 향후 멀티모달 통합 모델 설계에 중요한 참고점이 될 것입니다. 특히 Show-o2의 비디오 확장은 **시공간 통합 모델**의 가능성을 보여줍니다.
