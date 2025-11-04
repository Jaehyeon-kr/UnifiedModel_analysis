# Janus

## 목차
1. [개요](#개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [Understanding vs Generation 모드](#understanding-vs-generation-모드)
4. [CFG와 마지막 토큰](#cfg와-마지막-토큰)
5. [Interleaved Generation의 불가능성](#interleaved-generation의-불가능성)
6. [모드 선택 문제](#모드-선택-문제)
7. [결론](#결론)

---

## 개요

Janus는 "Unified Multimodal Understanding and Generation" 모델로 소개되지만, 실제 아키텍처를 분석하면 **Language Model만 공유하는 Dual-Path 구조**입니다. 이 문서는 코드 근거를 바탕으로 Janus의 실제 구조와 한계를 분석합니다.

---

## 아키텍처 분석

### 1. 실제 공유 구조

**파일**: `janus/models/modeling_vlm.py`

**라인 190-219**: `MultiModalityCausalLM` 클래스 초기화

```python
class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        # Understanding 전용 컴포넌트
        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)  # CLIP encoder

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)  # Image → LLM projection

        # Generation 전용 컴포넌트
        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()  # VQ-VAE decoder

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)  # Token → LLM

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)  # LLM → Token prediction

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        # ✅ 유일하게 공유되는 컴포넌트
        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)
```

### 2. "Unified"의 실체

| 컴포넌트 | Understanding | Generation | 공유 여부 |
|---------|---------------|------------|---------|
| Vision Encoder | `self.vision_model` | - | ❌ |
| Vision Decoder | - | `self.gen_vision_model` | ❌ |
| Input Aligner | `self.aligner` | - | ❌ |
| Output Aligner | - | `self.gen_aligner` | ❌ |
| Generation Head | - | `self.gen_head` | ❌ |
| Token Embedding | - | `self.gen_embed` | ❌ |
| **Language Model** | `self.language_model` | `self.language_model` | ✅ |

**결론**: 전체 파라미터의 약 70-80%를 차지하는 Language Model만 공유하고, 나머지 모든 컴포넌트는 완전히 분리되어 있습니다.

---

## Understanding vs Generation 모드

### 1. Understanding Mode (이미지 → 텍스트)

**파일**: `inference.py`

**라인 36-67**: Understanding 추론 파이프라인

```python
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>\nConvert the formula into latex code.",
        "images": ["images/equation.png"],
    },
    {"role": "Assistant", "content": ""},
]

# 이미지를 CLIP encoder로 인코딩
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# Understanding 경로: vision_model → aligner
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# Language model로 텍스트 생성
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)
```

**파일**: `janus/models/modeling_vlm.py`

**라인 221-260**: `prepare_inputs_embeds` 메서드

```python
def prepare_inputs_embeds(
    self,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    images_seq_mask: torch.LongTensor,
    images_emb_mask: torch.LongTensor,
    **kwargs,
):
    bs, n = pixel_values.shape[0:2]
    images = rearrange(pixel_values, "b n c h w -> (b n) c h w")

    # Understanding 전용 경로
    images_embeds = self.aligner(self.vision_model(images))  # CLIP → aligner

    images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
    images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

    input_ids[input_ids < 0] = 0
    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

    # 이미지 embedding을 텍스트 embedding에 삽입
    inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

    return inputs_embeds
```

### 2. Generation Mode (텍스트 → 이미지)

**파일**: `generation_inference.py`

**라인 37-52**: Generation 프롬프트 준비

```python
conversation = [
    {
        "role": "User",
        "content": "A close-up high-contrast photo of Sydney Opera House...",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
# 핵심: <begin_of_image> 토큰 추가
prompt = sft_format + vl_chat_processor.image_start_tag
```

**라인 55-108**: Generation 추론 파이프라인

```python
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id  # CFG용 masking

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    # 576개 이미지 토큰 순차 생성
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state

        # Generation 전용 head 사용
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        # CFG 적용
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        # 생성된 토큰을 embedding으로 변환 (Generation 전용 경로)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # VQ-VAE decoder로 이미지 복원
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
    )
    # ... 이미지 후처리
```

**파일**: `janus/models/modeling_vlm.py`

**라인 262-263**: `prepare_gen_img_embeds` 메서드

```python
def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
    # Generation 전용 경로: gen_embed → gen_aligner
    return self.gen_aligner(self.gen_embed(image_ids))
```

### 3. 두 모드의 완전한 분리

| 단계 | Understanding | Generation |
|-----|--------------|------------|
| **Input** | `<image_placeholder>` + 실제 이미지 | `<begin_of_image>` 토큰 |
| **Encoder** | `vision_model` (CLIP) | - |
| **Projection** | `aligner` | `gen_embed` → `gen_aligner` |
| **LLM** | ✅ `language_model` (공유) | ✅ `language_model` (공유) |
| **Output Head** | LLM head (텍스트) | `gen_head` (이미지 토큰) |
| **Decoder** | - | `gen_vision_model` (VQ-VAE) |
| **Output** | 텍스트 시퀀스 | 이미지 픽셀 |

---

## CFG와 마지막 토큰

### 1. Classifier-Free Guidance (CFG) 구현

**파일**: `generation_inference.py`

**라인 69-74**: Conditional vs Unconditional 입력 준비

```python
tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
for i in range(parallel_size*2):
    tokens[i, :] = input_ids
    if i % 2 != 0:
        # 홀수 인덱스: 첫 토큰과 마지막 토큰만 유지, 나머지는 패딩
        tokens[i, 1:-1] = vl_chat_processor.pad_id
```

**라인 85-88**: CFG 적용

```python
logits = mmgpt.gen_head(hidden_states[:, -1, :])
logit_cond = logits[0::2, :]      # 짝수 인덱스: full prompt (conditional)
logit_uncond = logits[1::2, :]    # 홀수 인덱스: masked prompt (unconditional)

logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
```

### 2. 마지막 토큰 유지의 의미

**파일**: `janus/models/processing_vlm.py`

**라인 87-111**: 특수 토큰 정의

```python
def __init__(
    self,
    image_processor: VLMImageProcessor,
    tokenizer: LlamaTokenizerFast,
    image_tag: str = "<image_placeholder>",
    image_start_tag: str = "<begin_of_image>",  # ← 마지막 토큰
    image_end_tag: str = "<end_of_image>",
    pad_tag: str = "<｜▁pad▁｜>",
    num_image_tokens: int = 576,
    add_special_token: bool = False,
    sft_format: str = "deepseek",
    mask_prompt: bool = True,
    ignore_id: int = -100,
    **kwargs,
):
    # ...
    self.image_start_tag = image_start_tag
    # ...
```

### 3. 마지막 토큰(`<begin_of_image>`)을 유지하는 이유

```
프롬프트 구조: [BOS] [프롬프트 내용...] [<begin_of_image>]
                 ↑       ↑                    ↑
               유지    패딩으로 가림          유지

Conditional:   [BOS] [cat, in, garden] [<begin_of_image>]
Unconditional: [BOS] [PAD, PAD, PAD]   [<begin_of_image>]
```

**의미**:
- **첫 토큰 (BOS)**: 문장 시작 표시
- **마지막 토큰 (`<begin_of_image>`)**: "지금부터 이미지를 생성해야 한다"는 **구조적 신호**
- **중간 프롬프트**: 무조건부 생성을 위해 마스킹

**효과**:
- **Conditional**: 프롬프트 내용을 알고 이미지 생성
- **Unconditional**: "이미지를 생성해야 한다"는 것만 알고, 내용은 모름
- **CFG**: 두 분포를 결합하여 프롬프트 일치도 향상

---

## Interleaved Generation의 불가능성

### 1. Generation 모드의 고정된 파이프라인

**파일**: `generation_inference.py`

**라인 77-95**: 576개 토큰 고정 생성

```python
generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

# 고정된 루프: 무조건 576개 토큰 생성
for i in range(image_token_num_per_image):  # image_token_num_per_image = 576
    outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, ...)
    hidden_states = outputs.last_hidden_state

    logits = mmgpt.gen_head(hidden_states[:, -1, :])
    # ...
    next_token = torch.multinomial(probs, num_samples=1)
    generated_tokens[:, i] = next_token.squeeze(dim=-1)

    img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
    inputs_embeds = img_embeds.unsqueeze(dim=1)
```

**문제점**:
- `<begin_of_image>` 토큰 이후 **무조건 576개 이미지 토큰 생성**
- 중간에 텍스트 생성으로 전환 불가능
- EOS 토큰이나 조건부 종료 메커니즘 없음
- 한 번 시작하면 반드시 끝까지 실행

### 2. Understanding 모드의 텍스트 전용 생성

**파일**: `inference.py`

**라인 55-64**: LLM의 일반 generate 사용

```python
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,  # 텍스트 토큰만 생성
    do_sample=False,
    use_cache=True,
)
```

**문제점**:
- `language_model.generate()` 사용 → `gen_head`가 호출되지 않음
- 오직 텍스트 토큰만 생성 가능
- 이미지 생성 불가능

### 3. 불가능한 사용 케이스

```
❌ "고양이를 그려줘 [이미지] 이제 개도 그려줘 [이미지]"
   → 멀티턴에서 이미지 생성 불가

❌ "설명: [텍스트] 예시: [이미지] 추가 설명: [텍스트]"
   → 텍스트와 이미지 혼합 출력 불가

❌ "이미지가 필요하면 그려주고, 아니면 텍스트로 설명해줘"
   → 조건부 모달리티 선택 불가
```

### 4. 왜 불가능한가?

**아키텍처적 제약**:

1. **분리된 출력 헤드**
   - 텍스트: `language_model.lm_head` (vocab_size 차원)
   - 이미지: `gen_head` (image_token_size 차원)
   - 단일 forward pass에서 둘 중 하나만 선택 가능

2. **분리된 토큰 공간**
   - 텍스트 토큰: 0 ~ 32,000 (tokenizer vocabulary)
   - 이미지 토큰: 0 ~ 8,192 (VQ-VAE codebook)
   - 통합된 vocabulary 없음

3. **분리된 generation 로직**
   - Understanding: `language_model.generate()` 호출
   - Generation: 커스텀 루프 + `gen_head` 호출
   - 런타임에 전환 불가능

### 5. 비교: 진정한 Unified 모델 (Chameleon)

```python
# Chameleon의 통합 vocabulary
vocabulary = {
    "text_tokens": 0 ~ 65,536,      # 텍스트
    "image_tokens": 65,537 ~ 73,728  # 이미지 (8,192개)
}

# 단일 autoregressive generation
for i in range(max_length):
    logits = model(input_ids)  # 전체 vocabulary에 대한 logits
    next_token = sample(logits)

    if next_token == IMAGE_START:
        # 자연스럽게 이미지 토큰 생성 시작
        continue
    elif next_token == EOS:
        break

    input_ids = torch.cat([input_ids, next_token])
```

---

## 모드 선택 문제

### 1. 현재 해결책: 사용자가 명시적 선택

**파일**: `demo/app_januspro.py`

**라인 175-242**: Gradio UI 구조

```python
# Gradio interface
with gr.Blocks() as demo:
    # Understanding UI
    gr.Markdown(value="# Multimodal Understanding")
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")

    understanding_button = gr.Button("Chat")  # ← Understanding 버튼
    understanding_output = gr.Textbox(label="Response")

    # Generation UI (별도 섹션)
    gr.Markdown(value="# Text-to-Image Generation")

    with gr.Row():
        cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")
        t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")

    prompt_input = gr.Textbox(label="Prompt. (Prompt in more detail can help produce better images!)")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")  # ← Generation 버튼
    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

    # 사용자가 명시적으로 클릭
    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    )

    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature],
        outputs=image_output
    )
```

**결론**: 모델이 자동으로 모드를 선택할 수 없으므로, UI에서 두 개의 버튼을 제공하여 사용자가 명시적으로 선택해야 합니다.

### 2. 문제 상황

```python
# 모호한 요청들
user_input = "고양이 보여줘"
# → 텍스트 설명? 이미지 생성? 🤔

user_input = "아름다운 풍경"
# → 설명? 생성? 🤔

user_input = "이것 좀 만들어줘"
# → 코드? 이미지? 🤔
```

### 3. 가능한 해결책들 (모두 추가 구현 필요)

#### Option 1: Intent Classifier (별도 모델)

```python
# 실제 제품에서 구현해야 하는 코드
def route_request(user_input, has_image):
    # 별도의 분류 모델 필요
    intent = intent_classifier(user_input)

    keywords_generate = ["그려", "생성", "만들어", "draw", "generate", "create"]

    if intent == "understand" and has_image:
        return understanding_mode(user_input, has_image)
    elif intent == "generate" or any(kw in user_input for kw in keywords_generate):
        return generation_mode(user_input)
    else:
        return text_only_mode(user_input)
```

**문제점**:
- 추가 모델 필요 (latency, cost 증가)
- 오분류 시 잘못된 모드 실행
- 키워드 의존 → 취약함

#### Option 2: 명시적 Prefix/Command

```python
# 사용자가 직접 표시
"[TEXT] 고양이에 대해 설명해줘"
"[IMAGE] 고양이를 그려줘"

# 또는 명령어 방식 (Midjourney)
"/imagine 고양이"
"/chat 고양이가 뭐야?"
```

**문제점**:
- 나쁜 UX (사용자가 문법 배워야 함)
- 자연스러운 대화 불가

#### Option 3: Two-stage Processing

```python
# 1단계: 별도 LLM 호출로 판단
decision_prompt = f"이 요청은 이미지 생성 요청인가? 예/아니오\n요청: {user_input}"
decision = llm_call(decision_prompt)

# 2단계: 적절한 모드 실행
if "예" in decision:
    return generation_mode(user_input)
else:
    return understanding_mode(user_input)
```

**문제점**:
- 2배 느림, 비용 2배
- 판단 오류 가능
- 추가 latency

### 4. 비교: 진정한 Unified 모델

**Chameleon/GPT-4o 같은 모델**:

```python
# 모델이 스스로 결정 (단일 forward pass)
user: "고양이를 그려줘"
model: [TEXT_TOKEN: "알겠습니다"]
       [IMAGE_TOKEN_1] [IMAGE_TOKEN_2] ... [IMAGE_TOKEN_576]
       [TEXT_TOKEN: "완성했어요!"]

# 자연스러운 전환
user: "이 이미지 설명해줘"
model: [TEXT_TOKEN: "귀여운"] [TEXT_TOKEN: "고양이가"] ...

user: "비슷한 걸 그려줘"
model: [TEXT_TOKEN: "네"] [IMAGE_TOKEN_1] [IMAGE_TOKEN_2] ...
```

**왜 가능?**
- **단일 토큰 공간**: 텍스트와 이미지 토큰이 하나의 vocabulary
- **단일 디코더**: 하나의 transformer가 다음 토큰 유형 결정
- **Autoregressive**: 매 스텝마다 텍스트/이미지 중 선택

### 5. Janus의 구조적 불가능성

**파일**: `janus/models/modeling_vlm.py` 전체 구조

```python
# 현재 구조상 불가능한 것들:

❌ 모델이 스스로 mode 선택
   → 두 개의 분리된 경로, 런타임 전환 불가

❌ 대화 중 자연스러운 전환
   → generate() 호출 전에 모드 고정

❌ "필요하면 이미지, 아니면 텍스트" 같은 유연한 응답
   → 출력 형태가 inference 시작 전에 결정됨

❌ 단일 응답에서 텍스트 + 이미지 혼합
   → 각 모드가 단일 modality만 출력
```

---

## 결론

### 1. "Unified"의 실체

| 주장 | 실제 |
|-----|------|
| "Unified Multimodal Model" | Language Model만 공유하는 Dual-Path 구조 |
| "Single Model" | 하나의 체크포인트에 두 개의 파이프라인 패킹 |
| "Flexible Multimodal" | 사용자가 명시적으로 모드 선택 필요 |

### 2. 핵심 코드 근거 요약

| 주장 | 근거 파일 | 라인 | 내용 |
|-----|---------|------|------|
| LLM만 공유 | `modeling_vlm.py` | 190-219 | Understanding/Generation 전용 컴포넌트 분리 |
| 모드 완전 분리 | `inference.py` | 52-64 | Understanding: `prepare_inputs_embeds` |
| | `generation_inference.py` | 77-108 | Generation: `gen_head` + 커스텀 루프 |
| CFG 마지막 토큰 | `generation_inference.py` | 69-74 | `tokens[i, 1:-1] = pad_id` |
| Interleaved 불가 | `generation_inference.py` | 77 | `for i in range(576)`: 고정 루프 |
| 모드 선택 불가 | `app_januspro.py` | 232-242 | 두 개의 버튼으로 사용자 선택 |

### 3. 장단점

**장점** ✅:
- 각 task에 최적화된 성능 (Understanding, Generation 모두 SOTA급)
- 학습 안정성 (두 모드가 서로 간섭하지 않음)
- 빠른 수렴
- 구현 단순성

**단점** ❌:
- **Interleaved generation 불가능**
- **모델이 스스로 출력 modality 선택 불가**
- 파라미터 비효율 (encoder/decoder 중복)
- 모달리티 간 깊은 상호작용 부족
- 멀티턴 대화에서 유연성 부족

### 4. 진정한 Unified vs Janus

| 특성 | Janus | Chameleon/GPT-4o |
|-----|-------|------------------|
| 토큰 공간 | 분리 (텍스트 / 이미지) | 통합 |
| 디코더 | 이중 경로 | 단일 경로 |
| 출력 헤드 | 분리 (`lm_head` / `gen_head`) | 통합 |
| 모드 선택 | 사용자/외부 시스템 | 모델 자체 |
| Interleaved | ❌ | ✅ |
| 구현 난이도 | 낮음 | 높음 |
| 성능 최적화 | 쉬움 | 어려움 |

### 5. 최종 평가

Janus는:
```
"Language Model을 공유하는 두 개의 독립적 모델"
≠ "진정한 Unified Multimodal Model"
```

**기술적으로 정확한 표현**:
- ✅ "Dual-Path Multimodal Model with Shared LLM Backbone"
- ✅ "Multitask Multimodal Model with Unified Language Representation"
- ❌ "Unified Multimodal Understanding and Generation Model"

**실용적 의미**:
- 연구/벤치마크: 매우 우수한 성능
- 실제 제품: 추가 라우팅 로직 필요
- 유연한 대화형 AI: 근본적 한계 존재

---

## 참고 자료

### 주요 분석 파일
- `janus/models/modeling_vlm.py`: 핵심 아키텍처
- `generation_inference.py`: Generation 모드 추론
- `inference.py`: Understanding 모드 추론
- `demo/app_januspro.py`: 실제 배포 예시
- `janus/models/processing_vlm.py`: 토큰 처리 로직

### 비교 대상 모델
- **Chameleon**: 진정한 unified token space
- **Emu2**: Unified autoregressive framework
- **Transfusion**: Diffusion + LLM 하이브리드
- **SEED-X**: Unified embedding space

---

**작성 일자**: 2025-11-03
**분석 대상**: Janus-1.3B / Janus-Pro-7B 코드베이스
