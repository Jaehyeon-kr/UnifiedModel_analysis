# SEED-X - Type B3 Learnable Query

## 목차
1. [개요](#개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [BOI 토큰과 이미지 토큰 체계](#boi-토큰과-이미지-토큰-체계)
4. [AutoImageTokenGenerationProcessor 분석](#autoimagetokengenerationprocessor-분석)
5. [Interleaved 생성 가능성](#interleaved-생성-가능성)
6. [핵심 특징](#핵심-특징)
7. [결론](#결론)

---

## 개요

**SEED-X**는 Tencent ARC Lab에서 개발한 "Unified Multimodal Comprehension and Generation" 모델로, **Learnable Query 방식**을 사용하는 Type B3 모델입니다. Chameleon과 달리 고정된 수의 learnable query tokens을 사용하여 이미지를 표현합니다.

**핵심 특징**:
- ✅ BOI 토큰 예측 가능
- ⚠️ 제한적 Interleaved 생성 (AutoImageTokenGenerationProcessor 의존)
- ✅ Resampler 기반 이미지 임베딩 변환
- ✅ SDXL 기반 De-tokenizer로 고품질 이미지 생성

**저장소**: `d:\Check_\janus\analysis\repos\type_b3_learnable_query\SEED-X`

**논문**: [SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation](https://arxiv.org/abs/2404.14396)

---

## 아키텍처 분석

### 1. 전체 구조

**파일**: `src/models/mllm/seed_x.py`

```
Input Image → Visual Encoder (Qwen-VIT) → Input Resampler
                                              ↓
                                    LLaMA2 (17B) ← Text Tokens
                                              ↓
                                    Output Resampler → De-tokenizer (SDXL)
                                                            ↓
                                                     Generated Image
```

### 2. ContinuousLVLM 클래스

**라인 22-41**: 초기화 코드

```python
class ContinuousLVLM(nn.Module):
    def __init__(self, llm, input_resampler, output_resampler,
                 lm_loss_scale=1.0, rec_loss_scale=1.0,
                 add_patch_pos=False, vit_down=False, mse=False) -> None:
        super().__init__()
        self.llm = llm  # LLaMA2 17B
        self.input_resampler = input_resampler   # 이미지 → LLM 임베딩
        self.output_resampler = output_resampler # LLM 임베딩 → 이미지
        self.lm_loss_scale = lm_loss_scale
        self.rec_loss_scale = rec_loss_scale
        # ...
```

**핵심 컴포넌트**:

| 컴포넌트 | 역할 | 입력 | 출력 |
|---------|------|------|------|
| Visual Encoder | 이미지 인코딩 | Image (448x448) | ViT Features (256 tokens) |
| Input Resampler | 압축 | ViT Features (256) | Query Embeddings (64) |
| LLM | 텍스트-이미지 추론 | Text + Query Embeddings | Hidden States |
| Output Resampler | 확장 | Hidden States (64) | ViT-like Features (256) |
| De-tokenizer | 이미지 생성 | ViT-like Features | Image (1024x1024) |

### 3. Resampler 구조

**Input Resampler**:
- 입력: 256개 ViT 토큰 (from Qwen-VIT)
- 출력: 64개 learnable queries
- 목적: 이미지 정보를 LLM이 처리 가능한 형태로 압축

**Output Resampler**:
- 입력: LLM의 64개 hidden states
- 출력: 256개 ViT-like features
- 목적: LLM 출력을 de-tokenizer가 처리 가능한 형태로 확장

---

## BOI 토큰과 이미지 토큰 체계

### 1. 토큰 정의

**파일**: `src/models/mllm/seed_x.py`

**라인 10-12**: 특수 토큰 정의

```python
BOI_TOKEN = '<img>'      # Begin of Image
EOI_TOKEN = '</img>'     # End of Image
IMG_TOKEN = '<img_{:05d}>'  # Image placeholder tokens: <img_00000> ~ <img_00063>
```

**토큰 체계**:
```
<img><img_00000><img_00001>...<img_00063></img>
 ↑          64개 learnable query tokens        ↑
BOI                                            EOI
```

### 2. 토큰 사용 방식

**파일**: `src/inference/eval_text2img_seed_x.py`

**라인 23, 82-84**: 텍스트-이미지 생성 프롬프트

```python
gen_prompt = '{caption}<img>'  # ⭐ BOI 토큰을 프롬프트에 명시

caption = 'A super math wizard cat, richly textured oil painting.'
prompt = gen_prompt.format_map({'caption': caption})
# 결과: "A super math wizard cat, richly textured oil painting.<img>"

prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
input_ids = torch.tensor([tokenizer.bos_token_id] + prompt_ids).to(device, dtype=torch.long).unsqueeze(0)
```

**핵심**:
- BOI 토큰(`<img>`)을 프롬프트에 **명시적으로 포함**
- 모델이 BOI 토큰을 "예측"하는 것이 아니라, **입력으로 제공**
- AutoImageTokenGenerationProcessor가 BOI 다음 토큰들을 강제 생성

### 3. 이미지 이해 시 토큰 구조

**파일**: `src/inference/eval_img2text_seed_x.py`

**라인 144-147**: 이미지 입력 토큰 생성

```python
patch_length = image_tensor.shape[0]  # Any-resolution patches
image_tokens = ''
for _ in range(patch_length-1):
    # Patch 토큰: <patch><img_00000>...<img_00063></patch>
    image_tokens += BOP_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOP_TOKEN
# 마지막 patch는 <img> 토큰 사용
image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN
```

**Any-Resolution 지원**:
- 이미지를 여러 patch로 분할 (1x1, 1x2, 2x2 등)
- 각 patch마다 64개 토큰 할당
- `<patch>` 토큰으로 중간 patch, `<img>` 토큰으로 전체 이미지 표현

---

## AutoImageTokenGenerationProcessor 분석

### 1. LogitsProcessor 구현

**파일**: `src/models/mllm/generation.py`

**라인 9-31**: AutoImageTokenGenerationProcessor 클래스

```python
class AutoImageTokenGenerationProcessor(LogitsProcessor):

    def __init__(self, tokenizer, num_img_gen_tokens=64) -> None:
        super().__init__()
        # BOI + 64개 이미지 토큰 + EOI 시퀀스 생성
        img_all_token_str = ''.join([BOI_TOKEN] +
                                    [IMG_TOKEN.format(int(item)) for item in range(num_img_gen_tokens)] +
                                    [EOI_TOKEN])
        # 전체 시퀀스를 토큰화: [<img>, <img_00000>, ..., <img_00063>, </img>]
        self.img_ids_list = tokenizer.encode(img_all_token_str, add_special_tokens=False)

    def __call__(self, input_ids, scores):
        bz = input_ids.shape[0]
        for i in range(bz):
            cur_input_id = input_ids[i, -1].item()
            if cur_input_id in self.img_ids_list[:-1]:  # ⭐ 이미지 시퀀스 내부라면
                # 다음 토큰을 강제로 지정
                output_id = self.img_ids_list[self.img_ids_list.index(cur_input_id) + 1]
                scores[i, ..., output_id] = scores[i, ...].max() + 10.
            else:  # 이미지 시퀀스 외부라면
                # 중간 이미지 토큰들을 생성하지 못하도록 차단
                scores[i, ..., torch.tensor(self.img_ids_list[1:]).to(dtype=torch.long)] = 0.0

        return scores
```

### 2. 동작 메커니즘

**자동 시퀀스 생성**:

```
입력: "A cat.<img>"
        ↓
cur_token = <img> → next_token = <img_00000> (강제)
        ↓
cur_token = <img_00000> → next_token = <img_00001> (강제)
        ↓
cur_token = <img_00001> → next_token = <img_00002> (강제)
        ↓
        ... (62번 반복)
        ↓
cur_token = <img_00063> → next_token = </img> (강제)
        ↓
출력: "A cat.<img><img_00000><img_00001>...<img_00063></img>"
```

**핵심 특징**:
1. **Deterministic Generation**: 확률적 샘플링이 아닌 강제 시퀀스
2. **Fixed Length**: 항상 정확히 64개 이미지 토큰 생성
3. **Blocking Mechanism**: 시퀀스 외부에서 이미지 토큰 생성 차단

### 3. 생성 과정

**파일**: `src/models/mllm/seed_x.py`

**라인 130-223**: `generate()` 메서드

```python
def generate(self,
             tokenizer,
             prompt=None,
             input_ids=None,
             image_embeds=None,
             embeds_cmp_mask=None,
             ids_cmp_mask=None,
             logits_processor=None,  # ⭐ AutoImageTokenGenerationProcessor
             num_img_gen_tokens=64,
             temperature=0.7,
             num_beams=1,
             max_new_tokens=120,
             top_p=0.5,
             dtype=torch.float16,
             device='cuda',
             patch_positions=None):
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
        logits_processor.append(
            AutoImageTokenGenerationProcessor(tokenizer=tokenizer, num_img_gen_tokens=num_img_gen_tokens)
        )

    # ... 생성 코드 ...

    output = self.llm.generate(input_ids=input_ids,
                               inputs_embeds=input_embeds,
                               output_hidden_states=True,
                               return_dict_in_generate=True,
                               logits_processor=logits_processor,  # ⭐ 적용
                               **generation_config)
```

**라인 191-216**: 생성 결과 처리

```python
generate_ids = output.sequences[0][input_ids.shape[1]:]
boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

last_hidden_states = torch.cat([hidden_state[-1] for hidden_state in output.hidden_states],
                               dim=1)[0, input_ids.shape[1]:, :]

# EOI 토큰 위치 찾기
eoi_indices = torch.where(generate_ids == eoi_token_id)[0].tolist()
num_gen_imgs = len(eoi_indices)

has_img_output = num_gen_imgs > 0
if has_img_output:
    img_gen_feats = []
    for eoi_idx in eoi_indices:
        # EOI 이전 64개 토큰의 hidden states 추출
        img_gen_feats.append(last_hidden_states[eoi_idx - num_img_gen_tokens:eoi_idx])

    img_gen_feats = torch.stack(img_gen_feats)
    # Output Resampler로 ViT-like features 생성
    img_gen_feat = self.output_resampler(img_gen_feats)  # 64 → 256 features
else:
    img_gen_feat = None
```

**처리 과정**:
1. 생성된 시퀀스에서 EOI 토큰 위치 탐색
2. EOI 이전 64개 토큰의 hidden states 추출
3. Output Resampler로 256개 ViT-like features 생성
4. De-tokenizer (SDXL)로 이미지 생성

---

## Interleaved 생성 가능성

### 1. BOI 토큰 예측 vs 입력

**Chameleon과의 차이**:

| 특징 | Chameleon | SEED-X |
|------|-----------|--------|
| BOI 토큰 | 모델이 예측 | 프롬프트에 포함 |
| 자동 전환 | ✅ TextDecoder → ImageDecoder | ❌ AutoImageTokenGenerationProcessor 의존 |
| 반복 가능 | ✅ 무한 반복 가능 | ⚠️ 제한적 |

**SEED-X의 경우**:
```python
# 텍스트-이미지 생성
gen_prompt = '{caption}<img>'  # BOI를 명시적으로 포함
prompt = gen_prompt.format_map({'caption': caption})
```

**Chameleon의 경우**:
```python
# 모델이 자동으로 BOI 예측
# 프롬프트: "A cat."
# 생성: "A cat.<racm3:break><IMGIMG...>"  # BOI를 모델이 예측
```

### 2. Interleaved 생성 시나리오 분석

**시나리오 1: 텍스트 → 이미지 → 텍스트**

```python
# 가능한 프롬프트
prompt = "Here is a cat<img>"
# 생성 결과:
# "Here is a cat<img><img_00000>...<img_00063></img> and a dog."
#                                                      ↑
#                                              EOI 이후 텍스트 생성 가능
```

**분석**:
- ✅ AutoImageTokenGenerationProcessor는 EOI 이후 비활성화
- ✅ EOI 이후 일반 텍스트 생성 재개 가능
- ✅ 단일 이미지 삽입 후 텍스트 계속 생성 가능

**시나리오 2: 텍스트 → 이미지 → 텍스트 → 이미지**

```python
# 프롬프트에 두 번째 BOI를 어떻게 포함?
prompt = "A cat<img>... and a dog<img>"  # ❌ 사용자가 명시적으로 제공해야 함
```

**문제점**:
- ❌ 모델이 두 번째 BOI 토큰을 자동 예측 불가능
- ❌ AutoImageTokenGenerationProcessor는 이미지 시퀀스 외부에서 `<img_*>` 토큰 차단
- ❌ EOI 이후 BOI를 생성하려면 특수 처리 필요

### 3. AutoImageTokenGenerationProcessor의 제약

**라인 19-30**: Logits Processor 동작

```python
def __call__(self, input_ids, scores):
    bz = input_ids.shape[0]
    for i in range(bz):
        cur_input_id = input_ids[i, -1].item()
        if cur_input_id in self.img_ids_list[:-1]:
            # 이미지 시퀀스 내부: 다음 토큰 강제
            output_id = self.img_ids_list[self.img_ids_list.index(cur_input_id) + 1]
            scores[i, ..., output_id] = scores[i, ...].max() + 10.
        else:
            # ⭐ 이미지 시퀀스 외부: 중간 토큰 차단
            # BUT BOI 토큰(<img>)은 차단하지 않음!
            scores[i, ..., torch.tensor(self.img_ids_list[1:]).to(dtype=torch.long)] = 0.0
```

**핵심 발견**:
- `self.img_ids_list[1:]`: `<img_00000>` ~ `</img>` 토큰만 차단
- `self.img_ids_list[0]` = `<img>` (BOI)는 차단하지 않음
- ✅ 이론적으로 모델이 BOI 토큰을 예측 가능!

### 4. 실제 Interleaved 가능성

**추론 기반 분석**:

1. **BOI 토큰은 차단되지 않음**
   - AutoImageTokenGenerationProcessor는 `<img_*>` 형태의 중간 토큰만 차단
   - `<img>` (BOI)와 `</img>` (EOI)는 정상적으로 생성 가능

2. **모델이 BOI를 학습했는가?**
   - Training 데이터에 interleaved 예제가 포함되었다면 가능
   - 하지만 README와 inference 코드를 보면 대부분 단일 이미지 생성 예제

3. **실제 사용 방식**
   - 공식 예제들은 모두 프롬프트에 BOI를 명시적으로 포함
   - 모델이 자발적으로 BOI를 예측하는 예제 없음

**결론**:
- ⚠️ **기술적으로는 가능하지만, 실용적으로는 제한적**
- 모델이 BOI를 예측하도록 훈련되었는지 불확실
- Chameleon처럼 명시적인 디코더 전환 메커니즘 없음

---

## 핵심 특징

### 1. Learnable Query 방식

**Type B3의 정의**:
- 고정된 수(64개)의 learnable query tokens 사용
- 이미지 토큰이 실제 discrete codebook이 아닌 placeholder
- Resampler를 통해 연속적인 임베딩으로 변환

**장점**:
- 적은 토큰 수로 이미지 표현 (64 vs Chameleon의 1024)
- 유연한 해상도 지원 (Any-Resolution)
- 더 빠른 생성 속도

**단점**:
- AutoImageTokenGenerationProcessor 의존성
- 고정된 토큰 수로 인한 표현력 제약
- Deterministic generation (확률적 샘플링 불가)

### 2. De-tokenizer 아키텍처

**파일**: `src/inference/eval_text2img_seed_x.py`

**라인 59-78**: De-tokenizer 초기화

```python
# SDXL 컴포넌트
noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device_2, dtype=dtype)
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device_2, dtype=dtype)

# Adapter: ViT features → SDXL conditioning
adapter_cfg = OmegaConf.load(adapter_cfg_path)
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device_2, dtype=dtype).eval()

adapter.init_pipe(vae=vae,
                  scheduler=noise_scheduler,
                  visual_encoder=visual_encoder,
                  image_transform=image_transform,
                  discrete_model=discrete_model,
                  dtype=dtype,
                  device=device_2)
```

**라인 90-93**: 이미지 생성

```python
if output['has_img_output']:
    # img_gen_feat: Output Resampler 출력 (256 ViT-like features)
    images = adapter.generate(image_embeds=output['img_gen_feat'].to(device_2),
                             num_inference_steps=50)
    images[0].save(save_path)
```

**De-tokenizer 구조**:
```
LLM Hidden States (64)
    ↓
Output Resampler
    ↓
ViT-like Features (256)
    ↓
SDXL Adapter
    ↓
SDXL UNet (50 diffusion steps)
    ↓
Generated Image (1024x1024)
```

**특징**:
- SDXL 기반으로 고품질 이미지 생성
- ViT features를 diffusion model의 conditioning으로 사용
- Chameleon의 VQ-VAE보다 복잡하지만 품질 향상

### 3. Any-Resolution 지원

**파일**: `src/inference/eval_img2text_seed_x.py`

**라인 58, 126-130**: Resolution grids

```python
resolution_grids = ['1x1', '1x2', '1x3', '2x1', '3x1', '1x4', '4x1', '2x2']
base_resolution = 448

grid_pinpoints = []
for scale in resolution_grids:
    s1, s2 = scale.split('x')
    grid_pinpoints.append([int(s1)*base_resolution, int(s2)*base_resolution])
```

**라인 135-147**: 이미지 처리

```python
image = Image.open(image_path).convert('RGB')
# Any-resolution processing
image_tensor, patch_pos_tensor = process_anyres_image(image, image_transform, grid_pinpoints, base_resolution)

patch_length = image_tensor.shape[0]  # 동적 patch 수
image_tokens = ''
for _ in range(patch_length-1):
    # 각 patch마다 64개 토큰
    image_tokens += BOP_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOP_TOKEN
# 전체 이미지
image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN
```

**지원 해상도**:
- 448x448 (1x1)
- 448x896 (1x2)
- 448x1344 (1x3)
- 896x896 (2x2)
- 등등...

**장점**:
- 고해상도 이미지를 효율적으로 처리
- Patch 단위로 분할하여 각각 64개 토큰 할당
- Position embedding으로 patch 위치 정보 유지

---

## 결론

### Type B3 (Learnable Query)의 특징

SEED-X는 **Learnable Query 방식의 대표적인 구현**입니다:

| 특징 | SEED-X | Chameleon | Janus |
|------|--------|-----------|-------|
| 토큰 방식 | Learnable Query (64) | Discrete VQ (1024) | VQ-VAE (8192) |
| BOI 예측 | ⚠️ 제한적 | ✅ 완전 자동 | ❌ 불가능 |
| Interleaved | ⚠️ 이론적 가능, 실용적 제한 | ✅ 완전 자동 | ❌ 불가능 |
| 이미지 품질 | ✅ SDXL 기반 고품질 | ⚠️ VQ-VAE 기반 | ⚠️ VQ-VAE 기반 |
| 생성 속도 | ✅ 빠름 (64 토큰) | ⚠️ 느림 (1024 토큰) | ⚠️ 느림 (8192 토큰) |
| 해상도 | ✅ Any-Resolution | ❌ 고정 (256x256) | ❌ 고정 (256x256) |

### 핵심 설계 원칙

1. **Placeholder Tokens**
   - 이미지 토큰이 실제 discrete codes가 아닌 placeholder
   - AutoImageTokenGenerationProcessor로 강제 시퀀스 생성
   - Hidden states를 통해 실제 이미지 정보 전달

2. **Resampler Architecture**
   - Input Resampler: 256 ViT features → 64 query embeddings
   - Output Resampler: 64 hidden states → 256 ViT-like features
   - 정보 압축과 확장을 통한 효율성

3. **SDXL De-tokenizer**
   - VQ-VAE 대신 diffusion model 사용
   - ViT features를 conditioning으로 활용
   - 고품질 이미지 생성 (1024x1024)

4. **Any-Resolution Support**
   - 다양한 해상도의 이미지 처리
   - Patch 단위 분할 및 position encoding
   - 각 patch마다 독립적인 64개 토큰

### 장점

1. **효율성**
   - 64개 토큰으로 이미지 표현 (vs 1024)
   - 빠른 생성 속도
   - 메모리 효율적

2. **품질**
   - SDXL 기반 고품질 이미지
   - Any-resolution 지원
   - Flexible image understanding

3. **범용성**
   - 이미지 이해 (VQA, Detection)
   - 이미지 생성 (Text-to-Image)
   - 이미지 편집 (SEED-X-Edit)

### 한계점

1. **AutoImageTokenGenerationProcessor 의존성**
   - Deterministic sequence generation
   - 모델의 자발적 BOI 예측 불확실
   - Interleaved 생성의 제한적 지원

2. **고정된 토큰 수**
   - 항상 64개 토큰 사용
   - 간단한 이미지도 64개 토큰 필요
   - 복잡한 이미지는 표현력 제약

3. **De-tokenizer 복잡도**
   - SDXL 실행 오버헤드 (50 diffusion steps)
   - VQ-VAE보다 느린 이미지 생성
   - 추가 GPU 메모리 필요

### Chameleon과의 비교

**Chameleon의 우위**:
- ✅ 완전 자동 interleaved 생성
- ✅ 명시적 디코더 전환 메커니즘
- ✅ 확률적 샘플링 가능

**SEED-X의 우위**:
- ✅ 훨씬 빠른 생성 속도 (64 vs 1024 토큰)
- ✅ 고품질 이미지 (SDXL vs VQ-VAE)
- ✅ Any-resolution 지원
- ✅ 더 다양한 태스크 지원 (편집, detection 등)

### 의의

SEED-X는 **실용성과 효율성**을 중시한 설계입니다:
- Learnable Query 방식으로 토큰 수 최소화
- SDXL 통합으로 이미지 품질 극대화
- Any-resolution으로 유연성 확보

**Type B3의 대표 모델**로서, Chameleon의 순수성보다는 **실제 응용**에 초점을 맞춘 접근법을 보여줍니다.
