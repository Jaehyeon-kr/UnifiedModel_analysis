# Chameleon - Type B1 Single Token

## 목차
1. [개요](#개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [BOI 토큰 예측 메커니즘](#boi-토큰-예측-메커니즘)
4. [자동 Interleaved 생성](#자동-interleaved-생성)
5. [핵심 특징](#핵심-특징)
6. [결론](#결론)

---

## 개요

**Chameleon**은 Meta AI에서 개발한 "Mixed-Modal Early-Fusion Foundation Model"로, **사용자의 명시적 모드 전환 없이 완전히 자동으로** 텍스트와 이미지를 interleaved 방식으로 생성할 수 있는 Type B1 (Single Token) 모델입니다.

**핵심 특징**:
- ✅ BOI (Begin of Image) 토큰을 모델이 직접 예측
- ✅ 완전 자동 모달리티 전환 (텍스트 ↔ 이미지)
- ✅ 디코더 자동 교체 메커니즘
- ✅ True interleaved 멀티모달 생성

**저장소**: `d:\Check_\janus\analysis\repos\type_b1_single_token\chameleon`

---

## 아키텍처 분석

### 1. 토큰 체계

**파일**: `chameleon/inference/vocab.py`

**라인 11-44**: VocabInfo 클래스 정의

```python
class VocabInfo:
    def __init__(self, vocab_map: dict[str, int]):
        self.name2val = vocab_map

        self.bos_id = vocab_map.get("<s>")           # Begin of Sequence
        self.eos_id = vocab_map.get("</s>")          # End of Sequence
        self.boi_id = vocab_map.get("<racm3:break>") # Begin of Image ⭐
        self.eoi_id = vocab_map.get("<eoss>")        # End of Image
        self.pad_id = vocab_map.get("<pad>")         # Padding
        self.eot_id = vocab_map.get("<reserved08706>") # End of Turn

    @property
    def begin_image(self) -> int:
        return self.boi_id  # BOI 토큰 접근자

    @property
    def end_image(self) -> int:
        return self.eoi_id  # EOI 토큰 접근자
```

**핵심**:
- BOI 토큰: `<racm3:break>` - 이미지 생성 시작을 알림
- EOI 토큰: `<eoss>` - 이미지 생성 종료를 알림
- 이미지 토큰: `IMGIMG*` 형식으로 1024개 토큰 사용 (32x32 VQ-VAE)

### 2. 토큰 타입별 분류

**라인 54-74**: 토큰 분류 속성

```python
@cached_property
def image_tokens(self) -> list[int]:
    return sorted(
        [val for name, val in self.name2val.items() if name.startswith("IMGIMG")]
    )

@cached_property
def special_tokens(self) -> list[int]:
    return sorted(
        [
            val
            for name, val in self.name2val.items()
            if name.startswith("<") and name != "<"
        ]
    )

@cached_property
def text_tokens(self) -> list[int]:
    return sorted(
        set(self.all_tokens) - set(self.image_tokens) - set(self.special_tokens)
    )
```

**토큰 분류**:
- 텍스트 토큰: 일반 BPE 토큰
- 이미지 토큰: IMGIMG으로 시작하는 8192개 토큰 (VQ-VAE codebook)
- 특수 토큰: `<`, `</>`로 둘러싸인 제어 토큰

### 3. 디코더 구조

**파일**: `chameleon/inference/chameleon.py`

Chameleon은 **3가지 디코더**를 사용하여 동적으로 전환:

| 디코더 | 역할 | 허용 토큰 | 전환 조건 |
|--------|------|-----------|----------|
| `TextDecoder` | 텍스트 생성 | 텍스트 토큰 + BOI | BOI 예측 시 → ImageDecoder |
| `ImageDecoder` | 이미지 생성 | 이미지 토큰만 | 1024개 생성 후 → TextDecoder |
| `Generator` | 전체 관리 | - | 디코더 교체 오케스트레이션 |

---

## BOI 토큰 예측 메커니즘

### 1. TextDecoder의 허용 토큰 설정

**파일**: `chameleon/inference/chameleon.py`

**라인 251-257**: `_allowed_tokens()` 메서드

```python
def _allowed_tokens(self) -> list[int]:
    allowed_tokens = [self.vocab.eos_id]
    if self.options.txt:
        allowed_tokens += self.vocab.text_tokens  # 텍스트 토큰 허용
    if self.options.img:
        allowed_tokens += [self.vocab.begin_image]  # ⭐ BOI 토큰 허용
    return allowed_tokens
```

**핵심**:
- `options.img=True`일 때, BOI 토큰이 허용 토큰 리스트에 추가됨
- 모델이 텍스트 생성 중 자연스럽게 BOI 토큰을 예측 가능

### 2. Logits Processor 적용

**라인 259-276**: `_logits_processors()` 메서드

```python
def _logits_processors(self) -> list[LogitsProcessor]:
    logits_processors = [
        AllowOnlyTokensLogitsProcessor(self._allowed_tokens()),  # ⭐
    ]
    if isinstance(self.options.img, Options.Image):
        logits_processors += [
            DisallowTokensAtOrAfterIndexLogitsProcessor(
                [self.vocab.begin_image],
                self.options.max_seq_len - 1026,  # 이미지 공간 확보
            ),
        ]
    if isinstance(self.options.txt, Options.Text):
        logits_processors += [
            RepetitionPenaltyLogitsProcessor(self.options.txt.repetition_penalty),
            TemperatureLogitsWarper(self.options.txt.temp),
            TopPLogitsWarper(self.options.txt.top_p),
        ]
    return logits_processors
```

**핵심**:
- `AllowOnlyTokensLogitsProcessor`: 허용된 토큰만 생성 가능하도록 제한
- `DisallowTokensAtOrAfterIndexLogitsProcessor`: 시퀀스 끝부분에서 BOI 금지 (공간 부족 방지)

### 3. BOI 예측 감지 및 전환

**라인 278-286**: TextDecoder의 `__next__()` 메서드

```python
def __next__(self) -> DecodePiece:
    tok = next(self.gen)  # 다음 토큰 생성
    next_decoder = None
    if (
        self.vocab.begin_image not in self.eos_ids  # BOI가 종료 토큰이 아니고
        and (tok.id == self.vocab.begin_image).all()  # ⭐ 예측된 토큰이 BOI라면
    ):
        next_decoder = ImageDecoder  # ImageDecoder로 전환 준비
    return DecodePiece(tok, next_decoder)
```

**동작 흐름**:
1. 텍스트 생성 중 모델이 BOI 토큰 예측
2. BOI 감지 시 `next_decoder = ImageDecoder` 설정
3. Generator가 디코더 교체 수행

---

## 자동 Interleaved 생성

### 1. 이미지 생성 (ImageDecoder)

**파일**: `chameleon/inference/chameleon.py`

**라인 289-371**: ImageDecoder 클래스

```python
class ImageDecoder(Decoder):
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[list[int]],
    ):
        # ... 초기화 코드 ...

        logits_processors = [
            InBatchInstructCFGLogitsProcessor(
                options.img.cfg.guidance_scale_text,
                options.img.cfg.guidance_scale_image,
            ),
            AllowOnlyTokensLogitsProcessor(vocab.image_tokens),  # ⭐ 이미지 토큰만 허용
            TemperatureLogitsWarper(options.img.temp),
            TopPLogitsWarper(options.img.top_p),
        ]

        # BOI 토큰 추가 확인
        for inp in input_ids:
            if inp[-1] != self.vocab.begin_image:
                inp.append(self.vocab.begin_image)  # ⭐ BOI 토큰 보장

        # ... 생성기 설정 ...
        self.gen_count = 0  # 이미지 토큰 카운터
```

**핵심**:
- 이미지 토큰만 생성 가능하도록 제한
- Classifier-Free Guidance (CFG) 적용
- 1024개 토큰 카운팅

### 2. 자동 텍스트 복귀

**라인 356-371**: ImageDecoder의 `__next__()` 메서드

```python
def __next__(self) -> DecodePiece:
    if self.gen_count == 1024:  # ⭐ 1024개 이미지 토큰 생성 완료
        # EOI 토큰 강제 추가
        id = torch.tensor([self.vocab.end_image] * self.batch_size)
        logits = torch.full(
            (self.batch_size, len(self.vocab.all_tokens)), -math.inf
        )
        logits[:, self.vocab.end_image] = 0
        return DecodePiece(
            ChameleonGenerator.Token(id=id, logits=logits),
            TextDecoder,  # ⭐ TextDecoder로 자동 복귀
        )

    tok = next(self.gen)  # 이미지 토큰 생성
    tok.id = tok.id.chunk(3)[0]  # CFG 결과에서 첫 번째만 사용
    self.gen_count += 1
    return DecodePiece(tok, None)
```

**동작 흐름**:
1. 이미지 토큰 1024개 생성 (32x32 VQ-VAE)
2. `gen_count == 1024`일 때 EOI 토큰 추가
3. `next_decoder = TextDecoder` 설정하여 텍스트 모드로 복귀

### 3. 디코더 교체 오케스트레이션

**라인 402-422**: Generator의 `__next__()` 메서드

```python
def __next__(self) -> ChameleonGenerator.Token:
    piece = next(self.dyngen)  # 현재 디코더에서 토큰 생성
    self.generated_token_ids.append(piece.token.id)

    if piece.next_decoder is not None:  # ⭐ 디코더 전환 신호 감지
        if not self.options.txt:
            raise StopIteration  # 이미지 전용 모드면 종료

        # 생성된 토큰들을 입력에 추가
        self.input_ids = [
            old_list + generated
            for old_list, generated in zip(
                self.input_ids, torch.stack(self.generated_token_ids).T.tolist()
            )
        ]
        self.generated_token_ids = []

        # ⭐ 디코더 교체
        self.dyngen.gen = piece.next_decoder(
            self.model,
            self.vocab,
            self.options,
            self.input_ids,
        )
    return piece.token
```

**핵심 메커니즘**:
1. `piece.next_decoder`가 설정되면 디코더 전환 필요
2. 기존 생성 토큰들을 입력 시퀀스에 통합
3. 새 디코더 인스턴스 생성 및 교체
4. 완전히 자동으로 동작 (사용자 개입 불필요)

### 4. 전체 생성 흐름

```
초기화: TextDecoder
    ↓
텍스트 생성 중...
    ↓
BOI 토큰 예측 감지
    ↓
ImageDecoder로 전환
    ↓
이미지 토큰 1024개 생성
    ↓
EOI 토큰 추가
    ↓
TextDecoder로 복귀
    ↓
텍스트 생성 재개...
    ↓
(반복 가능)
```

---

## 핵심 특징

### 1. Single Token 방식

**파일**: `chameleon/inference/vocab.py`

**라인 83-123**: BPE ↔ Image 토큰 변환

```python
class VocabTranslation:
    @cached_property
    def bpe2img(self) -> dict[int, int]:
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

        def remap(old_name: str) -> str:
            return "".join(
                img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1]
            )

        return {
            tok: int(remap(self._vocab.val2name[tok]))
            for tok in self._vocab.image_tokens
        }

    def convert_bpe2img(self, bpe_batch: torch.Tensor) -> torch.Tensor:
        bpe_tok, img_tok = self.bpe2img_search_tensors
        return img_tok[torch.searchsorted(bpe_tok, bpe_batch)]
```

**특징**:
- 단일 토큰 공간에서 텍스트와 이미지 토큰 통합
- `IMGIMG0000` ~ `IMGIMG8191`: 8192개 VQ-VAE codebook
- 토큰 ID 변환을 통해 이미지 디코딩

### 2. Classifier-Free Guidance

**파일**: `chameleon/inference/logits_processor.py`

**라인 312-337**: InBatchInstructCFGLogitsProcessor

```python
class InBatchInstructCFGLogitsProcessor(LogitsProcessor):
    def __init__(self, guidance_scale_text: float, guidance_scale_image: float):
        self.guidance_scale_text = guidance_scale_text
        self.guidance_scale_image = guidance_scale_image

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape=[3*batch, seq-len]
        # logits.shape=[3*batch, vocab]
        (
            full_conditioned_logits,      # 텍스트 + 이미지 조건
            image_conditioned_logits,     # 이미지만 조건
            unconditioned_logits,         # 무조건
        ) = logits.chunk(3)

        mixed_logits = (
            unconditioned_logits
            + self.guidance_scale_image * (image_conditioned_logits - unconditioned_logits)
            + self.guidance_scale_text * (full_conditioned_logits - image_conditioned_logits)
        )
        return mixed_logits.repeat(3, 1)
```

**특징**:
- 3-way CFG: Full-conditioned / Image-only / Unconditioned
- 텍스트와 이미지 guidance를 독립적으로 제어
- 이미지 품질 향상

### 3. 사용 예시

**파일**: `chameleon/inference/examples/multimodal_input.py`

```python
from chameleon.inference.chameleon import ChameleonInferenceModel

model = ChameleonInferenceModel(
    "./data/models/7b/",
    "./data/tokenizer/text_tokenizer.json",
    "./data/tokenizer/vqgan.yaml",
    "./data/tokenizer/vqgan.ckpt",
)

tokens = model.generate(
    prompt_ui=[
        {"type": "image", "value": "file:/path/to/image.jpeg"},
        {"type": "text", "value": "What do you see?"},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ]
)
print(model.decode_text(tokens)[0])
```

**특징**:
- Multimodal input 지원 (이미지 + 텍스트)
- 자동 interleaved 생성
- 명시적 모드 전환 불필요

---

## 결론

### Type B1 (Single Token)의 이상적 구현

Chameleon은 **Type B1 방식의 가장 완성도 높은 구현**입니다:

| 특징 | Chameleon | Janus | 설명 |
|------|-----------|-------|------|
| BOI 예측 | ✅ 자동 | ❌ 불가능 | TextDecoder가 BOI 토큰 예측 가능 |
| Interleaved | ✅ 완전 자동 | ❌ 불가능 | 디코더 자동 전환으로 반복 가능 |
| 모달리티 전환 | ✅ 무한 반복 | ❌ 1회만 | TextDecoder ↔ ImageDecoder 순환 |
| 사용자 개입 | ❌ 불필요 | ✅ 필수 | 모드 선택 없이 자동 생성 |
| 단일 토큰 공간 | ✅ 통합 | ✅ 통합 | 텍스트/이미지 토큰 단일 vocabulary |

### 핵심 설계 원칙

1. **BOI as Regular Token**
   - BOI 토큰을 특수 토큰이 아닌 일반 예측 대상으로 처리
   - Logits processor를 통해 조건부로 허용

2. **Decoder Chaining**
   - TextDecoder와 ImageDecoder를 동적으로 교체
   - `DecodePiece.next_decoder`를 통한 전환 신호

3. **Fixed Image Length**
   - 1024개 토큰으로 고정 (32x32 VQ-VAE)
   - 카운터 기반 자동 종료

4. **Seamless Integration**
   - 생성된 토큰을 즉시 입력에 통합
   - Context 유지하며 모달리티 전환

### 한계점

1. **고정 이미지 크기**
   - 항상 1024 토큰 (256x256 픽셀)
   - 가변 크기 이미지 불가능

2. **시퀀스 길이 제약**
   - `max_seq_len - 1026` 이후 BOI 금지
   - 긴 문서에서 이미지 생성 제한

3. **단방향 생성만**
   - Autoregressive 생성만 가능
   - 이미지 편집/수정 불가능

### 의의

Chameleon은 **"진정한 멀티모달 통합"**의 가능성을 보여줍니다:
- 명시적 모드 전환 없이 자연스러운 interleaved 생성
- Single token space로 텍스트와 이미지의 완전한 통합
- Early-fusion 아키텍처의 실질적 장점 실현

**Type B1 방식의 벤치마크 모델**로서, 향후 멀티모달 모델 설계의 중요한 참고점이 될 것입니다.
