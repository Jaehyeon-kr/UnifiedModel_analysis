# UniDisc의 Interleaved 구조 — 코드 기반 상세 분석 (BOI 토큰 근거 포함)

## 1. 개요

UniDisc는 텍스트와 이미지를 하나의 시퀀스 안에서 번갈아(interleaved) 처리할 수 있도록 설계되어 있다.
이 구조의 핵심은 `InterleavedBatch`와 `InterleavedElement` 두 클래스이며,
이들이 데이터 차원에서 “텍스트 → 이미지 → 텍스트” 순서를 유지하면서 모델이 이를 학습하고 생성할 수 있게 한다.

---

## 2. `InterleavedBatch`: 교차 시퀀스 단위 관리

### 정의

```python
@tensorclass
class InterleavedBatch:
    input_ids: torch.Tensor
    modality: torch.Tensor
    sample_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
```

* `input_ids`: 텍스트와 이미지 토큰이 섞인 전체 시퀀스
* `modality`: 각 토큰이 텍스트(0)인지 이미지(1)인지 구분
* `sample_ids`: 동일한 샘플 내 구간을 표시
* `attention_mask`: 학습 중 활성화할 토큰 위치를 명시 (선택적)

이 클래스는 interleaved된 데이터를 관리하고, 각 샘플을 독립적으로 분리하여 처리할 수 있도록 한다.

---

### 2.1 data_defs.py `to_ragged_batch()`

```python
batch_indices, start_positions, end_positions = get_contiguous_blocks(self.sample_ids)
for i in range(batch_indices.shape[0]):
    data.append(self[batch_indices[i], start_positions[i]:end_positions[i]])
```

* `get_contiguous_blocks()`를 통해 동일한 `sample_id`를 갖는 연속된 블록을 추출
* 결과적으로 interleaved된 데이터를 “샘플 단위”로 분할
* 이렇게 분리된 데이터는 이후 `TensorDict.lazy_stack()`을 통해 배치로 재조합된다

→ 즉, **interleaved 시퀀스를 샘플 단위로 정리하여 모델 입력으로 사용할 수 있는 형태로 만든다.**

---

### 2.2  data_defs.py  `to_elements()`

```python
data = self.to_ragged_batch()
new_data = []
for i in range(data.shape[0]):
    new_data.append(InterleavedElement.from_raw(data[i]))
```

* `to_ragged_batch()`로 얻은 배치를 각 샘플별로 `InterleavedElement` 객체로 변환
* 이렇게 변환된 데이터는 텍스트/이미지 구간을 독립적으로 다룰 수 있다

---

## 3. `InterleavedElement`: interleaved 데이터 복원

### 정의

```python
@tensorclass
class InterleavedElement:
    txt_input_ids: Optional[list[torch.Tensor]] = None
    img_input_ids: Optional[list[torch.Tensor]] = None
    img_pos_ids: Optional[torch.Tensor] = None
```

* `txt_input_ids`: 텍스트 블록 리스트
* `img_input_ids`: 이미지 블록 리스트
* `img_pos_ids`: 각 이미지 블록이 어떤 텍스트 블록 뒤에 붙는지 인덱스로 기록

이 클래스는 실제 interleaved 시퀀스(텍스트 ↔ 이미지)를 복원할 수 있는 구조를 제공한다.

---

### 3.1  data_defs.py `from_raw()`

```python
batch_indices, start_positions, end_positions = get_contiguous_blocks(interleaved_batch.modality[None])
block_modality = interleaved_batch.modality[start_positions]

for i in range(batch_indices.shape[0]):
    if block_modality[i] == 1:
        img_input_ids.append(interleaved_batch.input_ids[start_positions[i]:end_positions[i]])
        img_pos_ids.append(len(txt_input_ids) - 1)
    else:
        txt_input_ids.append(interleaved_batch.input_ids[start_positions[i]:end_positions[i]])
```

#### 동작 과정:

1. `get_contiguous_blocks()`로 텍스트/이미지 구간을 나눈다.
2. `block_modality[i] == 1`이면 이미지 블록으로 인식.
3. `img_pos_ids.append(len(txt_input_ids) - 1)`를 통해
   해당 이미지가 **어느 텍스트 블록 뒤에 붙는지** 명시적으로 기록한다.
4. 텍스트와 이미지 블록이 교차 구조로 저장된다.

이 구조는 모델이 “텍스트 다음에 오는 이미지”의 문맥을 학습할 수 있도록 한다.

---

### 3.2 data_defs.py `to_list()`

```python
while txt_idx < len(self.txt_input_ids) or img_idx < len(self.img_input_ids):
    if not has_added_txt and txt_idx < len(self.txt_input_ids):
        data.append(self.txt_input_ids[txt_idx])
        modalities.append(0)
        has_added_txt = True
    elif img_idx < len(self.img_input_ids) and self.img_pos_ids[img_idx] == txt_idx:
        data.append(self.img_input_ids[img_idx])
        modalities.append(1)
        img_idx += 1
    else:
        has_added_txt = False
        txt_idx += 1
```

#### 주요 특징:

* 텍스트 블록을 먼저 추가 (`modality = 0`)
* `img_pos_ids`가 현재 텍스트 인덱스와 일치하면 이미지 블록을 이어 붙임 (`modality = 1`)
* 다음 텍스트 블록으로 넘어가며 **텍스트 → 이미지 → 텍스트 → 이미지** 순서가 형성됨

이 함수는 **interleaved 생성 순서를 실제로 복원하는 핵심 부분**이다.

---

## 4. 구조적 해석

| 항목            | 설명                                                                       |
| ------------- | ------------------------------------------------------------------------ |
| **공통 시퀀스 공간** | 텍스트와 이미지가 동일한 `input_ids` 내에서 함께 표현됨                                     |
| **모달리티 추적**   | `modality`를 통해 각 토큰의 유형(텍스트/이미지)을 명확히 구분                                 |
| **연결 관계 유지**  | `img_pos_ids`로 “이미지가 어느 텍스트 뒤에 오는가”를 추적                                  |
| **교차 복원 가능**  | `to_list()`를 통해 interleaved 시퀀스를 재구성                                     |
| **학습 연동**     | 모델의 `_sample()` 및 `predict_step()`에서 이 구조를 그대로 사용해 joint diffusion 생성 수행 |

---

## 5. BOI(`<image>`) 토큰 예측 근거

UniDisc는 `<image>`(BOI) 토큰을 **직접 예측 대상**으로 포함한다. 이는 interleaved 구조의 핵심 신호이자, 모델이 텍스트에서 이미지로 전환하는 시점을 인식하게 한다.

### 근거 1 — `model_eval.py` 내 문자열 기반 삽입 코드

```python
if img_first:
    dec_txt = ['<image> ' + txt for txt in dec_txt]
else:
    dec_txt = [txt + ' <image>' for txt in dec_txt]
```

이 구문은 모델이 생성하는 텍스트 샘플에 `<image>` 토큰을 명시적으로 포함시키는 부분으로,
모델이 해당 토큰을 예측하고 시퀀스 내 위치를 복원해야 함을 의미한다【52:0†model_eval.py†L24-L28】.

---

### 근거 2 — `get_interleaved_block_mask()` 호출부

```python
if self.config.trainer.interleaved_training_flex_attention:
    kwargs['block_mask'] = get_interleaved_block_mask(kwargs['sample_ids'], x.shape[0], x.shape[-1], self.device)
```

이 부분은 interleaved block 단위(즉, 텍스트 + `<image>` + 이미지 토큰 전체)를 마스크 및 복원 대상으로 지정하는 코드이다.
따라서 `<image>` 토큰도 diffusion 타깃(예측 대상)에 포함된다【52:1†model_eval.py†L34-L37】.

---

### 근거 3 — `x0_unmask` 및 `image_indices` 로직

```python
_img_indices = torch.isin(x0, torch.tensor(list(image_indices), device=self.device))
x0_unmask |= (~_img_indices)
```

이 로직은 이미지 관련 토큰(즉, `<image>` 및 이미지 콘텐츠 토큰)을 구분하여 마스킹/언마스킹을 수행한다.
이는 `<image>` 토큰이 단순 구분자가 아닌 예측 가능한 토큰임을 보여준다【52:1†model_eval.py†L15-L25】.

---

### 근거 4 — Joint diffusion 설계

`model_eval.py` 전반의 샘플링 루프(`_sample()` 및 `_ar_sampler`)는 모든 `input_ids`를 대상으로 노이즈-복원(diffusion) 과정을 수행하며,
텍스트/이미지 토큰의 구분은 `modality`를 통해서만 이뤄진다. 따라서 `<image>` 토큰 역시 동일한 예측 과정에 포함된다.

---

## 6. 결론

| 구분            | 설명                                                                |
| ------------- | ----------------------------------------------------------------- |
| **BOI 포함 여부** | 포함됨 — `<image>`는 모델이 직접 예측하는 토큰                                   |
| **근거 파일**     | `model_eval.py`, `tensor_utils.py` (`get_interleaved_block_mask`) |
| **핵심 함수**     | `_sample()`, `get_interleaved_block_mask()`, `decode_latents()`   |
| **학습 메커니즘**   | joint diffusion을 통해 `<image>` 위치 및 이미지 내용을 동시에 복원                 |
| **의미**        | UniDisc는 “이미지 시작 위치까지 예측하는 멀티모달 생성”을 수행함                          |
