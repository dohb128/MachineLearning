# 📝 TOEIC Problem Solving Model

## 1. 프로젝트 개요
본 프로젝트는 **TOEIC Part 5 (단문 공란 메우기)** 유형의 문제를 자동으로 풀이하는 인공지능 모델을 구축하고 분석하는 프로젝트입니다.  
BERT 기반의 사전 학습된 언어 모델을 활용하여 **문맥을 이해하고 가장 적절한 정답을 예측**하는 방식을 사용합니다.

---

## 2. 데이터셋 구조
데이터셋은 다음과 같은 필드로 이루어진 TOEIC Part 5 문제들로 구성됩니다.

- **Question_ID**: 문제 고유 ID  
- **Question_Text**: 문제 지문 (빈칸 포함)  
- **Option_A, Option_B, Option_C, Option_D**: 4지 선다형 보기  
- **Answer_Label**: 정답 레이블 (A, B, C, D)

---

## 3. 사용 모델 및 기술

### 3-1. 모델 아키텍처
- **모델**: BERT (Bidirectional Encoder Representations from Transformers)
- **Task**: Multiple Choice Classification (4지문 정답 선택)
- **입력 형식**:  
  \`\`\`
  [CLS] Question_Text [SEP] Option_X [SEP]
  \`\`\`
  → 네 개의 보기 각각을 Question_Text와 조합하여 총 4개의 입력을 생성  
- **출력**: 4개 입력 각각의 score 계산 → 가장 높은 score가 정답

### 3-2. 사용 라이브러리
- `transformers` (Hugging Face)
- `torch` (PyTorch)
- `pandas`, `numpy`
- `matplotlib`, `seaborn`

---

## 4. 분석 프로세스

### 4-1. 데이터 로드 및 전처리
- 문제와 보기를 BERT 입력 형식에 맞게 토큰화  
- Attention Mask, Token Type IDs 생성  
- Sequence Length 통일 후 모델 입력

### 4-2. 모델 학습 (Fine-tuning)
- 학습 데이터를 이용해 BERT를 Multiple Choice Classification task로 미세 조정

아래는 **학습과 검증 전체 과정 로그**입니다.
| Epoch | Training Loss | Training Acc | Validation Loss | Validation Acc |
|-------|----------------|---------------|------------------|-----------------|
| 1 | 1.3848 | 0.2694 | 1.3799 | 0.3705 |
| 2 | 1.2063 | 0.4601 | 0.8792 | 0.6773 |
| 3 | 0.7199 | 0.7278 | 0.5945 | **0.8000** |
| 4 | 0.3927 | 0.8548 | 0.4720 | 0.8318 |
| 5 | 0.2595 | 0.9117 | 0.5184 | 0.8477 |
| 6 | 0.1712 | 0.9396 | 0.5060 | 0.8523 |
| 7 | 0.1243 | 0.9550 | 0.4741 | 0.8614 |
| 8 | 0.0978 | **0.9727** | 0.4762 | **0.8659** |

- Epoch 1 → 낮은 성능에서 시작 (37%)
- Epoch 3 → Validation 80% 도달 (큰 향상)
- Epoch 4~8 → 성능 안정화 + 꾸준한 상승
- 최종 Validation Accuracy → 86.59% (Epoch 8)
- Training Accuracy는 지속적으로 증가하여 97% 이상

#### 🔍 학습 결과 요약
- 초기 정확도: **27% → 37%**  
- Epoch 3에서 급격히 개선 (**80% 도달**)  
- 최종 Validation Accuracy: **86.59%**

학습이 매우 안정적으로 진행되었고, Epoch 4~8 구간에서 Validation Accuracy가 꾸준히 유지됨.


### 4-3. 성능 평가 및 시각화

#### 📌 Validation Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.82 | 0.86 | 0.84 | 109 |
| 1 | 0.87 | 0.86 | 0.86 | 115 |
| 2 | 0.87 | 0.90 | 0.89 | 115 |
| 3 | 0.91 | 0.83 | 0.87 | 101 |
| **Accuracy** |  |  | **0.87** | **440** |
| **Macro Avg** | 0.87 | 0.86 | 0.87 | 440 |
| **Weighted Avg** | 0.87 | 0.87 | 0.87 | 440 |

#### ✔ 정답/오답 비율 시각화
<img width="481" height="505" alt="download" src="https://github.com/user-attachments/assets/a584312e-7a64-43da-9270-4ec017555b0a" />

#### ✔ 클래스별 정확도 (A/B/C/D 보기별)
<img width="689" height="374" alt="download" src="https://github.com/user-attachments/assets/9be8320b-4c31-41e7-821f-cdba7f737f77" />

#### ✔ Confusion Matrix (정답-예측 관계)
<img width="502" height="469" alt="download" src="https://github.com/user-attachments/assets/ec1e9ced-5fdf-4f9a-8cb5-2767fd7ee442" />

---

### 4-4. 오답 노트 분석 (총 59개 중 5개 예시)

#### 📝 오답 예시

| ID | 정답 유형 | 문제 | 정답 | 오답 |
|---|---|---|---|---|
| 56 | 어휘/문법 | With all the negative reviews about the novel,... | misinterpreted | misinterpreting |
| 1213 | 어휘 | The national economy has been ------- for near... | stagnant | stagnating |
| 1516 | 대명사 | Mr. Forester is confident that _______ can con... | you | your |
| 2425 | 형용사 | Pedestrians are advised to be _______ and look... | cautious | cautioning |
| 3031 | 부사 | Automobile factories in areas _______ affected... | severely | severest |

**컬럼 설명**  
- **ID**: Question_ID  
- **정답**: Answer_Label 기반 실제 정답 텍스트  
- **오답**: 모델 예측 값 (`pred_text`)  

---

## 5. 실행 방법 (Demo)

<img width="940" height="935" alt="image" src="https://github.com/user-attachments/assets/c58477cc-55fb-404d-b0dd-8666b24ffed8" />

`http://mc-toeic.kro.kr/` 에 접속하여  
문제 + 보기 4개를 입력하고 **“정답 예측하기”** 버튼을 클릭하면 모델이 정답을 출력합니다.

### 📌 예시 화면

| 문제 입력 | 정답 예측 |
|---|---|
| <img width="943" height="927" alt="image" src="https://github.com/user-attachments/assets/114aa669-e8c2-4e39-b21d-0caecb7f6b89" /> | <img width="947" height="934" alt="image" src="https://github.com/user-attachments/assets/43f683eb-e7d4-4331-af3a-e7c04becf3f7" /> |

---
## 6. 결론 및 향후 계획

### 6-1. 결론 💡

BERT 모델은 TOEIC Part 5 문제에서 **87%**의 높은 성능을 보였습니다.

Confusion Matrix 분석 결과, 모델은 [특정 보기: 예: D 보기]를 [다른 보기: 예: C 보기]로 잘못 예측하는 경향이 다수 있었습니다.

### 6-2. 향후 계획

- 데이터 확장: 오답 문제들을 분석하여, 어떤 유형의 문제가 오답이 많은지 분석하고, 해당 문제에 대한 데이터셋을 추가하여 모델 성능을 개선.
- 모델 최적화: ELECTRA, RoBERTa 등 다른 Transformer 기반 모델을 시도하여 성능 비교 및 경량화.
- 오류 분석 심화: 오답 문제의 특징(예: 지문의 길이, 단어의 빈도 등)을 분석하여 모델의 취약점을 보완.
