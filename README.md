# 📝 TOEIC AI Predictor
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
아래는 **학습과 검증 과정 전체 로그**입니다.
```
======== Epoch 1 / 8 ========
Training Loss: 1.3848, Acc: 0.2694
Validation Loss: 1.3799, Acc: 0.3705

======== Epoch 2 / 8 ========
Training Loss: 1.2063, Acc: 0.4601
Validation Loss: 0.8792, Acc: 0.6773

======== Epoch 3 / 8 ========
Training Loss: 0.7199, Acc: 0.7278
Validation Loss: 0.5945, Acc: 0.8000

======== Epoch 4 / 8 ========
Training Loss: 0.3927, Acc: 0.8548
Validation Loss: 0.4720, Acc: 0.8318

======== Epoch 5 / 8 ========
Training Loss: 0.2595, Acc: 0.9117
Validation Loss: 0.5184, Acc: 0.8477

======== Epoch 6 / 8 ========
Training Loss: 0.1712, Acc: 0.9396
Validation Loss: 0.5060, Acc: 0.8523

======== Epoch 7 / 8 ========
Training Loss: 0.1243, Acc: 0.9550
Validation Loss: 0.4741, Acc: 0.8614

======== Epoch 8 / 8 ========
Training Loss: 0.0978, Acc: 0.9727
Validation Loss: 0.4762, Acc: 0.8659

=== Training Complete ===
```
#### 🔍 학습 결과 요약
- 초기 정확도: **27% → 37%**  
- Epoch 3에서 급격히 개선 (**80% 도달**)  
- 최종 Validation Accuracy: **86.59%**

-> 학습이 매우 안정적으로 진행되었고, Epoch 4~8 구간에서 Validation Accuracy가 꾸준히 유지됨.

### 4-3. 성능 평가

#### 📌 Validation Classification Report


|| precision |recall| f1-score |support|
|---|---|---|---|---|
|A|0.82| 0.86 |0.84 |109|
|B |0.87| 0.86| 0.86 |115|
|C |0.87 |0.90 |0.89| 115|
|D |0.91 |0.83 |0.87 |101|
|accuracy| -| -|0.87| 440|
|macro avg |0.87 |0.86 |0.87 |440|
weighted avg |0.87 |0.87 |0.87 |440|
- **최종 Validation Accuracy: 0.87 (87%)**
- 클래스별 F1-score가 모두 **0.84~0.89**로 안정적
- 가장 강한 클래스: **Label 3 (precision 0.91)**
- Recall이 가장 높은 클래스: **Label 2 (0.90)**

---

#### 📌 Confusion Matrix
```
[[ 94 8 3 4]
[ 7 99 8 1]
[ 6 2 104 3]
[ 8 5 4 84]]
```


|  | A | B |C|D|
| --- | --- | --- | --- | ---|
| A |94|8|3|4|
| B |7|99|8|1|
| C |6|2|104|3|
| D |8|5|4|84|

#### 🔍 Confusion Matrix 해석
- Label 0 → 주로 Label 1과 3으로 오분류  
- Label 3 → Label 0, 1로 소수 오분류  
- Label 2는 가장 안정적으로 예측됨 (정확 예측 104건)

---

#### 📌 종합 평가
- 전체 Accuracy 87%는 BERT 기반 4지 선다형 태스크로 **우수한 성능**  
- Precision/Recall/F1 모두 균형적  
- 극단적인 클래스 불균형이나 편향이 없음  
- Confusion Matrix에서도 네 클래스 모두 비교적 안정적으로 예측함  

### 4-4. 학습 결과 시각화
#### ✔ 정답/오답 비율
<img width="481" height="505" alt="download" src="https://github.com/user-attachments/assets/a584312e-7a64-43da-9270-4ec017555b0a" />

#### ✔ 클래스별 정확도 (A/B/C/D 보기별)
<img width="689" height="374" alt="download" src="https://github.com/user-attachments/assets/9be8320b-4c31-41e7-821f-cdba7f737f77" />

#### ✔ Confusion Matrix (정답-예측 관계)
<img width="502" height="469" alt="download" src="https://github.com/user-attachments/assets/ec1e9ced-5fdf-4f9a-8cb5-2767fd7ee442" />

---

## 4-5. 오답 노트 분석 (총 59개 중 5개 예시)

### 📝 오답 예시

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

### 📌 실행 화면
![TOEIC 데모](https://github.com/user-attachments/assets/e9e1f9df-55f8-461e-853c-5b8cfa5f6333)

| 문제 입력 | 정답 예측 |
|---|---|
| <img width="943" height="927" alt="image" src="https://github.com/user-attachments/assets/114aa669-e8c2-4e39-b21d-0caecb7f6b89" /> | <img width="947" height="934" alt="image" src="https://github.com/user-attachments/assets/43f683eb-e7d4-4331-af3a-e7c04becf3f7" /> |

## 6. 결론
- BERT 기반 모델은 TOEIC Part 5 문제에서 **87% 정확도**를 달성  
- 오답 분석 결과, 문법/어휘 문제에서 특히 강점  
- 향후 ELECTRA·RoBERTa 등 다양한 모델 실험 예정 
