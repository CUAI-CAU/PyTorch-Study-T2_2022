## BERT Model

- BERT(Bidirectional Encoder Representations from Transformers)는 구글이 개발한 사전훈련(pre-training) 모델입니다. 위키피디아 같은 텍스트 코퍼스를 사용해서 미리 학습을 하면, 언어의 기본적인 패턴을 이해한 모델이 만들어집니다. 이를 기반으로 새로운 문제에 적용하는 전이학습(transfer learning)을 수행합니다. 좀 더 적은 데이터로 보다 빠르게 학습이 가능하다는 장점이 있습니다. 그래서 최근 자연어처리의 핵심 기법으로 떠오르고 있습니다.
- 양방향 모델

## 특징

- BERT는 단어보다 더 작은 단위로 쪼개는 서브워드 토크나이저(subword tokenizer)를 사용한다. 기본적으로 자주 등장하는 단어는 그대로 단어 집합에 추가하고, 자주 등장하지 않는 단어는 더 작은 단위인 서브워드로 분리되어 단어 집합에 추가된다. → 단어 집합이 갖추어 지면 토큰화를 진행한다.
- `Position Embedding`

Transformer에서는 Positional Encoding을 사용하여 단어의 위치정보를 표현했다. BERT는 학습을 통해 위치정보를 얻는 Position Embedding을 사용한다.

- `Masked Language Model`

MLM은 일련의 단어가 주어지면 그 단어를 예측하는 작업. 입력 텍스트의 15% 단어를 랜덤으로 Masking한다. 신경망이 가려진 단어들을 예측하도록 한다.**(빈칸 추론 시키기)**

## 구조

- Transformer의 Encoder를 쌓아올린 구조. Base 버전에서 총 12개, Large 버전에서 총 24개를 쌓았다.
- 입력 : 임베딩 층(Embedding Layer)을 지난 임베딩 벡터, 768차원의 임베딩 벡터가 되어 입력으로 사용한다.
- 출력 : 768차원의 벡터

### 전이학습 모델

- 사전학습된 대용량의 Unlabeled data를 이용하여 언어 모델을 학습하고, 이를 토대로 특정 작업(문서 분류, 질의 응답, 번역)을 위한 신경망을 추가하는 전이 학습 방법

### 사전 학습 모델

- BERT는 사전학습 되어 있는 모델을 제공하기 때문에, 상대적으로 적은 자원만으로도 충분히 가능

## BERT의 Input Representation (3개의 임베딩 층)

### 1) Token Embeddings(WordPiece Embedding)

- Word piece 임베딩 방식을 사용. → 자주 등장하면서, 가장 긴 길이의 sub-word를 하나의 단위로 만든다.
- 실질적인 입력이 되는 워드 임베딩
- 임베딩 벡터의 종류는 단어 집합의 크기

### 2) Segment Embeddings

- 두개의 문장을 구분하기 위한 임베딩 → 종류는 문장의 최대 개수
- 토큰으로 나누어진 단어들을 다시 하나의 문장으로 만들고 첫번째 [SEP] 토큰 까지는 0으로 그 이후 [SEP] 토큰까지는 1 값으로ㅓ 마스크를 만들어 각 문장들을 구분합니다.

### 3) Position Embeddings

- 위치 정보를 학습하기 위한 임베딩 → 종류는 문장의 최대 길이
- 토큰의 순서를 인코딩한다. (BERT는 transformer의 encoder를 사용함.)

## BERT의 Pre-training과 Fine-Tuning

- 자연어 처리는 2단계로 진행된다.(Pre-training, Fine-tuning)
1. 마스크 언어 모델(Masked LM) : 
2. 다음 문장 예측 (Next Sentence Prediction) 

## Transformer기반의 BERT

- BERT는 MLM과 NSP를 위해 Transformer을 기반으로 구성된다.

### BERT의 MLM(Maksed Language Model)

- 

### BERT의 NSP(Next Sentence Prediction)

- 두 문장의 관계를 이해하기 위해 BERT의 학습 과정에서 두 번째 문장이 첫 번째 문장의 바로 다음에 오는 문장인지 예측하는 방식
- BERT는 [SEP] 특수 토큰으로 문장을 분리한다.
