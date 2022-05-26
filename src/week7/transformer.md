- Main idea : Attention

## 등장 배경

**기존 Seq2Seq 모델들의 한계**

- context vector v에 소스 문장의 정보를 합축한다. → 병목 현상이 발생하여 성능 하락의 원인이 된다.
- 디코더가 context vector를 매번 참고하는 방법으로 해결 할 수 있으나, 이 또한 여전히 소스 문장을 하나의 벡터로 압축해야한다.

### 1. Linear Interaction distance

- RNN 은 단방향 진행
- 가까이 있는 단어들은 서로의 의미에 영향을 주기 쉽다.

→ Vanashing Gradient

### 2. 병렬화의 어려움(Lack of parallelizability)

- RNN은 직렬적인 모델 ⇒ 기존의 Encoder Decoder을 그대로 간직

## 특징

### 1. RNN, CNN을 사용하지 않는다. → Positional Encoding 사용.

- RNN은 단어의 위치정보를 가질 수 있다.

### 2. Attention Mechanism 적용

- 디코더는 인코더의 모든 출력을 참고한다.
- 디코더는 매번 인코더의 모든 출력 중 어떤 정보가 중요한지를 계산하여, 중요한 정보만을 선택하여 활용한다.

## 주요 하이퍼파라미터

- dmodel = 512

트랜스포머의 인코더와 디코더에서의 정해진 입력과 출력의 크기를 의미합니다. 임베딩 벡터의 차원 또한 dmodel이며, 각 인코더와 디코더가 다음 층의 인코더와 디코더로 값을 보낼 때에도 이 차원을 유지합니다. 논문에서는 512입니다.

- num_layers = 6

트랜스포머에서 하나의 인코더와 디코더를 층으로 생각하였을 때, 트랜스포머 모델에서 인코더와 디코더가 총 몇 층으로 구성되었는지를 의미합니다. 논문에서는 인코더와 디코더를 각각 총 6개 쌓았습니다.

- num_heads = 8

트랜스포머에서는 어텐션을 사용할 때, 한 번 하는 것 보다 여러 개로 분할해서 병렬로 어텐션을 수행하고 결과값을 다시 하나로 합치는 방식을 택했습니다. 이때 이 병렬의 개수를 의미합니다.

- dff = 2048

트랜스포머 내부에는 피드 포워드 신경망이 존재하며 해당 신경망의 은닉층의 크기를 의미합니다. 피드 포워드 신경망의 입력층과 출력층의 크기는 dmodel입니다.

## 구조



### 1) Encoder



- RNN은 데이터가 순서에 대한 정보를 갖지만, Transformer는 RNN을 사용하지 않는다.
- 따라서, 위치 정보를 포함하는 임베딩을 사용해야 한다.
- 문장 전체를 임베딩하여 입력값으로 활용한다.
- 위치 정보를 전달하기 위해서 Positional Encoding을 사용한다. → 각 단어에 대한 위치정보를 가진 데이터를 추가함으로써 위치 정보를 별도로 기억하는 방법이다.
- 주기 함수를 활용한 공식을 사용하여, 각 단어의 상대적인 위치 정보를 네트워크에 입력한다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb119f2e-7fba-40f0-bbfb-1a018c3c86a3/Untitled.png)

- 임베딩한 각각의 단어를 MHA에 넣어 Attention을 수행한다. (인코더 파트에서 시행한 Attention을 `Self Attention` 이라고 함.)
- 각 단어간의 연관성을 Attention을 계산하여 알 수 있다.

- 추가적으로 Residual Learning을 사용하여 성능을 향상 → 특정 레이어를 건너뛰면서, 기존 정보를 포함하여 잔여된 부분만 학습하여 학습 난이도가 낮고 초기의 모델 수렴속도가 높고 Global Optima를 찾기 좋아진다.

- 이후 정규화를 한 후 FeedForwarding을 거치고 다시 정규화를 함으로써 인코더 레이어에서 정보 값을 추출한다.
- 여기서 각각의 레이어는 서로 다른 파라미터를 가진다.

### 2) Decoder



- 3개의 서브층이 존재한다.
- 서브층 연산 후에는 Drop out, Residual Connection, Layer Normalization이 이루어진다.

### 첫번째 서브층 (Look-ahead mask)

- 첫 어텐션은 이코더의 어텐션과 마찬가지로 입력값에 대해 각 단어들의 가중치를 찾는 `self-attention` 이다. (Query, key, Value가 같은 경우 self-attention)
- `attention-score matrix` 에 마스킹을 적용한다.

- 가장 마지막 Encoder layer에서 나온 출력 값을 입력으로 받는다.
- 매번 출력마다 입력 소스 문장 중 초점을 두어야하는 단어를 찾는다.
- 디코더도 마찬가지로 여러 개의 레이어로 구성되어 있다. 그리고 모든 레이어는 인코더의 마지막 출력 값을 입력으로 받는다.

### 두번째 서브층 (Encoder-Decoder Attention)

- 두 번째 서브층의 Attention은 `Encoder-Decoder Attention` 이라고 한다.
- Query는 디코더인 행렬, Key, Value는 인코더 행렬

## Attention

- Encoder Self-Attention
- Masked Decoder Self-Attention
- Encoder-Decoder Attention

## (포지션-아이즈 피드 포워드 신경망, Position-Wise FFNN)

- Encoder, Decoder 층에 공통적으로 갖고 있는 서브층
