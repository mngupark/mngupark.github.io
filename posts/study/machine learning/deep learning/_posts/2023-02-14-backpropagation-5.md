---
layout: post
title: Backpropagation-5
category: deep learning
post-order: 15
post-series: Deep learning from scratch
---

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/backpropation-4)에서는 Affine 계층의 역전파에 대해서 배웠습니다. 이번 post에서는 Softmax-with-Loss 계층에 대해서 설명하고 오차역전파법을 정리하겠습니다.

---

# Softmax-with-Loss 계층

마지막으로 출력층에서 사용하는 **소프트맥스**(Softmax) 함수 계층입니다. 이 함수는 입력 값을 정규화하여 출력하기에 출력 값의 확률 분포라고도 볼 수 있다고 말씀을 드렸었습니다.

> 📗 신경망에서 수행하는 작업은 **학습**과 **추론** 두 가지입니다. 추론할 때는 일반적으로 **Softmax** 계층을 사용하지 않습니다. 즉, 마지막 Affine 계층의 출력을 인식 결과로 사용합니다. 또한, 신경망에서 정규화하지 않는 출력 결과를 **점수**(score)라고 합니다. 신경망 추론에서 답을 하나만 내는 경우에는 가장 높은 점수만 알면 되기 때문에 굳이 Softmax 계층을 사용하지 않는 것입니다. 하지만, 신경망을 **학습**할 때는 Softmax 계층이 필요합니다.

일반적으로 소프트맥스 계층을 구현할 때 손실 함수인 **교차 엔트로피 오차**도 포함해서 "***Softmax-with-Loss***" 계층이라는 이름으로 구현합니다. 먼저 계산 그래프의 결과부터 보여드리겠습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_23.jpg"
          title="Computational graph of the softmax-with-loss"
          alt="Image of computational graph of the softmax-with-loss"
          class="img_center"
          style="width: 75%"/>
     <figcaption>Softmax-with-Loss 계층의 역전파</figcaption>
</figure>

[Fig. 1.]에서는 3클래스 분류를 수행하는 신경망을 가정하고 있습니다. 이전 계층으로부터의 입력은 $(a_1,\ a_2,\ a_3)$이며 소프트맥스 계층은 $(y_1,\ y_2,\ y_3)$를 출력합니다. 정답 레이블은 $(t_1,\ t_2,\ t_3)$이며 교차 엔트로피 오차 계층은 손실 $L$을 출력합니다.

지금까지 배웠던 계산 그래프의 원리를 잘 활용하면 순전파와 역전파를 계산하는데 큰 무리는 없을 것입니다. 다만 유의해야할 점은 바로 정답 레이블이 "**one-hot-encoding**"으로 만들어진 벡터라는 것입니다. 그래서 모든 정답 레이블의 합은 항상 **1**이 됩니다. [Fig. 1.]에서 소프트맥스 계층의 나눗셈 노드에서 덧셈 노드의 역전파를 계산할 때 정답 레이블을 다 더하는 과정이 있습니다. 그래서 $(t_1+t_2+t_3)/S=1/S$이라는 결과가 나오는 것입니다.

또한, [Fig. 1.]을 자세히 보시면 두 계층의 입출력과 정답 레이블만으로 순전파와 역전파를 계산할 수 있는 것을 확인하실 수 있습니다. 그래서 이를 좀 더 간소화한 계산 그래프는 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_24.jpg"
          title="Simplified computational graph of the softmax-with-loss"
          alt="Image of simplified computational graph of the softmax-with-loss"
          class="img_center"
          style="width: 75%"/>
     <figcaption>간소화된 Softmax-with-Loss 계층의 역전파</figcaption>
</figure>

[Fig. 2.]에서는 Softmax-with-Loss 계층의 마지막 역전파의 출력이 **소프트맥스 계층 출력과 정답 레이블간의 차분**$(y_1-t_1,\ y_2-t_2,\ y_3-t_3)$으로 계산됩니다. 신경망의 역전파에서는 이 차이인 **오차**가 앞 계층에 순차적으로 전파되는 것입니다. 그런데 신경망 학습의 목적은 **신경망의 출력**(소프트맥스 계층의 출력)이 **정답 레이블**과 가까워지도록 가중치 매개변수의 값을 조정하는 것이었습니다. 즉, Softmax-with-Loss 계층의 역전파는 **신경망의 현재 출력과 정답 레이블의 오차**를 있는 그대로 드러내는 것입니다. 오차역전파법이라는 이름의 유래도 바로 여기서 나온 것입니다.

> 💡 **소프트맥스 함수**의 손실 함수로 **교차 엔트로피 오차**를 사용하니 역전파가 말끔하게 떨어집니다. 사실 이런 말끔함은 우연이 아니라 교차 엔트로피 오차라는 함수가 그렇게 설계되었기 때문입니다. 또한 **회귀**(regression)의 출력층에서 사용하는 **항등 함수**의 손실 함수로 **오차제곱합**을 이용하는 이유도 이와 같습니다. 항등 함수의 손실 함수로 오차제곱합을 사용하면 역전파의 결과가 똑같이 말끔하게(신경망의 현재 출력과 정답 레이블의 차분) 떨어집니다.

그럼 이제 Python을 이용해서 Softmax-with-Loss 계층을 구현해보겠습니다.
```python
class SoftmaxWithLossLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
```

구현 시 유의할 점은 역전파 때 전파하는 값을 배치의 수로 나눠서 **데이터 1개당 오차**를 앞 계층으로 전파하는 것입니다.

---

# 오차역전파법 구현

구현에 앞서, 신경망 학습의 순서를 다시 정리해보겠습니다.

* 전제
  * 신경망에는 적응 가능한 **매개변수**(가중치와 편향)가 있고, 이 매개변수를 **훈련 데이터**에 적응하도록 조정하는 것을 학습이라고 합니다. 신경망 학습은 다음과 같이 4단계로 수행됩니다.
* 1단계 - 미니배치
  * 훈련 데이터 중 일부(**미니배치**)를 무작위로 가져옵니다. 이 미니배치의 매개변수에 대한 손실 함수 값을 줄이는 것이 목표입니다.
* 2단계 - **기울기 산출**
  * 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 **기울기**를 구합니다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시합니다.
* 3단계 - 매개변수 갱신
  * 가중치 매개변수를 기울기 방향으로 **아주 조금** 갱신합니다(경사 하강법).
* 4단계 - 반복
  * 1~3단계를 반복합니다.

지금까지 설명한 오차역전파법이 등장하는 단계는 바로 두 번째 단계인 "**기울기 산출**"입니다.

---

이제 손글씨 숫자를 학습하는 신경망을 구현해보겠습니다. MNIST 데이터셋을 가지고 **학습**을 진행하는 신경망의 구조는 아래와 같습니다.

- 2층 네트워크 (입력층, 은닉층 1개, 출력층)
- 입력층의 뉴런 784개 (28x28 pixels)
- 은닉층의 뉴런의 개수 **50**개
- 출력층의 뉴런 10개 (0에서 9까지의 숫자)
- 활성화 함수는 **ReLU** 함수, 출력층 활성화 함수는 소프트맥스 함수
- 가중치 매개변수는 정규분포를 따르는 난수, 편향 매개변수는 0으로 초기화
- 손실 함수는 교차 엔트로피 오차 함수
- 미니배치의 크기는 100, 학습률은 0.1, 경사법에 의한 갱신 횟수는 10,000
- 훈련 데이터의 크기는 60,000, 시험 데이터의 크기는 10,000

이전에 구현했던 2층 신경망과 다른 점은 굵은 글씨로 표기했습니다.

> :smiley: 자세한 코드는 Git 저장소[^fn-two-layered-network-learning]를 통해 확인해주시기 바랍니다.

아래의 그림은 손실 함수의 값의 변화를 기록한 그래프입니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/backpropagation_25.jpg"
         title="Graph of loss function"
         alt="Image of graph of loss function"
         class="img_center"
         style="width: 50%"/>
    <figcaption>학습에 의한 손실 함수값의 변화</figcaption>
</figure>

[Fig. 3.]을 보면 학습 횟수가 늘어나면서 손실 함수값이 줄어듭니다. 이는 학습이 잘 되고 있다는 뜻으로, 신경망의 가중치 매개변수가 서서히 데이터에 적응하고 있음을 의미합니다. 신경망이 학습을 하고 있다는 의미입니다.

아래의 그림은 훈련 데이터와 시험 데이터에 대한 정확도를 기록한 그래프입니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/backpropagation_26.jpg"
         title="Graph of accuracy"
         alt="Image of graph of accuracy"
         class="img_center"
         style="width: 50%"/>
    <figcaption>훈련 데이터 vs 시험 데이터</figcaption>
</figure>

[Fig. 4.]에서 실선은 훈련 데이터, 점선은 시험 데이터에 대한 정확도를 나타냈습니다. 학습이 진행될수록 두 데이터셋에 대한 정확도가 모두 좋아지는 것을 확인할 수 있습니다. 즉, 오버피팅이 일어나지 않았다고 생각할 수 있겠습니다.

지금까지 기울기를 구하는 방법으로 **수치 미분**을 이용한 방법과 **오차역전파법**을 이용한 방법에 대해서 알아보았습니다. 수치 미분은 구현하기 쉽다는 장점이 있지만 속도가 느리다는 단점이 있었고, 오차역전파법은 속도가 매우 빠르다는 장점이 있지만 구현하기 어렵다는 단점이 있었습니다. 그래서 오차역전파법은 구현하기 복잡하기에 가끔 **실수**가 포함되어 있습니다. 이를 검증하기 위한 방법이 바로 **기울기 확인**(gradient check)입니다. 상대적으로 구현하기 쉬운 수치 미분의 값과 오차역전파법으로 계산한 값을 비교하는 것입니다.

> :smiley: 기울기 확인에 대한 자세한 코드는 Git 저장소[^fn-check-gradient]를 통해 확인해주시기 바랍니다.

각 가중치 매개변수들의 차이의 절댓값을 평균한 오차는 아래와 같습니다.

```python
W1:2.0902133031008377e-10
b1:1.528760207091743e-09
W2:8.575880791007656e-10
b2:9.159714665125877e-10
```

CPU의 부동소수점 계산 능력 정확도와 메모리의 한계로 오차가 완전히 0이 될수는 없습니다. 하지만 $10^{-9}\sim 10^{-10}$의 오차라면 충분히 작은 수치라고 생각할 수 있겠습니다.

지금까지 오차역전파법의 기초와 이론, 구현까지 정리해봤습니다. 다음 post에서는 신경망의 효과적인 학습을 위한 여러 방법들에 대해서 다뤄보겠습니다.

## 오차역전파법 요약
- 계산 그래프를 이용하면 계산 과정을 시각적으로 파악할 수 있다.
- 계산 그래프의 노드는 국소적 계산으로 구성된다. 국소적 계산을 조합해 전체 계산을 구성한다.
- 계산 그래프의 순전파는 통상의 계산을 수행한다. 한편, 계산 그래프의 역전파로는 각 노드의 미분(국소적 미분, local differentiation)을 구할 수 있다.
- 신경망의 구성 요소를 계층으로 구현하여 기울기를 효과적으로 계산할 수 있다. 이를 오차역전파법이라고 부른다.
- 수치 미분과 오차역전파법의 결과를 비교하면 오차역전파법의 구현에 잘못이 없는지 확인할 수 있다. 이를 기울기 확인이라고 부른다.

---

[^fn-two-layered-network-learning]: [Github repository](https://github.com/Gyuhub/dl_scratch/blob/main/dl_scratch/ch5/two_layered_network_learning.py)에서 2층 신경망 학습에 관한 코드를 확인하실 수 있습니다.

[^fn-check-gradient]: [Github repository](https://github.com/Gyuhub/dl_scratch/blob/main/dl_scratch/ch5/check_gradient.py)에서 기울기 확인에 관한 코드를 확인하실 수 있습니다.