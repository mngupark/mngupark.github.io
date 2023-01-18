---
layout: post
title: Neural-Network-4
category: deep learning
post-order: 7
---

# 신경망의 순전파

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/neural-network-3)에서는 3층 신경망을 구현하면서 신경망의 순전파에 대해서 알아보았었습니다. 신경망 학습으로 넘어가기 전에 소프트맥스 함수에 대해서 좀 더 정리하고, 손글씨 숫자 이미지를 신경망의 순전파를 통해 추론하는 과정을 구현해보겠습니다.

---

# 소프트맥스

신경망에 대해서 공부하다보면 매우 빈번하게 등장하는 함수입니다. 수식으로는 아래와 같이 표현할 수 있습니다.

$$
h(x)_i=\frac{\exp{z_i}}{\sum_{j=1}^k \exp{z_j}}\ \text{for}\ i=1,\cdots,k\ \text{and}\ \boldsymbol{z}=\begin{bmatrix} z_1 & \cdots & z_k \end{bmatrix} \in \mathbb{R}^k \label{softmax_activation} \tag{1}
$$

하지만 엄밀히 말하자면 위의 수식은 **소프트맥스**(softmax) 함수가 아니라 **소프트맥스 활성화**(softmax activation) 함수입니다. 소프트맥스 활성화 함수란 신경망의 활성화 함수로 소프트맥스 함수가 사용되었을 경우를 말하는 것입니다. 그렇다면 소프트맥스 함수는 무엇일까요?

# LogSumExp

**RealSoftMax**라고도 불리는 **LogSumExp**가 소프트맥스 함수입니다. 수식으로는 아래와 같이 표현할 수 있습니다.

$$
LSE\begin{pmatrix} x_1,\ \cdots,\ x_n \end{pmatrix}=\log{(\exp{x_1}+\cdots+\exp{x_n})}=\log{\sum_{k=1}^n \exp{x_k}} \label{log_sum_exp} \tag{2}
$$

이 함수는 $\boldsymbol{max(a, b)}$함수의 **smooth approximation**입니다. $max(a, b)$함수는 **최댓값 함수**로 $a$와 $b$중 더 큰 값을 반환하는 함수입니다. 최댓값 함수의 smooth approximation이라는 말이 잘 이해가 되지 않을 수 있습니다. 먼저 수식으로 증명해보겠습니다.

최댓값 함수 $\underset{i}{max}\ x_i$에 대해서 LSE함수는 아래 경계 조건을 만족합니다.

$$
max\{x_1,\ \cdots,\ x_n\}\ {\color{Red}\le}\ LSE(x_1,\ \cdots,\ x_n)\ {\color{Blue}\le}\ max\{x_1,\ \cdots,\ x_n\} + \log{(n)} \label{bound_condition_1} \tag{3}
$$

식 $(\ref{bound_condition_1})$에서 $n=1$이라면 <span style="color: red">빨간색</span> equality를 만족합니다. <span style="color: blue">파란색</span> equality는 모든 $x$가 **같을때**만 만족하게 됩니다. 위 식에서 $m=\underset{i}{max}\ x_i$이라고 한다면, 최댓값 함수의 특징에 따라서 어떤 $x_i$도 $m$보다 **작거나 같을**것입니다. 따라서 단조 증가함수인 $\exp{x}$을 적용해도 그 대소 관계는 변하지 않을것입니다.

$$
\exp{x_i}\ \le\ \exp{m} \tag{4}
$$

또한, 최댓값 $m$역시 $x_i (i=1,\ \cdots,\ n)$중 하나의 원소이기 때문에 $m$은 모든 $x_i$를 더한 값보다 **작거나 같습니다**. 그리고 모든 $x_i$의 합은 $m$을 $n$번 더한 값보다 **작거나 같습니다**.

$$
m\ \le\ \sum_{i=1}^n x_i \\
\sum_{i=1}^n x_i = (x_1+\cdots+x_n)\ \le\ (m+\cdots+m) = \sum_{i=1}^n m \tag{5}
$$

따라서 이러한 대소 관계에 다시 $\exp$함수를 적용한다면 아래와 같은 식을 얻을 수 있습니다.

$$
\exp{m}\ \le\ \sum_{i=1}^n \exp{x_i}\ \le\ n\exp{m} \label{bound_condition_2} \tag{6}
$$

식 $(\ref{bound_condition_2})$의 각 변에 $\boldsymbol{\log}$를 적용하면 식 $(\ref{bound_condition_1})$이 도출되게 됩니다. 이를 예제를 통해 비교해보겠습니다.

$$
y=max(x, 0)=\begin{cases} 0\ (x \le 0) \\ x\ (x > 0) \end{cases} \tag{7}
$$

위 식을 만족하는 최댓값 함수 $\underset{i}{max}\ x_i$와 $LSE$ 함수를 비교하려고 합니다. 하지만 위의 수식을 어디서 본 것 같지 않나요? 그렇습니다. 이전에 배운 **ReLU**(Rectified Linear Unit) 함수와 같은 식입니다. ReLU 함수는 $x$가 $0$보다 작으면 항상 그 출력 값이 0이고, $0$보다 크면 그 값이 그대로 출력되는 함수였습니다.

이 ReLU 함수와 $LSE(x, 0)$ 함수를 비교해보겠습니다. $LSE$함수의 수식은 아래와 같습니다.

$$
y=LSE(x, 0)=\log{(\exp{x}+\exp{0})}=\log{(\exp{x}+1)} \tag{8}
$$

Python을 통해서 구현한 코드와 그래프[^fn-lse]는 아래와 같습니다.

```python
from common import *

x = np.arange(-5.0, 5.0, 0.1)
y1 = relu(x)
def lse(x):
    return np.log(np.exp(x)+1)
y2 = lse(x)

plt.figure()
plt.plot(x, y1, label='relu', linestyle='--')
plt.plot(x, y2, label='lse')
plt.title('ReLU and LSE')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```

<figure>
    <img src="/posts/study/machine learning/deep learning/images/2023-01-06-neural_network_9.jpg"
          title="ReLU and LSE function"
          alt="Image of ReLU and LSE function"
          class="img_center"
          style="width: 50%"/>
     <figcaption>ReLU 함수와 LSE 함수의 비교</figcaption>
</figure>

[Fig. 1.]을 보면 지금까지 배운 LSE 함수의 특징을 알 수 있습니다. 바로 soft/smooth approximation입니다. $\underset{i}{max}\ x_i$ 함수에 부드럽게 근사하는 함수가 되는 것입니다. 즉, **미분이 불가능한** 최댓값 함수의 **미분 가능한 근사치**를 계산할 수 있게 되는 것 입니다.

따라서 **LSE** 함수가 **RealSoftMax** 함수라고 불리는 이유는
- $max$ 함수의 근삿값입니다.
- $max$ 함수의 soft/smooth approximation 함수입니다. 즉, 미분 가능한 함수입니다.

---

# Softmax activation function

지금까지 RealSoftMax, 즉 LSE(LogSumExp) 함수에 대해서 알아보았습니다. 그렇다면 흔히 신경망에서 말하는 softmax함수인 **softmax activation** 함수는 무엇일까요? 수식으로는 식 $(\ref{softmax_activation})$으로 나타낼 수 있는 softmax activation 함수는 $LSE$ 함수 $(\ref{log_sum_exp})$와 유사하게 생겨서 그 이름을 가져왔습니다. 그렇다면 어떤 부분이 다르기에 softmax activation 함수라고 부르는 것이고, 그 형태가 저렇게 되었을까요?

그것은 바로 $\boldsymbol{LSE}$의 **gradient**가 softmax activation 함수라는 점입니다. 앞서 얘기했듯이 $LSE$ 함수는 **미분이 불가능한** $max$ 함수의 smooth approximation을 통해 **미분 가능한 근삿값**을 구할 수 있게 해줍니다. 앞으로 다루겠지만 **신경망 학습**에서는 활성화 함수의 **미분값**이 굉장히 중요한 역할을 하게 됩니다. 또한 신경망의 은닉층은 여러 가지 **다변수** 뉴런으로 이루어져 있습니다. 따라서 $\boldsymbol{x}=\begin{pmatrix} x_1,\ \cdots,\ x_n \end{pmatrix}$라는 벡터에 대해서 각 변수들에 대해서 편미분을 하게 되면 아래와 같이 나옵니다.

$$
\frac{\partial}{\partial x_i}LSE(\boldsymbol{x})=\frac{\exp{x_i}}{\sum_j \exp{x_j}} \tag{9}
$$

그래서 $\boldsymbol{LSE}$ 함수의 **gradient**는 **softmax activation** 함수가 되는 것입니다.

---

# 구현 시 주의점

**Softmax activation** 함수는 **지수** 함수를 사용하기에 이론상으로는 문제가 없지만, 컴퓨터로 구현을 할 때 **오버플로우** 문제가 발생할 수도 있습니다. 지수 함수는 단조 증가함수이지만 입력 값이 커질수록 출력 값이 커지는 폭이 매우 **큽니다**. 컴퓨터로 계산하면 $\exp{1000}$만 해도 무한대를 뜻하는 **inf**가 출력됩니다. 따라서 컴퓨터로 구현할 때는 식 $(\ref{softmax_activation})$을 개선해서 사용해야 합니다. 개선한 수식은 아래와 같습니다.

$$
h(x)_i=\frac{\exp{z_i}}{\sum_{j=1}^K \exp{z_j}}=\frac{C\exp{z_i}}{C\sum_{j=1}^K \exp{z_j}} \\
=\frac{\exp{(z_i+\log{C})}}{\sum_{j=1}^K \exp{(z_j+\log{C})}}=\frac{\exp{(z_i+C')}}{\sum_{j=1}^K \exp{(z_j+C')}} \label{improved_softmax} \tag{10}
$$

임의의 상수 $C$와 $C'=\log{C}$에 대해서 항상 위 식 $(\ref{improved_softmax})$을 만족합니다. 여기서 $C'$에 어떤 값을 대입해도 그 결과는 바뀌지 않지만 일반적으로 입력 신호 중 **최댓값** $m=\max{(x_1,\ \cdots,\ x_n)}$을 사용합니다. $\boldsymbol{LSE}$ 함수는 지수 함수의 역함수인 $\boldsymbol{\log}$ 함수를 사용합니다. 하지만 $\log$ 함수는 입력 값이 커질수록 출력 값의 변화폭이 매우 **줄어듭니다**. 따라서 **정확도 개선**의 목적과 softmax activation 함수와 마찬가지로 **오버플로우(혹은 언더플로우)** 문제가 발생할 수 있기에 컴퓨터로 구현할 때에는 아래와 같은 개선된 수식을 사용합니다.

$$
\begin{gather*}
LSE\begin{pmatrix} x_1,\ \cdots,\ x_n \end{pmatrix}=\log{(\exp{x_1}+\cdots+\exp{x_n})} \\
=\log{(\exp{C}(\exp{(x_1-C)}+\cdots+\exp{(x_n-C)}))}=\log{\exp{C}}+\log(\exp{(x_1-C)}+\cdots+\exp{(x_n-C)}) \label{improved_lse} \tag{11} \\
=C+\log(\exp{(x_1-C)}+\cdots+\exp{(x_n-C)})
\end{gather*}
$$

임의의 상수 $C$에 대해서 항상 위 식 $(\ref{improved_lse})$을 만족합니다. **Softmax activation** 함수와 마찬가지로 어떤 값을 $C$로 사용해도 그 결과는 같지만 일반적으로 입력 신호 중 **최댓값**을 사용합니다.

---

# 신경망 추론(분류)

Softmax activation 함수의 지수 함수는 단조 증가함수이기에 원소들간의 대소 관계가 **변하지 않습니다**. 그래서 신경망을 이용한 **추론** 과정에서는 가장 큰 출력을 내는 뉴런에 해당하는 클래스만 사용하기에 결과적으로 추론(분류) 과정에서 출력층의 softmax 함수는 생략해도 됩니다.

## 출력층 뉴런 수 결정

위에서 말한 **출력층**의 **뉴런의 개수**를 결정하는 것은 신경망을 통해 해결하고자 하는 **문제**에 따라 다릅니다. 신경망을 이용한 **분류**에서는 *분류하고 싶은 클래스의 수*로 설정하는 것이 일반적입니다. 예를 들어 밑에서 진행할 손글씨 숫자 분류에서는 숫자를 **0에서 9** 중 하나로 분류하는 것이 목적이기에 출력층의 뉴런의 개수는 **10**개입니다.

## 손글씨 숫자 분류

기계학습의 문제 풀이는 **학습**과 **추론**의 두 단계로 나뉩니다. **학습** 단계에서 신경망 **모델**을 **학습**하고, **추론** 단계에서는 학습 단계에서 학습한 모델로 <ins>미지의 데이터</ins>에 대해서 **추론(분류)**를 수행합니다. 이 추론 과정을 **순전파**(forward propagation)이라고 합니다. 따라서 이미 학습된 매개변수(모델)를 가져와서 사용하겠습니다.

## MNIST 데이터셋

> MNIST 데이터베이스 (Modified National Institute of Standards and Technology database)는 손으로 쓴 숫자들로 이루어진 대형 데이터베이스이며<br>, 다양한 화상 처리 시스템을 트레이닝하기 위해 일반적으로 사용된다. 이 데이터베이스는 또한 기계 학습 분야의 트레이닝 및 테스트에 널리 사용된다.<br>
> <ins>출처: [위키백과](https://ko.wikipedia.org/wiki/MNIST_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4)</ins>

MNIST 데이터셋에는 **훈련 이미지**가 60,000장, **시험 이미지**가 10,000장 준비되어 있습니다. 일반적으로 훈련 이미지로 모델을 학습하고, 학습된 모델로 시험 이미지를 얼마나 정확하게 분류하는지를 평가합니다.

책에서 제공된 코드를 그대로 사용하지 않고 필요한 부분을 가져와서 구현했습니다. 자세한 코드는 Git 저장소[^fn-forward]를 통해 확인하시면 좋을 것 같습니다.

이 MNIST 데이터셋을 가지고 **추론**을 진행하는 신경망의 구조는 아래와 같습니다.

- 3층 네트워크 (입력층, 은닉층 2개, 출력층)
- 입력층의 뉴런 784개 (28x28 pixels)
- 첫번째 은닉층의 뉴런의 개수 50개
- 두번째 은닉층의 뉴런의 개수 100개
- 출력층의 뉴런 10개 (0에서 9까지의 숫자)
- 활성화 함수는 시그모이드 함수, 출력층 활성화 함수는 소프트맥스 함수

은닉층의 뉴런의 수는 **임의로** 학습된 가중치를 가지고 왔기 때문에 미리 정해져있습니다. 추후에 신경망 학습에 대해 다루면서 **은닉층의 뉴런의 수**가 학습과 모델 성능에 어떤 영향을 미치는지 알아보겠습니다. 위의 조건대로 학습된 모델을 가지고 추론을 수행한다면 아래와 같은 결과가 나옵니다.
```text
Accuracy: 0.9352
```
**10,000**장의 시험 이미지를 가지고 3층 네트워크를 통과시킨 출력과 결과를 비교해보면 약 **9352**장의 이미지는 정답과 동일한 출력이 나왔다는 의미로 볼 수 있습니다. 비율로 따지면 **93.52%**의 정확도를 가진다고 말할 수 있습니다. 또한 해당 예제[^fn-forward]에서 pixel 값의 범위를 <ins>0 ~ 255</ins>가 아닌 <ins>0.0 ~ 1.0</ins> 사이로 **정규화**(normalization)를 진행했습니다. 이렇게 신경망의 입력 데이터에 특정 변환을 가하는 것을 **전처리**(pre-processing)라고 부릅니다.

> :memo: 여기에서는 입력 이미지 데이터에 대한 **전처리** 작업으로 **정규화**를 수행한 셈입니다.<br>현업에서도 이러한 전처리를 신경망(딥러닝)에 활발하게 사용합니다. 전처리를 통해 **식별 능력을 개선**하고 **학습 속도를 높이는** 등의 사례가 많이 제시되고 있기 때문입니다.<br>
> 또한, **전처리**에는 정규화 이외에도 데이터의 전체 평균과 표준편차를 이용해 0을 중심으로 분포하도록 이동시키거나, 데이터의 확산 범위를 제한하거나, 전체 데이터를 균일하게 분포시키는 데이터 **백색화** 등이 있습니다.

위의 예제는 입력 이미지를 1장이라고 가정하고 만든 추론 과정입니다. 하지만 입력 데이터를 하나로 묶은 **배치**(batch)를 통해 계산할 때 이점을 취할 수 있습니다.

> :memo: **배치 처리**는 컴퓨터 계산에 큰 이점을 줍니다.<br>컴퓨터 입장에서는 **배열 연산**보다 상대적으로 **느린** **파일 입출력**을 통해 **적은** 이미지(**작은** 배열들)를 **많이** 연산하는 것보다 배치 처리를 통해 파일 입출력의 횟수를 **줄이고** **많은** 이미지(큰 배열들)을 **적게** 연산하는 것이 훨씬 빠릅니다.

배치 처리를 통한 추론 과정의 결과는 아래와 같습니다.
```text
Accuracy(not using batch): 0.9352
Accuracy(using batch): 0.9352
```
위의 결과를 통해 batch를 사용하거나 사용하지 않아도 같은 결과가 나오는 것을 확인할 수 있습니다.

지금까지 신경망의 기초와 이론, 순전파까지 정리해봤습니다. 신경망과 퍼셉트론의 차이에 대해서도 많이 다뤄보았습니다. 다음 포스트부터는 이러한 차이를 다시 되새겨보면서 **신경망 학습**에 대해서 본격적으로 다뤄보겠습니다.

## 신경망 요약
- 신경망에서는 활성화 함수로 시그모이드, ReLU 함수와 같은 매끄러운 함수를 사용한다.
- 기계학습 문제는 크게 회귀(regression)와 분류(classification)로 나눌 수 있다.
- 출력층의 활성화 함수로는 회귀에서는 주로 항등 함수를, 분류에서는 주로 소프트맥스 함수를 사용한다.
- 분류에서 출력층의 뉴런의 수는 분류하려는 클래스의 수와 같게 설정한다.
- 입력 데이터를 묶은 것을 배치라고 부르고, 추론 처리를 배치 단위로 진행하면 연산 속도가 훨씬 빠르다.

---

[^fn-lse]: [Github repository](https://github.com/Gyuhub/dl_scratch/blob/main/dl_scratch/ch3/view_funcs.py)에서 LSE에 관한 코드를 확인하실 수 있습니다.
[^fn-forward]: [Github repository](https://github.com/Gyuhub/dl_scratch/blob/main/dl_scratch/ch3/forward.py)에서 순전파에 관한 코드를 확인하실 수 있습니다.