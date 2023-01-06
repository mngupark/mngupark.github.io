---
layout: post
title: Neural-Network-2
category: deep learning
post-order: 4
---
# 활성화 함수

## 계단 함수

이전의 [post](https://gyuhub.github.io/deep%20learning/2023/01/02/neural-network-1/)에서 배웠던 활성화 함수는 임계값을 경계로 출력이 0에서 1로 바뀝니다. 이러한 함수를 <ins>계단 함수</ins>(step function)라고 합니다.

> :bulb: 퍼셉트론은 활성화 함수로 계단 함수를 사용한다고 할 수 있습니다.

그렇다면 계단 함수가 아닌 다른 함수를 활성화 함수로 사용한다면 어떻게 될까요?

## 시그모이드 함수

신경망에서 자주 사용되는 활성화 함수인 <ins>시그모이드 함수</ins>(sigmoid function)를 아래의 수식으로 나타낼 수 있습니다. 시그모이드란 *"S자 모양"*이라는 의미입니다.

$$
h(x) = \frac{1}{1+\exp(-x)} \label{sigmoid} \tag{1}
$$

신경망에서는 입력 신호를 시그모이드 함수를 활성화 함수로 사용해서 변환된 신호를 다음 뉴런으로 전달합니다. 이전에 다룬 퍼셉트론과 앞으로 다룰 신경망의 주된 차이는 바로 이 **활성화 함수**뿐입니다.

> 일반적으로<br>**단순 퍼셉트론**은 단층 네트워크에서 계단 함수를 활성화 함수로 사용한 모델을 가리키고,<br>
> **다층 퍼셉트론**은 **신경망**(여러 층으로 구성되고 시그모이드 함수 등의 매끈한 활성화 함수를 사용하는 네트워크)을 가리킵니다.

계단 함수와 시그모이드 함수의 비교를 통해 활성화 함수에 대해서 좀 더 파고들어 보겠습니다.

---

## 구현

Python을 이용해서 계단 함수와 시그모이드 함수를 구현해보겠습니다.

### step function
```python
def step(x):
    y = x > 0
    return y.astype(np.int)
```

### sigmoid function
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

위의 두 코드가 계단 함수와 시그모이드 함수를 구현한 Python 코드입니다. 수식이 어렵지 않아 구현에 있어 큰 어려움은 없지만, 두 함수 둘 다 입력이 **numpy** 모듈의 다차원 배열 $x$라는 점을 고려해서 코드를 작성했다는 것을 기억해주시면 좋겠습니다. 자세한 구현 코드는 여기[^fn-functions]를 참고해주시면 감사하겠습니다.

## 결과 비교

이제 그래프를 통해 두 함수를 비교해보겠습니다.

<figure>
    <img src="/assets/images/study/machine_learning/deep_learning/2023-01-05-neural_network_4.jpg"
          title="Step and Sigmoid functions"
          alt="Images of Step and Sigmoid functions"
          class="img_center"
          style="width: 50%"/>
     <figcaption>계단 함수(점선)와 시그모이드 함수(실선)</figcaption>
</figure>

가장 먼저 확연하게 보이는 차이는 바로 ***매끄러움***입니다. 시그모이드 함수는 부드러운(계단 함수에 비해) 곡선 형태이며 입력에 따라서 출력이 연속적으로 변화합니다.
하지만 계단 함수는 각진(시그모이드 함수에 비해) 직선이 합쳐진 형태이며 0을 경계로 출력이 0에서 1로 급격하게 변화합니다. 시그모이드 함수의 이 매끈함이 신경망 학습에서는 매우 중요합니다.

> 개인적으로 이 **매끄러움**이 신경망에서 중요한 이유는 바로 <mark>함수의 미분</mark>때문이라고 생각합니다.<br>
> 함수가 매끄럽다라는 것은 입력과 출력의 관계가 *연속적이다*라고 말할 수 있을 것 같습니다. 즉, 연속적인 함수는 미분 가능한 함수의 중요한 조건 중 하나이며,<br>
> 활성화 함수의 미분을 통해 *가중치 매개변수의 변화량*을 측정하여 신경망 학습에 활용하기 때문에 매끄러움이 중요하다고 생각합니다.

또다른 특징으로는 두 함수 둘 다 입력이 작으면 출력은 0에 가깝고(혹은 0이며), 입력이 클수록 출력은 1에 가까운(혹은 1인) 구조입니다. 또한 두 함수 둘 다 **비선형 함수**입니다.

신경망에서는 활성화 함수로 이러한 **비선형 함수**를 사용해야 합니다. **선형 함수**를 활성화 함수로 사용하게 되면 신경망의 층을 아무리 깊게 하여도 <ins>은닉층이 없는 단층 네트워크</ins>로 똑같이 구현할 수 있습니다.

예를 들면 입력이 $x$일때 $h(x)=cx+d, (c,d\in\mathbb{R})$라는 선형 함수를 활성화 함수로 사용한 3층 네트워크를 상상해 봅시다. 그렇다면 출력 $y$는 아래와 같이 표현이 가능할 것입니다.

$$
y=h(h(h(x)))=c(c(cx+d)+d)+d=c^3x+c^2d+cd+d \label{linear_activation_function_example1} \tag{2}
$$

식 $(\ref{linear_activation_function_example1})$를 보면 출력이 복잡해 보이지만 아래와 같이 분리한다면 결국 다른 **선형 함수**를 활성화 함수로 이용한 <ins>은닉층이 없는</ins> 단층 네트워크로 표현이 가능합니다.

$$
y=\bar{h}(x) (\bar{h}(x)=\bar{c}x+\bar{d}, \bar{c}=c^3, \bar{d}=c^2d+cd+d) \tag{3}
$$

따라서 신경망에서 *은닉층을 여러 층*으로 구성하는 이점을 살리고 싶다면 **활성화 함수**는 반드시 **비선형 함수**를 사용해야 합니다.

---

## ReLU 함수

시그모이드 함수는 신경망 분야에서 오래전부터 이용해왔으나, 최근에는 <ins>ReLU 함수</ins>(Rectified Linear Unit function)를 주로 이용합니다.

ReLU 함수는 입력이 0을 넘으면 그 입력을 *그대로* 출력하고, 0 이하이면 0을 출력하는 함수입니다. 수식으로 나타내면 아래와 같습니다.

$$
h(x)=\begin{cases}
x\,(x > 0) \\
0\,(x \le 0) \end{cases} \label{relu} \tag{4}
$$

## 구현

Python을 이용해서 ReLU 함수를 구현해보겠습니다.

### ReLU function
```python
def relu(x):
    return np.maximum(x, 0)
```

## 결과

그래프를 통해 ReLU 함수를 확인해보겠습니다.

<figure>
    <img src="/assets/images/study/machine_learning/deep_learning/2023-01-05-neural_network_5.jpg"
          title="ReLU functions"
          alt="Image of ReLU functions"
          class="img_center"
          style="width: 50%"/>
     <figcaption>ReLU 함수</figcaption>
</figure>

식 $(\ref{relu})$와 그래프를 확인해보면 ReLU 함수는 비교적 간단한 함수입니다. ReLU 함수도 위의 함수들과 마찬가지로 **비선형 함수**이고, **매끄럽습니다**(계단 함수에 비해).
마지막으로 이러한 활성화 함수들을 표를 통해 공통점과 차이점을 확인해보겠습니다.

<table class="aligned-center">
  <caption>신경망에 사용되는 활성화 함수들</caption>
  <tr><th></th><th>Step</th> <th>Sigmoid</th> <th>ReLU</th></tr>
  <tr><th>함수 종류</th> <td>비선형</td> <td>비선형</td> <td>비선형</td> </tr>
  <tr><th>매끄러움의 정도</th> <td>안 매끄러움</td> <td>매끄러움</td> <td>덜 매끄러움</td> </tr>
  <tr><th>미분 가능성</th> <td>$x=0$에서 불가능</td> <td>가능</td> <td>$x=0$에서 불가능</td> </tr>
  <tr><th>출력 값의 범위</th> <td>$[0,1]$</td> <td>$[0,1]$</td> <td>$[0,\infty)$</td> </tr>
</table>

---

[^fn-functions]: 앞으로 이 [github repository](https://github.com/Gyuhub/dl_scratch.git)에서 관련된 예제들과 구현 코드들을 다룰 예정입니다.