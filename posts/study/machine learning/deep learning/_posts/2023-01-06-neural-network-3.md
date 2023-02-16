---
layout: post
title: Neural-Network-3
category: deep learning
post-series: Deep learning from scratch
post-order: 6
---

# 3층 신경망 구현 {#neural-network-3-layered}

지금까지 배운 여러 이론과 코드를 통해서 본격적으로 신경망을 구현해보겠습니다. 이를 위해서 **Python** 언어를 사용하고 **numpy**와 **matplotlib** 모듈만 사용할 예정입니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/neural_network_6.jpg"
          title="3-layers neural network"
          alt="Images of 3-layers neural network"
          class="img_center"
          style="width: 50%"/>
     <figcaption>3층 신경망</figcaption>
</figure>

**3층 신경망**은 위의 그림처럼 입력층은 2개, 첫번째 은닉층은 3개, 두번째 은닉층은 2개 그리고 출력층은 2개의 뉴런으로 구성할 예정입니다.
신경망에서 층이 깊어질수록 **가중치**와 **편향 매개변수**$(w,\,b)$의 개수도 많아질 것입니다. 활성화 함수의 **입력**을 계산하기 위해서 입력층의 뉴런들과 매개변수들을 일일이 곱한다면 <ins>연산량이 매우 많아지겠죠</ins>.

> :bulb: 따라서 이러한 큰 비용의 연산을 효율적으로 하기 위해서 **행렬**을 사용하고 행렬 계산에 특화되어 있는 Python 모듈이 바로 **numpy**입니다.

# 신경망 행렬

신경망 내부 연산에서 사용되는 행렬은 바로 신경망 내부층 각각의 뉴런의 정보를 담은 행렬과 매개변수들의 행렬입니다. 하나씩 차례대로 살펴보겠습니다.

## 입력층

먼저 입력층의 행렬에는 말 그대로 **입력 값**들이 들어있습니다. 예를 들어 입력이 $x_1,\,x_2$의 두 개의 입력을 받는 네트워크라면 $x=\begin{bmatrix} x_1 & x_2 \end{bmatrix}$처럼 **행 벡터**로 그 값들을 순서대로 나열하여 표현할 수 있습니다.

> :memo: **편향**의 경우에는 입력 값이 항상 **1**로 고정되어 있기에 그림에도 생략하고 입력 값의 행렬에도 포함하지 않습니다.

## 은닉층

다음으로 은닉층입니다. 사실 은닉층은 그 이름대로 **숨겨져있기에**(hidden) 내부에 어떤 값들이 들어있는지 외부에서는 알 수 없습니다. 따라서 행렬 계산 결과를 *임시적*으로 저장할 뿐 따로 그 값을 저장해서 활용하지는 않습니다.

## 출력층

출력층의 행렬에는 입력층과 같이 **출력 값**들이 들어있습니다. 예를 들어 출력이 $y_1, y_2, y_3$의 세 개의 출력이 나오는 네트워크라면 $y=\begin{bmatrix} y_1 & y_2 & y_3 \end{bmatrix}$처럼 **행 벡터**로 그 값들을 순서대로 나열해서 표현할 수 있습니다.

---

## 매개변수

신경망에서 입력이나 출력만큼 중요한 역할을 하는 매개변수입니다. 매개변수에는 **가중치 매개변수**와 **편향 매개변수**가 있습니다. 

매개변수들은 신경망층 사이의 뉴런과 뉴런 사이에서 이전층의 뉴런에서 받은 입력을 다음층의 뉴런의 활성화 함수로 넘겨주는 과정에서 **연산**에 관여합니다. 따라서 이전층과 다음층의 **뉴런 모두**에게 영향을 미치는 것입니다.

### 가중치 매개변수

먼저 가중치 매개변수의 행렬에 대해서 살펴보겠습니다. 가중치 매개변수 행렬 $w$는 아래와 같이 수식적으로 나타낼 수 있습니다.

$$
w^{(n)}=\begin{bmatrix}
w^{(n)}_{11} & w^{(n)}_{21} & \cdots & w^{(n)}_{j1} \\
w^{(n)}_{12} & w^{(n)}_{22} & \cdots & w^{(n)}_{j2} \\
\vdots & \vdots & \ddots & \vdots \\
w^{(n)}_{1i} & w^{(n)}_{2i} & \cdots & w^{(n)}_{ji} \\
\end{bmatrix} \label{weight_matrix_1} \tag{1}
$$

식 $(\ref{weight_matrix_1})$에서 가중치 $w$의 위 첨자 $\boldsymbol{(n)}$은 해당 가중치 행렬이 $\boldsymbol{n}$**번째 층**의 가중치들의 행렬임을 나타냅니다. 행렬 내부의 원소들을 통해 이전층$\boldsymbol{(n-1)}$의 입력 값들을 다음층$\boldsymbol{(n)}$의 활성화 함수의 입력을 연산합니다.

가중치 $w$의 **아래 첨자**는 이전층과 다음층의 **뉴런의 순서**를 나타내는 번호입니다. 어떤 층의 뉴런들을 가장 **위쪽부터 차례대로** 번호를 매긴다면 아래 첨자의 **앞**의 번호는 ***다음층***의 뉴런의 번호를, **뒤**의 번호는 ***이전층***의 뉴런의 번호를 나타냅니다. 따라서 식 $(\ref{weight_matrix_1})$의 아래 첨자 $i$는 이전층의 뉴런의 총 개수, $j$는 다음층의 뉴런의 총 개수를 의미합니다. 예를 들어 [[Fig. 1.]](#neural-network-3-layered)과 같은 신경망이 있을때 2층의 가중치 매개변수 행렬은 아래와 같습니다.

$$
w^{(2)}=\begin{bmatrix}
w^{(2)}_{11} & w^{(n)}_{21} \\
w^{(n)}_{12} & w^{(n)}_{22} \\
w^{(n)}_{13} & w^{(n)}_{23} \\
\end{bmatrix} \label{weight_matrix_2} \tag{2}
$$

각각의 층의 뉴런의 총 개수를 계산할 때에는 편향이 포함되지 않습니다. 그림에도 없고 수식에도 없는 이유는 다음절에서 설명하겠습니다.

### 편향 매개변수

편향 매개변수 행렬 $b$는 아래와 같이 수식적으로 나타낼 수 있습니다.

$$
b^{(n)}=\begin{bmatrix}
b^{(n)}_{1} & b^{(n)}_{2} & \cdots & b^{(n)}_{j} \end{bmatrix} \label{bias_matrix_1} \tag{3}
$$

식 $(\ref{bias_matrix_1})$에서 편향 $b$ 위 첨자 $\boldsymbol{(n)}$은 가중치 행렬의 위 첨자와 **같은 의미**를 나타냅니다. 다만 가중치 매개변수 행렬과의 차이점은 이전층의 입력이 항상 $\boldsymbol{1}$ **하나**로 고정되어 있다는 점입니다.

이전층의 입력이 $\boldsymbol{1}$ **하나**로 고정되어 있기 때문에 편향 매개변수 행렬 $b$는 항상 **행 벡터**로 이루어져 있습니다.[^fn-bias-matrix] 따라서 **아래 첨자**도 **다음층**의 뉴런의 번호만 나타내게 되는 것입니다. [위 그림](#neural-network-3-layered)의 예시상으로는 편향 매개변수 행렬은
$$b^{(2)}=\begin{bmatrix} b^{(2)}_{1} & b^{(2)}_{2} \end{bmatrix}$$로 표현이 가능합니다.

---

# 신경망 행렬 연산 구현

간단한 예시를 통해 행렬과 그 연산을 구현해보고 3층 신경망 구현으로 넘어가겠습니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/neural_network_1.jpg"
          title="example of 2-layers neural network"
          alt="Images of example of 2-layers neural network"
          class="img_center"
          style="width: 50%"/>
     <figcaption>2층 신경망 예제</figcaption>
</figure>

## 2층 신경망 예제 구현

상황은 이렇습니다.
- 2층 네트워크 (입력층, 은닉층 1개, 출력층)
- 입력층의 뉴런 $x_1, x_2$
- 은닉층의 뉴런의 개수 3개
- 출력층의 뉴런 $y_1, y_2$
- 활성화 함수는 시그모이드 함수

이러한 네트워크를 아래와 같이 구현하고 결과를 확인해보겠습니다.
```python
x = np.array([0.8, 0.6])
w1 = np.array([[0.2, 0.4, 0.6], [0.9, 0.6, 0.3]])
b1 = np.array([0.3, 0.4, 0.2])

print(x.shape)
print(w1.shape)
print(b1.shape)

a1 = np.dot(x, w1) + b1 #a1 = x*w1 + b1
z1 = sigmoid(a1)

print(a1)
print(z1)
```
### 결과
```text
(2,) # x.shape
(2, 3) # w1.shape
(3,) # b1.shape
[1.   1.08 0.86] # a1
[0.73105858 0.74649398 0.70266065] # z1
```

코드상에서 $a_1 = x w_1 + b_1$을 표현한 $a$는 이전의 post에서 표기한 임시 은닉층 변수 $a$를 가져왔습니다. 또한 $z_1 = sigmoid(a_1)$도 같은 맥락으로 가져왔습니다. 그리고 입력, 가중치, 편향등의 값은 전부 임의로 설정한 값을 사용했습니다.

---

## 3층 신경망 구현

이제 본격적으로 3층 신경망을 구현하겠습니다. 상황은 아래와 같이 요약했습니다.

- 3층 네트워크 (입력층, 은닉층 2개, 출력층)
- 입력층의 뉴런 $x_1, x_2$
- 첫번째 은닉층의 뉴런의 개수 3개
- 두번째 은닉층의 뉴런의 개수 2개
- 출력층의 뉴런 $y_1, y_2$
- 활성화 함수는 시그모이드 함수

2층 네트워크와 흐름은 크게 다르지 않고 같습니다. 딱 하나, 마지막 은닉층에서 출력층으로의 **활성화 함수**만 **시그모이드 함수**가 아닌 **항등 함수**를 사용합니다.

**항등 함수**(identity function)은 입력을 그대로 출력합니다. 수식으로 표현하면 아래와 같습니다.

$$
h(x)=x \label{identity_function} \tag{4}
$$

그림으로 표현하면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/neural_network_7.jpg"
          title="Identity function"
          alt="Image of identity function"
          class="img_center"/>
     <figcaption>활성화 함수로 사용된 항등 함수</figcaption>
</figure>

[Fig. 3.]에서 눈여겨 볼 부분은 바로 활성화 함수가 $\boldsymbol{h(x)}$가 아닌 $\boldsymbol{\sigma(x)}$로 표현되어 있다는 점입니다. 이는 **은닉층**의 활성화 함수와 다르게 **출력층**의 활성화 함수라는 부분을 강조하기 위함입니다.

> :memo: **출력층**의 **활성화 함수**는 신경망을 통해 해결하고자 하는 문제의 성질[^fn-property]에 맞게 정합니다.<br>일반적으로 회귀에는 항등 함수, 2 클래스 분류에는 시그모이드 함수, 다중 클래스 분류에는 소프트맥스 함수를 사용하는 것이 일반적입니다.

분류에서 사용하는 **소프트맥스**(softmax)함수의 수식은 아래와 같습니다.

$$
h(x)_i=\frac{\exp{z_i}}{\sum_{j=1}^K \exp{z_j}}\ \text{for}\ i=1,\cdots,K\ \text{and}\ \boldsymbol{z}=\begin{bmatrix} z_1 & \cdots & z_K \end{bmatrix} \in \mathbb{R}^K \label{softmax} \tag{5}
$$

그림으로 표현하면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/neural_network_8.jpg"
          title="Softmax function"
          alt="Image of softmax function"
          class="img_center"/>
     <figcaption>활성화 함수로 사용된 소프트맥스 함수</figcaption>
</figure>

식 $(\ref{softmax})$에서 소프트맥스 함수는 원소가 $\boldsymbol{K}$개인 입력 벡터 $\boldsymbol{z}$를 입력으로 받습니다. **소프트맥스 함수**는 **정규화된 지수 함수**로 출력층의 활성화 함수를 소프트맥스 함수로 사용하게 되면 출력 값들의 합은 **1**이 됩니다. 출력 값들의 합이 1이므로 각각의 출력 값들은 해당 출력이 정답일 확률이라고도 생각할 수 있습니다.

이제 3층 신경망을 아래와 같이 Python으로 구현하고 그 결과를 확인해보겠습니다.
```python
def init_network():
     network = {}
     network['w1'] = np.array([[0.9, 0.6, 0.3], [0.4, 0.5, 0.6]])
     network['b1'] = np.array([0.3, 0.2, 0.4])
     network['w2'] = np.array([[0.5, 0.4], [0.4, 0.5], [0.8, 0.2]])
     network['b2'] = np.array([0.5, 0.8])
     network['w3'] = np.array([[0.1, 0.2], [0.3, 0.4]])
     network['b3'] = np.array([0.7, 0.4])
     return network

def forward(network, x):
     w1, w2, w3 = network['w1'], network['w2'], network['w3']
     b1, b2, b3 = network['b1'], network['b2'], network['b3']

     a1 = np.dot(x,  w1) + b1
     z1 = sigmoid(a1)
     a2 = np.dot(z1, w2) + b2
     z2 = sigmoid(a2)
     a3 = np.dot(z2, w3) + b3
     y = identity(a3)
     return y

network = init_network()
x = np.array([0.5, 0.8])
y = forward(network, x)
print(y)
```

### 결과
```text
[1.03478215 0.90314115] # y
```

네트워크(가중치, 편향)를 임의의 값으로 초기화하고 임의의 입력값을 설쟁해서 3층 신경망을 구현했습니다. 네트워크에 입력을 전달해서 결과를 확인하는 함수의 이름이 forward()인 이유는 신호가 순방향(입력에서 출력 방향)으로 전달됨을 알리기 위함입니다. 이런 과정을 **순전파**라고 부릅니다. 앞으로의 신경망 학습에서는 역방향(backward, 출력에서 입력 방향) 처리에 대해서도 다룰 예정입니다.

---

[^fn-bias-matrix]: $1$과 어떤 실수 $b$를 곱하면 그 값은 항상 $b$가 됩니다. 그래서 편향 뉴런은 표시하지 않고 편향 매개변수 행렬만으로 그 존재를 표현할 수 있는겁니다.
[^fn-property]: 기계학습 문제는 **분류**(classification)와 **회귀**(regression)으로 나뉩니다. **분류**는 데이터가 어느 class에 속하느냐 하는 문제입니다. 예를 들어 사진 속 동물의 종류를 분류하는 문제가 여기에 속합니다. **회귀**는 입력 데이터에서 (연속적인) 수치를 예측하는 문제입니다. 사진 속 동물의 나이를 예측하는 문제가 여기에 속합니다.