---
layout: post
title: Backpropagation-3
category: deep learning
post-order: 13
---

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/backpropation-2)에서는 간단한 계산 그래프 계층(덧셈, 곱셈)의 역전파에 대해서 배웠습니다. 이번 post에서는 좀 더 심화된 계산 그래프의 계층에 대해서 설명하겠습니다.

---

# 활성화 함수 계층

이제 계산 그래프를 본격적으로 신경망에 적용해보겠습니다👊! 지금부터는 신경망을 구성하는 층(계층) 각각을 클래스 하나로 구현합니다. 우선 활성화 함수인 **ReLU**와 **Sigmoid** 계층을 구현하겠습니다.

## ReLU 계층

활성화 함수로 사용되는 **ReLU**(Rectified Linear Unit) 함수의 수식은 아래와 같습니다.

$$
y=\begin{cases}
x\ (x > 0) \\
0\ (x \le 0) \label{relu} \tag{1}
\end{cases}
$$

식 $(\ref{relu})$에 대한 미분은 아래와 같습니다.

$$
\frac{\partial{y}}{\partial{x}}=\begin{cases}
1\ (x > 0) \\
0\ (x \le 0) \label{diff_relu} \tag{2}
\end{cases}
$$

식 $(\ref{diff_relu})$에서와 같이 순전파 때의 입력인 $x$가 0보다 크면 역전파는 상류의 값을 **그대로** 하류에 흘립니다. 하지만, 순전파 때 $x$가 $0$ 이하면 역전파 때는 하류로 신호를 **보내지 않습니다**.

> ⚠️ 정확히는 $0$을 보낸다고 말할 수 있겠습니다.

계산 그래프로 나태나면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_11.jpg"
          title="Propagation of relu layer"
          alt="Image of propagation of relu layer"
          class="img_center"/>
     <figcaption>ReLU 계층의 순전파(검은색)와 역전파(회색)</figcaption>
</figure>

Python을 이용해서 ReLU 계층을 구현해보겠습니다.
```python
class ReluLayer:
    def __init__(self):
        self.mask = None # boolean numpy array. true when x <= 0, false when others

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
```

## Sigmoid 계층

활성화 함수로 사용되는 **Sigmoid** 함수의 수식은 아래와 같습니다.

$$
y=\frac{1}{1+\exp{-x}} \label{sigmoid} \tag{3}
$$

식 $(\ref{sigmoid})$를 바로 미분하기보다는 계산 그래프가 **국소적 미분**이 가능하다는 점을 살려서 이를 하나씩 먼저 분해해보겠습니다. 계산 그래프로 나타낸 **Sigmoid** 함수는 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_12.jpg"
          title="Forward propagation of sigmoid layer"
          alt="Image of forward propagation of sigmoid layer"
          class="img_center"
          style="width: 75%"/>
     <figcaption>Sigmoid 계층의 순전파</figcaption>
</figure>

[Fig. 2.]에서는 곱셈과 덧셈 노드 말고도 $\boldsymbol{\exp}$와 $\boldsymbol{/}$노드가 새롭게 등장했습니다. **순전파**에서 $\boldsymbol{\exp}$ 노드는 $y=\exp{x}$ 계산을 수행하고, $\boldsymbol{/}$ 노드는 $y=\frac{1}{x}$ 계산을 수행합니다. 이러한 **국소적 계산의 전파**를 통해 Sigmoid 계층의 순전파가 이루어집니다.

그럼 반대로 Sigmoid 계층의 역전파를 계산하기 위해서 오른쪽에서 왼쪽으로 차례대로 짚어보겠습니다.

### 나눗셈 노드

$/$ 노드, 즉 $y=\frac{1}{x}$을 미분하면 아래와 같습니다.

$$
\frac{\partial{y}}{\partial{x}}=-\frac{1}{x^2}=-y^2 \label{division} \tag{4}
$$

식 $(\ref{division})$에 따르면, 역전파 때는 상류에서 흘러온 값 $\frac{\partial{L}}{\partial{y}}$에 $-y^2$(순전파의 출력을 제곱한 후 마이너스를 붙인 값)을 곱해서 하류로 흘립니다. 계산 그래프로는 다음과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_13.jpg"
          title="Backpropagation of division node"
          alt="Image of backpropagation of division node"
          class="img_center"
          style="width: 75%"/>
     <figcaption>나눗셈 노드의 역전파</figcaption>
</figure>

### 덧셈 노드

덧셈 노드는 상류의 값을 그대로 하류로 흘러보냅니다. 계산 그래프로는 다음과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_14.jpg"
          title="Backpropagation of addition node"
          alt="Image of backpropagation of addition node"
          class="img_center"
          style="width: 75%"/>
     <figcaption>덧셈 노드의 역전파</figcaption>
</figure>

### Exponential 노드

$\exp$ 노드, 즉 $y=\exp{x}$를 미분하면 아래와 같습니다.

$$
\frac{\partial{y}}{\partial{x}}=\exp{x}=y \label{exp} \tag{5}
$$

식 $(\ref{exp})$에 따르면, 역전파 때에도 상류의 값에 순전파 때의 출력$(y=\exp{-x})$을 곱해서 하류로 전파합니다. 계산 그래프로는 다음과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_15.jpg"
          title="Backpropagation of exponential node"
          alt="Image of backpropagation of exponential node"
          class="img_center"
          style="width: 75%"/>
     <figcaption>Exponential 노드의 역전파</figcaption>
</figure>

### 곱셈 노드

곱셈 노드는 상류의 값에 순전파 때의 값을 "**서로 바꿔서**" 곱합니다. 이 예시에서는 $-1$을 곱하면 되겠습니다. 계산 그래프로는 다음과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_16.jpg"
          title="Backpropagation of multiplication node"
          alt="Image of backpropagation of multiplication node"
          class="img_center"
          style="width: 75%"/>
     <figcaption>곱셈 노드의 역전파</figcaption>
</figure>

이상으로 [Fig. 6.]과 같이 Sigmoid 계층의 역전파 계산 그래프를 완성했습니다. 계산 그래프를 잘 살펴보면 Sigmoid 계층의 역전파는 순전파의 입력 $x$와 출력 $y$만으로 계산할 수 있습니다. 그래서 계산 그래프의 중간 과정을 모두 묶어서 하나의 계층으로 완성할 수 있습니다. 계산 그래프로는 다음과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_17.jpg"
          title="Backpropagation of sigmoid layer"
          alt="Image of backpropagation of sigmoid layer"
          class="img_center"/>
     <figcaption>Sigmoid 계층의 역전파</figcaption>
</figure>

[Fig. 7.]은 [Fig. 6.]의 간소화 버전입니다. 두 계산 그래프의 결과는 **같습니다**. 따라서 간소화 버전은 역전파 과정의 중간 과정 계산들을 **생략**할 수 있어 더 **효율적**이고 노드를 그룹화하여 만든 계층은 세세한 내용을 노출하지 않아 입출력에만 **집중**할 수 있다는 중요한 이점이 있습니다. 또한 Sigmoid 계층의 역전파 출력인 $\frac{\partial{L}}{\partial{y}}y^2\exp{-x}$는 다음과 같이 정리할 수 있습니다.

$$
\begin{matrix}
\frac{\partial{L}}{\partial{y}}y^2\exp{-x}&=&\frac{\partial{L}}{\partial{y}}\frac{1}{(1+\exp{-x})^2}\exp{-x} \\
&=&\frac{\partial{L}}{\partial{y}}\frac{1}{1+\exp{-x}}\frac{\exp{-x}}{1+\exp{-x}} \\
&=&\frac{\partial{L}}{\partial{y}}y(1-y) \label{backpropagation_sigmoid} \tag{6}
\end{matrix}
$$

식 $(\ref{backpropagation_sigmoid})$과 같이 Sigmoid 계층의 역전파는 **순전파의 출력**$(y)$만으로도 계산할 수 있습니다. 이에 해당하는 계산 그래프는 다음과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_18.jpg"
          title="Backpropagation of sigmoid layer"
          alt="Image of backpropagation of sigmoid layer"
          class="img_center"/>
     <figcaption>Sigmoid 계층의 역전파: 순전파의 출력 $y$만으로 계산한 역전파</figcaption>
</figure>

Python을 이용해서 Sigmoid 계층을 구현해보겠습니다.
```python
class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
```