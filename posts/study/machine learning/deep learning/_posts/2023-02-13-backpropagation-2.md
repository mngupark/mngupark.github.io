---
layout: post
title: Backpropagation-2
category: deep learning
post-series: Deep learning from scratch
post-order: 12
---

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/backpropation-1)에서는 **계산 그래프**의 **연쇄 법칙**에 대해서 배웠습니다. 이번 post에서는 역전파의 구조를 설명하겠습니다.

---

# 역전파

이전에 배웠던 예시의 덧셈과 곱셉 노드의 역전파에 대해서 알아보겠습니다.

## 덧셈 노드

우선 덧셈 노드의 역전파입니다. 간단한 덧셈 노드와 그 노드의 편미분에 대한 수식은 아래와 같습니다.

$$
z=x+y \\
\frac{\partial{z}}{\partial{x}}=1,\ \frac{\partial{z}}{\partial{y}}=1 \label{add_node} \tag{1}
$$

식 $(\ref{add_node})$에서와 같이 덧셈 노드의 편미분 $\frac{\partial{z}}{\partial{x}}$와 $\frac{\partial{z}}{\partial{y}}$는 모두 $\boldsymbol{1}$이 됩니다. 이를 계산 그래프를 통해 표현하면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_7.jpg"
          title="Propagation of addition node"
          alt="Image of propagation of addition node"
          class="img_center"/>
     <figcaption>덧셈 노드의 순전파(검은색)와 역전파(회색)</figcaption>
</figure>

[Fig. 1.]과 같이 덧셈 노드의 역전파에서는 상류에서 전해진 미분$(\frac{\partial{L}}{\partial{z}})$에 1을 곱하여 하류로 흘립니다. 즉, 덧셈 노드의 역전파는 **입력된 값**을 **그대로** 다음 노드로 보내게 됩니다. 상류에서 전해진 미분 값을 $\frac{\partial{L}}{\partial{z}}$이라 지칭한 이유는 최종적으로 $L$이라는 값을 출력하는 큰 계산 그래프를 가정하기 때문입니다. 그림으로 나타내면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_8.jpg"
          title="Local differentiation"
          alt="Image of local differentiation"
          class="img_center"
          style="width: 60%"/>
     <figcaption>국소적 미분의 전파</figcaption>
</figure>

## 곱셈 노드

이어서 곱셈 노드의 역전파입니다. 덧셈 노드와 같이 간단한 곱셈 노드와 그 노드의 편미분에 대한 수식은 아래와 같습니다.

$$
z=xy \\
\frac{\partial{z}}{\partial{x}}=y,\ \frac{\partial{z}}{\partial{y}}=x \label{mul_node} \tag{2}
$$

식 $(\ref{mul_node})$에서와 같이 곱셈 노드의 편미분 $\frac{\partial{z}}{\partial{x}}$와 $\frac{\partial{z}}{\partial{y}}$는 각각 $y$와 $x$가 됩니다. 이를 계산 그래프를 통해 표현하면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_9.jpg"
          title="Propagation of multiplication node"
          alt="Image of propagation of multiplication node"
          class="img_center"/>
     <figcaption>곱셈 노드의 순전파(검은색)와 역전파(회색)</figcaption>
</figure>

[Fig. 3.]과 같이 곱셈 노드의 역전파에서는 상류에서 전해진 미분에 순전파 때의 입력 신호들을 "**서로 바꾼 값**"을 곱해서 하류로 보냅니다. **덧셈 노드**의 역전파에서는 상류의 값을 그대로 흘려보내서 순방향 입력 신호(순전파)의 값은 필요하지 않았습니다만, **곱셈 노드**의 역전파에서는 순방향 입력 신호의 값이 필요합니다. 그래서 곱셈 노드를 구현할 때는 순전파의 입력 신호를 변수에 저장해둡니다.

간단한 예시로 이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/backpropation-1)에서 다뤘던 문제의 역전파를 계산 그래프로 표현해보겠습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_10.jpg"
          title="Example of backpropagation of doughnut shopping"
          alt="Image of example of backpropagation of doughnut shopping"
          class="img_center"
          style="width: 75%"/>
     <figcaption>도넛 쇼핑의 역전파</figcaption>
</figure>

또한 이 예시를 Python을 사용해서 덧셈 노드와 곱셈 노드의 **계층**들을 구현해보도록 하겠습니다.

> 💡 **계층**이란 신경망의 기능 단위입니다. 신경망을 구성하는 계층 각각을 하나의 클래스로 구현하면 신경망의 구성에 의존적이지 않은 일반적인 코드를 작성할 수 있습니다. 예를 들면 시그모이드 함수를 위한 Sigmoid, 행렬 곱셈을 위한 Affine 등의 기능을 계층 단위로 구현하는 것입니다.

모든 계층은 **forward()**와 **backward()**라는 공통의 method를 갖도록 구현할 것입니다. 이 method들은 각각 순전파와 역전파를 처리하는 역할을 합니다.

* 곱셈 노드(**MulLayer**)
```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
```
* 덧셈 노드(**AddLayer**)
```python
class AddLayer:
    def __init__(self):
        pass # do not anything

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
```