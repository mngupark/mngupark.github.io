---
layout: post
title: Perceptron-1
category: deep learning
post-order: 2
post-series: Deep learning from scratch
---

# 딥러닝의 출발점

딥러닝에서 복잡하게 쌓아 올린 인공 신경망 (Neural Network)은 전통적인 머신 러닝과 딥러닝을 구별하게 하는 수많은 머신 러닝 방법 중 하나입니다.
즉, 딥러닝을 이해하기 위해서는 인공 신경망을 이해해야 하고 인공 신경망을 이해하기 위해서는 우선 퍼셉트론이라는 것을 이해해야 합니다.
퍼셉트론(Perceptron)은 프랑크 로젠블라트가 1957년에 고안한 알고리즘입니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/frank-rosenblatt.jpg" 
          title="Frank Rosenblatt"
          alt="Frank Rosenblatt"
          class="img_center"/>
     <figcaption>Frank Rosenblatt</figcaption>
</figure>

퍼셉트론은 신경망(딥러닝)의 기원이 되는 알고리즘이기에 이 구조를 배우는 것은 신경망과 딥러닝으로 나아가는 중요한 시작점이라고 할 수 있습니다.

# 퍼셉트론이란? {#perceptron}

퍼셉트론은 다수의 **신호**를 입력으로 받아 하나의 **신호**를 출력합니다. 여기서 **신호**란 전류나 강물처럼 흐름이 있는 것을 비유할 수 있습니다.
하지만 퍼셉트론은 전류와 달리 *'흐른다/안 흐른다(1이나 0)'*의 두 가지 값을 가질 수 있습니다. 앞으로도 1을 '신호가 흐른다', 0을 '신호가 흐르지 않는다'라는 의미로 해석하겠습니다.

---

<figure>
     <img src="/posts/study/machine learning/deep learning/images/perceptron_1.jpg"
          title="Perceptron-1"
          alt="Perceptron-1"
          class="img_center"/>
     <figcaption>퍼셉트론의 예시</figcaption>
</figure>

위의 그림은 입력이 2개인 퍼셉트론의 한 예시입니다.

$x_1$과 $x_2$는 **입력 신호** , $y$는 **출력 신호**, $w_1$과 $w_2$는 **가중치**를 뜻합니다. 그리고 그림에 있는 원을 **뉴런** 혹은 **노드**라고 부릅니다.
입력 신호가 뉴런에 보내질 때는 각각의 고유한 **가중치**가 곱해집니다. 그리고 뉴런에서 보내온 신호의 총합이 **정해진 한계**를 넘어설 때만 **1**을 출력하게 됩니다.

> 이를 '뉴런이 활성화한다'라고 표현하기도 합니다.

여기선 이 정해진 한계를 임계값이라고 하며 $\theta$로 표현합니다. 이를 수식으로 나타내면 아래와 같습니다.

$$
y=\begin{cases}
0\ & (w_1 x_1 + w_2 x_2 \le \theta) \\
1\ & (w_1 x_1 + w_2 x_2 > \theta)
\end{cases} \label{perceptron_2} \tag{1}
$$

퍼셉트론은 복수의 입력 신호 **각각에** 고유한 **가중치**를 부여합니다. **가중치**는 각 신호가 결과에 주는 영향력을 조절하는 요소로 작용합니다.

---

# 퍼셉트론 구현

이러한 퍼셉트론을 활용한 예시로 논리 게이트를 구현할 수 있습니다. 구현해 볼 논리 게이트는 AND, NAND 그리고 OR 게이트입니다.

- AND 게이트

아래에는 해당 게이트에 대한 진리표와 Python 코드가 있습니다.

<table style="margin-left: auto; margin-right: auto; width: 30%;">
  <caption>AND 게이트의 진리표</caption>
  <tr><th>$x_1$</th> <th>$x_2$</th> <th>$y$</th></tr>
  <tr><td>0</td> <td>0</td> <td>0</td></tr>
  <tr><td>1</td> <td>0</td> <td>0</td></tr>
  <tr><td>0</td> <td>1</td> <td>0</td></tr>
  <tr><td>1</td> <td>1</td> <td>1</td></tr>
</table>

```python
def and_gate(x1, x2):
     w1, w2, theta = 0.5, 0.5, 0.7
     tmp = x1*w1 + x2*w2
     if tmp <= theta:
          return 0
     elif tmp > theta:
          return 1
```
- NAND 게이트

NAND 게이트를 구현하기 전에 퍼셉트론의 [수식](#perceptron)을 조금 수정하겠습니다.

$$
y=\begin{cases}
0\ & ({\color{yellow}b} + w_1 x_1 + w_2 x_2 \le 0) \\
1\ & ({\color{yellow}b} + w_1 x_1 + w_2 x_2 > 0)
\end{cases} \label{perceptron_3} \tag{2}
$$

크게 달라질 것은 없고 이전의 수식 $(\ref{perceptron_2})$에서 $\color{yellow}\theta$를 $\color{yellow}{-b}$로 표기만 바꿨을 뿐, 의미는 같습니다.
앞으로는 **임계값** $\theta$라는 표현 대신 **편향** $b$라는 표현으로 바꿔서 사용하겠습니다.

> :bulb: **편향** $b$는 **가중치** $w$와 기능이 다릅니다! **가중치**는 입력 신호가 결과에 주는 *영향력*을 조절하는 것이고, **편향**은 뉴런이 얼마나 쉽게 *활성화*하느냐를 조정하는 매개변수 입니다.

그럼 수정된 수식 $(\ref{perceptron_3})$을 통한 해당 게이트에 대한 진리표와 Python 코드가 아래에 있습니다.

<table style="margin-left: auto; margin-right: auto; width: 30%;">
  <caption>NAND 게이트의 진리표</caption>
  <tr><th>$x_1$</th> <th>$x_2$</th> <th>$y$</th></tr>
  <tr><td>0</td> <td>0</td> <td>1</td></tr>
  <tr><td>1</td> <td>0</td> <td>1</td></tr>
  <tr><td>0</td> <td>1</td> <td>1</td></tr>
  <tr><td>1</td> <td>1</td> <td>0</td></tr>
</table>

> :memo: 여기서부터는 Python의 <mark>numpy</mark> 모듈을 사용했습니다.

```python
def nand_gate(x1, x2):
     x = np.array([x1, x2])
     w = np.array([-0.5, -0.5])
     b = 0.7 # b = -theta
     tmp = np.sum(w*x) + b
     if tmp <= 0:
          return 0
     elif tmp > 0:
          return 1
```

- OR 게이트

아래에는 해당 게이트에 대한 진리표와 Python 코드가 있습니다.

<table style="margin-left: auto; margin-right: auto; width: 30%;">
  <caption>OR 게이트의 진리표</caption>
  <tr><th>$x_1$</th> <th>$x_2$</th> <th>$y$</th></tr>
  <tr><td>0</td> <td>0</td> <td>0</td></tr>
  <tr><td>1</td> <td>0</td> <td>1</td></tr>
  <tr><td>0</td> <td>1</td> <td>1</td></tr>
  <tr><td>1</td> <td>1</td> <td>1</td></tr>
</table>

```python
def or_gate(x1, x2):
     x = np.array([x1, x2])
     w = np.array([0.5, 0.5])
     b = -0.3 # b = -theta
     tmp = np.sum(w*x) + b
     if tmp <= 0:
          return 0
     elif tmp > 0:
          return 1
```

역할이 각각 다른 3가지 논리 게이트를 구현하면서, 우리는 한가지 중요한 사실을 알 수 있습니다.
바로 3가지 게이트 모두 다 **똑같은** 퍼셉트론의 구조이지만 **가중치**와 **편향** 매개변수의 <ins>값을 조절</ins>하면서 역할이 각각 다른 게이트를 구현할 수 있었다는 사실입니다.

기계 학습 문제란 바로 이러한 <mark>매개변수의 값</mark>을 정하는 작업을 컴퓨터가 자동으로 할 수 있게 하는 것입니다!
**학습**이란 적절한 <mark>매개변수</mark>를 정하는 작업이며, 사람은 **퍼셉트론의 구조** (<mark>모델</mark>)을 만들고 컴퓨터에 학습할 **데이터**를 주는 일을 합니다.