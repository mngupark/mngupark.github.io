---
layout: post
title: Perceptron-2
category: deep learning
post-order: 2
---

# 퍼셉트론의 한계

지금까지 AND, NAND, OR 총 3가지의 게이트를 살펴봤습니다. 다음으로 살펴볼 게이트는 XOR 게이트입니다.

XOR 게이트는 **베타적 논리합**이라는 논리 회로입니다. $x_1$과 $x_2$ 중 한 쪽이 1일때만 1을 출력합니다.

해당 게이트에 대한 진리표는 아래와 같습니다.

<table style="margin-left: auto; margin-right: auto; width: 30%;">
  <caption>XOR 게이트의 진리표</caption>
  <tr><th>$x_1$</th> <th>$x_2$</th> <th>$y$</th></tr>
  <tr><td>0</td> <td>0</td> <td>0</td></tr>
  <tr><td>1</td> <td>0</td> <td>1</td></tr>
  <tr><td>0</td> <td>1</td> <td>1</td></tr>
  <tr><td>1</td> <td>1</td> <td>0</td></tr>
</table>

결론부터 말씀드리자면 지금까지 다룬 **단층 퍼셉트론**만으로는 XOR 게이트를 구현할 수 없습니다.

**단층 퍼셉트론**(single-layer perceptron)은 입력 신호와 가중치 그리고 편향의 선형 조합을 통해 나타낼 수 있습니다.

즉, $$y=b+w_1*x_1+w_2*x_2$$라는 선형 방정식으로 원하는 모든 출력 값을 조절하는 것입니다.

하지만 XOR 게이트는 선형이 아닌 비선형 방정식으로 구현이 가능합니다.

# 다층 퍼셉트론

이 문제를 해결할 방법은 없는 걸까요? 아닙니다. 바로 **다층 퍼셉트론** (multi-layer perceptron)을 이용하면 해결할 수 있습니다.

XOR 게이트는 여러가지 방법으로 구현할 수 있습니다. 결국 XOR 게이트의 진리표를 만족하는 모든 게이트는 XOR 게이트라고 부를 수 있겠죠?

그 많은 방법 중 아래와 같이 AND, NAND, OR 게이트를 하나씩 조합해서 만드는 방법이 있습니다.

<center>

    sample image of XOR gate

</center>

위의 게이트 조합대로 NAND 게이트의 출력을 $s_1$, OR 게이트의 출력을 $s_2$라고 한다면 새롭게 만들어진 XOR 게이트의 진리표는 아래와 같습니다.

<table style="margin-left: auto; margin-right: auto; width: 30%;">
  <caption>XOR 게이트의 진리표</caption>
  <tr><th>$x_1$</th> <th>$x_2$</th> <th>$s_1$</th> <th>$s_2$</th> <th>$y$</th></tr>
  <tr><td>0</td> <td>0</td> <td>1</td> <td>0</td> <td>0</td></tr>
  <tr><td>1</td> <td>0</td> <td>1</td> <td>1</td> <td>1</td></tr>
  <tr><td>0</td> <td>1</td> <td>1</td> <td>1</td> <td>1</td></tr>
  <tr><td>1</td> <td>1</td> <td>0</td> <td>1</td> <td>0</td></tr>
</table>

이어서 위와 같이 조합된 XOR 게이트를 Python으로 구현해보겠습니다.

```python
def xor_gate(x1, x2):
     s1 = nand_gate(x1, x2)
     s2 = or_gate(x1, x2)
     y = and_gate(s1, s2)
     return y
```

이러한 XOR 게이트를 뉴런을 이용한 퍼셉트론으로 표현한다면 아래 그림과 같습니다.

<center>

    perceptron of XOR gate

</center>

왼쪽부터 차례대로 각각의 뉴런의 열들을 0층, 1층, 2층이라고 부릅니다. 층이 여러 개인 이런 퍼셉트론을 **다층 퍼셉트론**이라고 부릅니다.

반대로 층이 1개인 퍼셉트론을 **단층 퍼셉트론**이라고 부릅니다.

이렇게 *단층 퍼셉트론으로 표현하지 못했던 것을 층을 하나 늘려서* 비선형 구조인 XOR 게이트를 표현할 수 있었습니다.

퍼셉트론의 가능성은 바로 층을 쌓아 (깊게 하여) 더 다양한 것을 표현할 수 있다는 점에 있습니다!

퍼셉트론은 비교적 간단한 알고리즘이라서 그 구조를 쉽게 이해할 수 있었습니다. 또한 이 퍼셉트론이 신경망의 기초가 되기에 항상 그 원리를 새겨놓으면 좋습니다.

## 퍼셉트론 요약
- 퍼셉트론은 입출력을 갖춘 알고리즘. 입력 값을 정해진 규칙(가중치, 편향)에 따라서 출력한다.
- 단층 퍼셉트론은 선형 영역을 표현할 수 있다.
- 다층 퍼셉트론은 단층 퍼셉트론을 쌓아서 만들 수 있고, 비선형 영역도 표현할 수 있다.