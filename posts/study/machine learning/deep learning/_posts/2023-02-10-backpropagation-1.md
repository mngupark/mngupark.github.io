---
layout: post
title: Backpropagation-1
category: deep learning
post-order: 11
---

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/neural-network-learning-3)에서는 **신경망 학습**에 대해서 배웠습니다. 신경망에는 적응 가능한 매개변수들이 존재하고 이를 훈련 데이터에 잘 적응시키는 것을 **신경망 학습**이라고 불렀습니다. 그리고 손실 함수를 신경망의 성능 평가 **지표**로 사용하면서 매개변수에 대한 손실 함수의 변화량(**기울기**)을 경사법으로 계산했습니다. 기울기를 통해 신경망의 매개변수를 갱신하는 만큼 기울기를 구하는 것이 매우 중요한 작업입니다. 하지만 수치 미분으로 구현한 기울기는 계산 시간이 오래 걸린다는 단점이 있었습니다. 이를 효과적으로 개선한 방법이 바로 **오차역전파법**(backpropagation)입니다.

---

# 계산 그래프

오차역전파법을 설명하는 방법에는 크게 두가지가 있습니다. 첫번째는 **수식**으로 설명하는 방법이고 두번째는 **계산 그래프**를 이용해서 설명하는 방법입니다. 첫번째 방법이 일반적인 방법으로, 정확하고 간결한 설명이지만 직관적인 이해가 어려울 수도 있습니다. 그래서 두번째 방법인 계산 그래프를 이용해서 설명하겠습니다.

> 😂 기회가 된다면 수식적으로 설명하는 방법도 다뤄보겠습니다.

**계산 그래프**(computational graph)는 계산 과정을 **그래프**로 나타낸 것입니다. 여기에서의 그래프는 바로 그래프 자료구조를 말합니다. 그래서 복수의 **노드**(node)와 **간선**(edge)으로 표현됩니다.

> 🔖 그래프 자료구조에 대한 설명은 [여기]("https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)")를 참고하시길 바랍니다.

간단하게 예시를 들어서 계산 그래프를 설명해보겠습니다.

* 👨‍💻 문제 1: A가 마트에서 1개에 100원인 도넛🍩을 2개 샀습니다. 이때 지불해야할 금액을 구하세요. 단, 소비세가 10% 부과됩니다.

사실 이 문제는 암산으로도 풀 수 있을 정도로 간단합니다. 바로 **220원**이죠. 하지만 이를 계산 그래프로 푸는 과정을 보여드리겠습니다. 계산 그래프는 계산 과정을 **노드**와 **화살표**로 표현합니다. 노드는 원(⚪)으로 표기하고 원 안에 **연산 내용**을 적습니다. 그리고 **계산 결과**를 화살표(:arrow_right:) 위에 적어 각 노드의 계산 결과가 왼쪽에서 오른쪽으로 전해지게 합니다. 문제 1을 계산 그래프로 풀면 아래 그림과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_1.jpg"
          title="Example of computational graph"
          alt="Image of example of computational graph"
          class="img_center"
          style="width: 75%"/>
     <figcaption>계산 그래프로 풀어본 문제 1</figcaption>
</figure>

[Fig. 1.]에서는 $x2$와 $x1.1$을 하나의 연산으로 취급해 원 안에 표기했습니다. 하지만 곱셈인 $x$만을 연산으로 생각할 수도 있겠죠. 이렇게 하면 [Fig. 1.]은 아래 그림처럼 $2$와 $1.1$을 각각 "도넛의 개수"와 "소비세"라는 **변수**로 취급하여 원 밖에 표기할 수 있습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_2.jpg"
          title="Example of computational graph"
          alt="Image of example of computational graph"
          class="img_center"
          style="width: 75%"/>
     <figcaption>변수를 분리한 계산 그래프 문제 1</figcaption>
</figure>

좀 더 어려운 문제를 다뤄보겠습니다.

* 👨‍💻 문제 2: B가 마트에서 1개에 100원인 도넛🍩을 2개, 150원인 컵케이크🧁를 3개 샀습니다. 이때 지불해야할 금액을 구하세요. 단, 소비세가 10% 부과됩니다.

문제 2도 계산 그래프로 풀면 아래 그림과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_3.jpg"
          title="Example of computational graph"
          alt="Image of example of computational graph"
          class="img_center"
          style="width: 75%"/>
     <figcaption>계산 그래프로 풀어본 문제 2</figcaption>
</figure>

이 문제에서는 새로운 연산 노드인 **덧셈 노드**가 등장합니다. 도넛과 컵케이크의 가격을 더해서 소비세를 계산해야 하기 때문이죠. 지금까지 살펴본 것처럼 계산 그래프를 이용한 문제풀이는 다음 흐름으로 진행합니다.

1. 계산 그래프를 구성한다.
2. 그래프에서 계산을 **왼쪽에서 오른쪽**으로 진행한다.

여기서 2번째의 "계산을 <ins>왼쪽에서 오른쪽</ins>으로 진행"하는 단계를 이전에 배웠던 **순전파**(forward propagation)라고 부릅니다. 순전파는 계산 그래프의 출발점부터 종착점으로의 전파를 말합니다. 반대로 ***오른쪽에서 왼쪽***으로 계산을 하는 것을 **역전파**(backward propagation)라고 부릅니다.

---

## 특징

계산 그래프의 특징은 "**국소적 계산**"을 전파함으로써 최종 결과를 얻는다는 점에 있습니다. 국소적이란 "*자신과 직접 관계된 작은 범위*"라는 뜻입니다. 즉, 국소적 계산은 전체적인 계산에서 어떤 일이 벌어지든 상관없이 *자신과 관계된 정보*만으로 결과를 출력할 수 있다는 것입니다. 전체적인 계산이 아무리 복잡하고 광범위하더라도 각 단계에서 하는 일은 해당 노드의 "**국소적 계산**"입니다. 국소적 계산은 전체적인 계산보다 단순하지만 그 결과를 반복적으로 전달함으로써 전체를 구성하는 복잡한 계산을 해낼 수 있습니다.

## 이점

그렇다면 이 계산 그래프를 사용했을 때의 이점은 무엇일까요? 우선 방금 설명한 **국소적 계산**이 첫번째 이점입니다. 전체적인 계산이 아무리 복잡해도 각 노드에서는 단순한 계산에 집중하여 문제를 단순화할 수 있습니다. 두번째 이점으로, 계산 그래프는 중간 계산 결과를 모두 **보관**할 수 있습니다. 예를 들어 도넛 2개까지 계산했을 때의 금액은 200원, 소비세를 더하기 전의 금액은 650원임을 보관하고 있는 셈입니다. 사실 이러한 이점보다 더 중요한 것은 바로 역전파를 통해 **미분**을 *효율적으로* 계산할 수 있는 점에 있습니다.

계산 그래프의 역전파를 설명하기 위해 문제 1을 다시 들여다 보겠습니다. 문제 1은 도넛을 2개 사서 소비세를 포함한 최종 금액을 구하는 것이었습니다. 만약 문제를 아래와 같이 바꾸면 어떻게 계산해야 할까요?

* 👨‍💻 문제 1-1: A가 마트에서 1개에 100원인 도넛🍩을 2개 사려고 합니다. 도넛의 가격이 인상된다면 최종 금액은 얼마나 인상되는지 구하세요. 단, 소비세가 10% 부과됩니다.

이 문제는 "*도넛 가격에 대한 지불 금액의 미분*"을 구하는 문제에 해당합니다. 도넛 값을 $x$, 지불 금액을 $L$이라고 했을 때 $\frac{\partial{L}}{\partial{x}}$을 구하는 것입니다. 이를 계산 그래프의 역전파를 활용해서 계산한 그림은 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_4.jpg"
          title="Example of backpropagation"
          alt="Image of example of backpropagation"
          class="img_center"
          style="width: 75%"/>
     <figcaption>계산 그래프의 역전파로 풀어본 문제 1-1</figcaption>
</figure>

[Fig. 4.]와 같이 역전파는 순전파와는 반대 방향의 화살표(회색)로 그립니다. 이 전파는 "**국소적 미분**"을 전달하고 그 미분 값은 화살표의 아래에 적습니다. 결과적으로 "**도넛 가격에 대한 지불 금액의 미분**"값은 2.2라 할 수 있습니다. 즉, 도넛의 가격이 1원 오르면 최종 금액은 2.2원 오른다는 뜻입니다. 이 예시에서는 도넛 가격에 대한 최종 금액의 미분만 구했지만, "소비세에 대한 지불 금액의 미분"이나 "도넛 개수에 대한 지불 금액의 미분"도 같은 순서로 구할 수 있습니다. 중간 계산 결과를 **보관**할 수 있기에 **다수의 미분**을 아주 효율적으로 계산할 수 있습니다. 이처럼 계산 그래프의 이점은 순전파와 역전파를 통해 **각 변수의 미분**을 *효율적으로* 구할 수 있다는 것입니다.

---

# 연쇄법칙

계산 그래프의 순전파는 왼쪽에서 오른쪽으로 전파했습니다. 평소에 계산하는 방식과 큰 차이가 없어 이해하는데 무리는 없을것입니다. 하지만, 역전파는 **국소적인 미분**을 반대 방향인 오른쪽에서 왼쪽으로 전파합니다. 국소적 미분을 전파하는 원리는 **연쇄법칙**(chain rule)에 따른 것입니다. 우선 $y=f(x)$라는 계산의 역전파를 먼저 예를 들어보겠습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_5.jpg"
          title="Backpropagation of computational graph"
          alt="Image of backpropagation of computational graph"
          class="img_center"/>
     <figcaption>$y=f(x)$에 대한 계산 그래프</figcaption>
</figure>

[Fig. 5.]와 같이 역전파의 계산 절차는 **1.** 신호 $\boldsymbol{E}$에 노드의 국소적 미분$(\frac{\partial{y}}{\partial{x}})$을 곱한 후 **2.** 다음 노드로 전달하는 것입니다. 여기에서 말하는 **국소적 미분**은 순전파 때의 $y=f(x)$ 계산의 미분을 구한다는 의미이며, 이는 $x$에 대한 $y$의 미분$(\frac{\partial{y}}{\partial{x}})$을 구한다는 뜻입니다. 그리고 이 국소적인 미분을 **상류**에서 전달된 값(이 예에서는 $\boldsymbol{E}$)에 곱해 앞쪽 노드(**하류**)로 전달하는 것입니다. 이것이 역전파의 계산 순서입니다. 왜 그런 일이 가능한가는 연쇄법칙의 원리로 설명할 수 있습니다.

## 원리

연쇄법칙에 대해 설명하려면 먼저 **합성 함수**(composite function)에 대해 설명해야 합니다. **합성 함수**란 여러 함수로 구성된 함수를 말합니다. 예를 들어 $z=(x+y)^2$이라는 식은 아래의 두 개의 식으로 구성됩니다.

$$
z = t^2 \\
t = x+y \label{composite_function} \tag{1}
$$

연쇄법칙은 합성 함수의 미분에 대한 성질이며, 다음과 같이 정의[^fn-composite-function]됩니다.

> 📚 합성 함수의 미분은 합성 함수를 구성하는 *각 함수의 미분의 곱*으로 나타낼 수 있다.

식 $(\ref{composite_function})$을 예로 설명하면, $\frac{\partial{z}}{\partial{x}}$($x$에 대한 $z$의 미분)은 $\frac{\partial{z}}{\partial{t}}$($t$에 대한 $z$의 미분)과 $\frac{\partial{t}}{\partial{x}}$($x$에 대한 $t$의 미분)의 곱으로 나타낼 수 있다는 말입니다. 수식으로 다음과 같이 쓸 수 있습니다.

$$
\frac{\partial{z}}{\partial{x}}=\frac{\partial{z}}{ {\color{Red}\partial{ {\color{Red}t} }} }\frac{ {\color{Red}\partial{ {\color{Red}t} }} }{\partial{x}} \label{ex_composite_function} \tag{2}
$$

식 $(\ref{ex_composite_function})$는 연쇄법칙을 써서 **1.** 식 $(\ref{composite_function})$의 국소적 미분(편미분)을 구하고 **2.** 두 미분을 곱해서 최종 미분값을 구할 수 있습니다. 수식으로는 아래와 같습니다.

$$
\frac{\partial{z}}{\partial{t}}=2t,\ \frac{\partial{t}}{\partial{x}}=1 \\
\frac{\partial{z}}{\partial{x}}=\frac{\partial{z}}{ {\color{Red}\partial{ {\color{Red}t} }} }\frac{ {\color{Red}\partial{ {\color{Red}t} }} }{\partial{x}}=2t \cdot 1=2(x+y) \tag{3}
$$

위 수식의 연쇄법칙 계산을 계산 그래프로 나타내 보겠습니다. 2제곱 계산을 "**2" 노드로 나타내면 아래 그림과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_6.jpg"
          title="Chain rule and computational graph"
          alt="Image of chain rule and computational graph"
          class="img_center"/>
     <figcaption>연쇄법칙과 계산 그래프</figcaption>
</figure>

[Fig. 6.]과 같이 계산 그래프에서의 역전파는 노드로 들어온 **입력 신호**에 그 **노드의 국소적 미분**(편미분)을 곱한 후 다음 노드로 전달합니다. 국소적 미분(편미분)은 <ins>**순전파에서의 입력에 대한 출력의 편미분**</ins>을 말합니다. 또한 역전파의 첫 신호인 $\frac{\partial{z}}{\partial{z}}$의 값은 $1$이라서 앞의 수식에서는 언급하지 않았습니다.

[Fig. 6.]에서 또 주목할 점은 바로 맨 왼쪽까지 흘러간 신호입니다. 이 역전파는 연쇄법칙에 따라서 $\frac{\partial{z}}{\partial{z}}\frac{\partial{z}}{\partial{t}}\frac{\partial{t}}{\partial{x}}=\frac{\partial{z}}{\partial{t}}\frac{\partial{t}}{\partial{x}}=\frac{\partial{z}}{\partial{x}}$가 성립되어 "$x$에 대한 $z$의 미분"이 됩니다. 즉, **역전파가 하는 일**은 **연쇄법칙의 원리와 같다**는 말입니다.

---

[^fn-composite-function]: 자세한 정의는 [여기]("https://en.wikipedia.org/wiki/Chain_rule")를 참고해주세요.