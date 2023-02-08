---
layout: post
title: Neural-Network-Learning-2
category: deep learning
post-order: 9
---

지난 [post]("https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/neural-network-learning-1")에서는 신경망 학습과 손실 함수에 대해서 다뤄보았습니다. 이번 post에서는 신경망 학습에서 어떻게 손실 함수를 지표로 삼아서 가중치 매개변수의 값을 학습하는지에 대해 알아보겠습니다.

# 수치 미분

신경망에서는 가중치 매개변수에 대한 손실 함수의 *변화량*을 통해서 가중치 매개변수를 변화가 없을때까지 지속적으로 개선해나갑니다. 이때 사용되는 방법이 바로 **경사법**입니다. 경사법에서는 **함수의 기울기**(경사)값을 기준으로 나아갈 방향을 정합니다. 기울기에 대해 알아보기 전에 먼저 **미분**에 대해서 다시 복습해보겠습니다.

## 미분

먼저 미분의 사전적 정의를 알아보겠습니다.

> 📗 미적분(Calculus)에서 미분(Differentiation)이란, 도함수(Derivative)를 찾는 과정을 말합니다. 도함수는 함수의 argument(입력값)에 대한 함숫값(출력값)의 변화의 민감도를 측정하는 함수를 말합니다.<br>
> <small>출처: [위키백과](https://en.wikipedia.org/wiki/Derivative)</small>

이를 한마디로 말하자면 함수의 미분은 "입력값에 대한 함숫값의 한순간의 변화량"이라고 할 수 있습니다. 수식으로 나타내면 아래와 같습니다.

$$
\frac{df(x)}{dx}=\lim_{h \to \infty} \frac{f(x+h)-f(x)}{h} \label{differentiation} \tag{1}
$$

식 $(\ref{differentiation})$은 함수의 미분을 나타낸 식입니다. 좌변은 $f(x)$의 $x$에 대한 미분을 나타내는 기호입니다. 여기에는 $x$에 대한 $f(x)$의 작은 변화량을 포함시키기에 시간에 대한 개념이 포함되어 있습니다. 그래서 $\lim_{h \to \infty}$라는 표현을 통해서 시간 $h$를 한없이 0에 가깝게 한다는 극한을 나타냅니다.

하지만 위 미분식을 바로 컴퓨터가 계산할 수 있는 식으로 나타낼 수는 **없습니다**. 왜냐하면 식 $(\ref{differentiation})$의 분모가 한없이 0에 가까워지게 계산을 해야하는데, CPU가 계산할 수 있는 부동소수점의 정확도에도 한계가 있고 반올림 오차(rounding error) 문제를 일으키는 등, 분모를 굉장히 작게 만드는 일은 결과값에 좋은 영향을 주지 않습니다. 

```python
def diff(f, x): # bad example of differentiation
    h = 1e-50
    return (f(x + h) - f(x)) / h
```

따라서 이 미분식을 먼저 컴퓨터가 잘 계산할 수 있는 미분식으로 표현하고, 그 이후에 계산을 해야합니다. 바로 이 미분식을 **수치 미분식**이라고 부릅니다.

> 💡 엄연히 말하자면 수치 해석학에서 도함수의 근삿값을 추정하는 것을 수치 미분이라고 합니다. 수치 미분의 자세한 정의에 대해서는 [여기](https://en.wikipedia.org/wiki/Numerical_differentiation)를 참고하시기 바라겠습니다.

그럼 간단한 수치 미분식을 Python을 이용해서 구현해보겠습니다.

```python
def numerical_diff(f, x): # f는 함수, x는 입력값입니다.
    h = 1e-4 # 이정도의 값을 사용하면 좋은 결과를 얻는다고 알려져 있습니다.
    return (f(x + h) - f(x)) / h
```

식으로 나타내면 아래와 같습니다.

$$
\frac{df(x)}{dx} \approx \frac{f(x+\Delta)-f(x)}{\Delta} (\Delta \approx 0) \label{forward_difference} \tag{2}
$$

$\Delta$는 0에 가까운 아주 작은 값입니다. 하지만 식 $(\ref{forward_difference})$도 **오차**가 존재합니다. 사실 식 $(\ref{differentiation})$의 진정한 값을 구하려면 $x$ 위치의 $f(x)$의 **기울기**를 구해야 합니다. 하지만 위 식은 $x$와 $x+\Delta$ 사이의 기울기에 해당합니다. 그래서 진정한 미분값과 일치하지는 않습니다. 이 차이는 $h$(혹은 $\Delta$)를 무한히 0에 가깝게 만들어서 $f(x)$의 극한값을 컴퓨터로 계산하는 일은 불가능하기 때문에 생기는 한계입니다.

이를 좀 더 개선할 수 있는 방법은 바로 **중심 차분**(혹은 중앙 차분)을 사용하는 것입니다. $x$와 $x+h$ 사이의 기울기가 아닌 $x+h$와 $x-h$의 기울기를 구하는 것입니다. 이 차분은 $x$를 중심으로 그 전후의 차분을 계산한다고 해서 **중심 차분**이라고 부르고 이전의 방식은 **전방 차분**이라고 부릅니다. 그러면 이 개선된 수치 미분식을 Python을 이용해서 구현해보겠습니다.

> 1. **차분**이란 임의의 두 점에서 함숫값들의 차이를 말합니다.
> 2. 수식을 전개해 미분하는 것을 **해석적** 미분, 아주 작은 차분으로 미분값의 근삿값을 계산하는 것을 **수치적** 미분이라고 간단하게 요약할 수 있겠습니다.<br>
> 그래서 해석적 미분값은 오차가 포함되어 있지 않은 **진정한 미분값**입니다.

```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
```

또한, 그림을 통해서 함수 $f(x)=$에 대한 1. 해석적 미분, 2. 중앙 차분, 3. 전방 차분을 표현해보겠습니다.

<figure>
    <img src="/posts/study/machine%20learning/deep%20learning/images/2023-02-08-neural-network-learning-1_2.jpg"
         title="Various differentiations"
         alt="Image of various differentiations"
         class="img_center"
         style="width: 50%"/>
    <figcaption>함수 $f(x)$에 대한 다양한 미분들</figcaption>
</figure>

그림을 통해서 해석적 미분값을 이용한 접선에 중앙 차분으로 계산한 수치 미분값의 접선은 거의 동일한 것을 확인할 수 있습니다. 하지만 전방 차분으로 계산한 수치 미분값의 접선은 약간의 오차가 존재하는 것을 확인할 수 있습니다. 