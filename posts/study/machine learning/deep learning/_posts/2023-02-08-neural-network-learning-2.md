---
layout: post
title: Neural-Network-Learning-2
category: deep learning
post-order: 9
---

지난 [post]("https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/neural-network-learning-1")에서는 신경망 학습과 손실 함수에 대해서 다뤄보았습니다. 이번 post에서는 신경망 학습에서 어떻게 손실 함수를 지표로 삼아서 가중치 매개변수의 값을 학습하는지에 대해 알아보겠습니다.

---

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

이를 좀 더 개선할 수 있는 방법은 바로 **중심 차분**(혹은 중앙 차분)을 사용하는 것입니다. $x$와 $x+h$ 사이의 기울기가 아닌 $x+h$와 $x-h$의 기울기를 구하는 것입니다. 이 차분은 $x$를 중심으로 그 전후의 차분을 계산한다고 해서 **중심 차분**이라고 부르고 이전의 방식은 **전방 차분**이라고 부릅니다. 식으로 표현하면 아래와 같이 표현할 수 있겠습니다.

$$
\frac{df(x)}{dx} \approx \frac{f(x+\Delta)-f(x-\Delta)}{2\Delta} (\Delta \approx 0) \label{center_difference} \tag{3}
$$

그러면 이 개선된 수치 미분식을 Python을 이용해서 구현해보겠습니다.

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
    <img src="/posts/study/machine%20learning/deep%20learning/images/2023-02-08-neural-network-learning_2.jpg"
         title="Various differentiations"
         alt="Image of various differentiations"
         class="img_center"
         style="width: 50%"/>
    <figcaption>함수 $f(x)$에 대한 다양한 미분들</figcaption>
</figure>

그림을 통해서 해석적 미분값을 이용한 접선에 중앙 차분으로 계산한 수치 미분값의 접선은 거의 동일한 것을 확인할 수 있습니다. 하지만 전방 차분으로 계산한 수치 미분값의 접선은 약간의 오차가 존재하는 것을 확인할 수 있습니다.

---

## 편미분

위에서 다룬 미분들은 독립변수가 **한 개**인 함수입니다. 하지만 **편미분**은 독립변수가 한 개가 아닌 **두 개 이상**을 가진 다변수 함수에 대한 미분을 말합니다. 즉, 어떤 한 독립변수 이외에 변수들은 다 **상수**로 간주하고 미분해서 얻은 도함수를 편도함수라 부르며, 이 편도함수를 구하는 과정을 편미분이라고 부릅니다. 신경망에서 가중치 매개변수들이 서로 독립적으로 계산될 것을 생각하면, 앞으로 편미분을 잘 다뤄야 한다는 것을 직감적으로 알게되는 순간입니다. 수식으로는 $\frac{\partial{f(x)}}{\partial x}$로 나타냅니다.

> 편미분에 대한 자세한 정의는 [여기](https://en.wikipedia.org/wiki/Partial_derivative)를 참고하시길 바랍니다.

해석적으로 편미분을 계산하는 방법은 편미분하려는 독립변수 이외에는 상수로 간주하고 미분하기 때문에 일반적인 미분과 동일합니다. 그렇다면 이를 수치적으로 미분하려면 어떻게 해야 할까요? 수치 편미분도 해석적 편미분의 방식처럼 1. 다변수 함수에 편미분하려는 독립변수 이외에는 해당 입력을 그대로 넣고(상수 취급), 2. 수치 미분 함수에 이 함수와 독립변수의 값을 입력하면 됩니다.

예를 들어서 다변수 함수 $f(x_0, x_1)=x_0^2+x_1^2$에 대해서 $(x_0, x_1)=(3,4)$인 점에 대해서 수치 편미분을 Python으로 구현하면 아래와 같습니다.

```python
def func_x0(x0): # for df/dx0
    return (x0 ^ 2 + 4.0 ^ 2)
def func_x1(x1): # for df/dx1
    return (3.0 ^ 2 + x1 ^ 2)
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
df_dx0 = numerical_diff(func_x0, 3.0)
df_dx1 = numerical_diff(func_x1, 4.0)
```

---

# 기울기

이처럼 편미분 또한 단일변수 함수의 미분과 마찬가지로 특정한 장소의 **기울기**를 구합니다. 그래서 여러 변수 중 관심있는 변수 하나에 초점을 맞추고 다른 변수들은 값을 **고정**합니다. 그렇지만 위의 방식은 일일이 변수별로 편미분을 따로 계산했습니다. 그럼 **동시에** 이 값들을 계산할 방법은 없는 걸까요? 그렇습니다. **벡터**를 활용하면 한번에 계산할 수 있습니다.

앞으로 편미분한 값들을 모아서 벡터로 만든 것을 **기울기**(gradient)라고 부르겠습니다. 기울기는 다음과 같이 Python으로 구현할 수 있습니다.

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx] # save original value
        x[idx] = tmp_val + h # f(x+h)
        fxh1 = f(x)

        x[idx] = tmp_val - h # f(x-h)
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # restore original value
    return grad
```

이렇게 각각의 점들에서의 기울기를 수치 미분으로 계산할 수 있습니다. 그런데 이 기울기가 의미하는 것은 무엇일까요? 예제를 들어서 표현해보겠습니다.

아래의 그림은 다변수 함수 $f(x_0, x_1)=x_0^2+2x_1^2$와 그 함수의 기울기(gradient)에 마이너스를 곱한 벡터를 나타낸 그림입니다.

<figure>
    <img src="/posts/study/machine%20learning/deep%20learning/images/2023-02-08-neural-network-learning_3.jpg"
         title="Multivariate function"
         alt="Image of multivariate function"
         class="img_center"
         style="width: 50%"/>
    <figcaption>함수 $f(x_0,x_1)=x_0^2+2x_1^2$의 그래프</figcaption>
</figure>

<figure>
    <img src="/posts/study/machine%20learning/deep%20learning/images/2023-02-08-neural-network-learning_4.jpg"
         title="Gradient of multivariate function"
         alt="Image of gradient of multivariate function"
         class="img_center"
         style="width: 50%"/>
    <figcaption>함수 $f(x_0,x_1)=x_0^2+2x_1^2$의 gradient</figcaption>
</figure>

[Fig. 2.]를 보면 마치 접시처럼 안쪽으로 움푹 파여있는 형상의 그래프를 확인할 수 있습니다. 그리고 [Fig. 3.]을 보면 기울기들이 **방향**을 가진 벡터(화살표)로 그려집니다. 또한 그 벡터들은 "<ins>가장 낮은 장소</ins>"(**최솟값**)을 가리키는 것처럼 보입니다. 그리고 화살표의 크기도 <ins>가장 낮은 장소</ins>에서 멀어질수록 **커짐**을 알 수 있습니다.

하지만 기울기가 항상 **가장 낮은 장소**(최솟값)를 가리키지는 않습니다. 사실 기울기는 각 지점에서 낮아지는 방향을 가리킵니다. 즉, 기울기가 가리키는 쪽은 **각 장소에서 함수의 출력값을 가장 크게 줄이는 방향**입니다.

---

## 경사법

기계학습의 문제 대부분은 학습 단계에서 최적의 매개변수를 찾아냅니다. 신경망 역시 최적의 매개변수(가중치와 편향)를 학습 시에 찾아야 합니다. 여기서 최적이란 **손실 함수가 최솟값**이 될 떄의 매개변수 값입니다. 하지만 대부분의 손실 함수는 매우 **복잡한** 함수입니다. 따라서 그 매개변수의 공간이 광대하여 어디가 최솟값이 되는 지점인지를 쉽게 짐작할 수 없습니다. 이런 상황에서 <ins>매개변수에 대한 손실 함수의 변화량</ins>(**기울기**)을 잘 이용해서 **손실 함수의 최솟값**(혹은 *가능한 한 작은 값*)을 찾으려는 것이 경사법입니다.

여기에서 주의할 점은 각 지점에서 함수의 값을 낮추는 방안을 제시하는 지표가 **기울기**라는 것입니다. 그러니 기울기가 가리키는 곳에 정말 함수의 최솟값이 있는지, 즉 그쪽이 정말로 나아갈 방향인지는 보장할 수 없습니다.

> 💡 정말 복잡한 함수에서는 기울기가 가리키는 방향에 최솟값이 없는 경우가 대부분입니다.<br>
> ⚠️ 함수가 **극솟값**, **최솟값**, 그리고 **안장점**이 되는 장소에서는 기울기가 0입니다. 극솟값(local minimum point)은 국소적인 최솟값, 즉 한정된 범위에서의 최솟값인 점을 말하고 안장점(saddle point)은 어느 방향에서 보면 극댓값이고 다른 방향에서 보면 극솟값이 되는 점을 말합니다. 경사법은 기울기가 0인 장소를 찾지만 그것이 반드시 최솟값이라고 할 수는 없는 이유입니다.<br>
> 복잡하고 찌그러진 모양의 함수라면 평평한 곳으로 파고들면서 고원(plateau)이라 하는 학습이 진행되지 않는 정체기에 빠질 수 있습니다.

하지만 기울어진 방향이 꼭 최솟값을 가리키는 것은 아니나, 그 방향으로 가면 함수의 값을 **줄일 수는** 있습니다. 그래서 최솟값이 되는 장소를 찾는 문제에서는 **기울기** 정보를 **단서**로 나아갈 방향을 정해야 합니다. **경사법**이란, 함수의 현 위치에서 기울어진 방향으로 일정 거리만큼 이동하고, 이동한 곳에서도 마찬가지로 기울기를 구해서 기울어진 방향으로 나아가기를 반복하여 함수의 값을 점차 줄이는 방법을 말합니다. 경사법은 기계학습을 최적화하는 데 흔히 쓰이는 방법입니다.

> 💡 경사법은 **최솟값**을 찾느냐, **최댓값**을 찾느냐에 따라 경사 **하강**법(gradient **descent** method) 또는 경사 **상승**법(gradient **ascent** method)이라고 부릅니다. 손실 함수의 부호를 바꾸면 두 방법을 동일하게 생각할 수 있습니다. 신경망(딥러닝) 분야에서는 일반적으로 **경사 하강법**을 주로 사용합니다.

입력값 $x$(스칼라)의 함수 $f(x)$에 대한 경사법을 수식으로 나타내면 아래와 같습니다.

$$
x=x-\eta \frac{\partial f(x)}{\partial x} \label{gradient_descent} \tag{4}
$$

식 $(\ref{gradient_descent})$에서 $\eta$는 함수의 입력값 $x$를 갱신하는 양을 나타내고, 이를 신경망 학습에서는 **학습률**(learning rate)이라고 합니다. 한 번의 학습으로 얼마만큼 학습해야 할지, 즉 매개변수 값을 얼마나 갱신하냐를 정하는 것입니다. 학습률 값은 0.01이나 0.001 등 미리 *특정한* 값으로 정해둡니다. 일반적으로 이 값이 너무 ***크거나 작으면*** "**좋은 장소**"를 찾아가지 못합니다. 신경망 학습에서는 보통 이 학습률 값을 조정하면서 올바르게 학습하고 있는지를 확인하면서 진행합니다.

경사 하강법을 Python을 이용해서 구현해보겠습니다.

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x) # partial derivative
        x -= lr * grad # eqn(4)
    return x
```

이 방법을 위의 예제인 $f(x_0,x_1)=x_0^2+2x_1^2$에 적용해보겠습니다. 학습률은 0.1, 갱신 횟수는 40번으로 설정한 그림입니다.

<figure>
    <img src="/posts/study/machine%20learning/deep%20learning/images/2023-02-08-neural-network-learning_5.jpg"
         title="Gradient descent method"
         alt="Image of gradient descent method"
         class="img_center"
         style="width: 40%"/>
    <figcaption>함수 $f(x_0,x_1)=x_0^2+2x_1^2$의 gradient descent 갱신 과정</figcaption>
</figure>

그림에서 등고선을 통해 함수의 기울기를 표현했습니다. 그리고 초기 위치를 총 3곳을 설정하여 **위치와 무관하게** 최솟값에 도달하는 것을 확인할 수 있습니다. 또한, 기울기가 상대적으로 **큰 곳**(가파른 곳)에서는 한 번 갱신할 때 **많이** 움직이지만 기울기가 작은 곳(완만한 곳)에서는 **조금씩** 움직이는 것을 확인할 수 있습니다. 학습률을 너무 크거나 작게 설정하면 갱신이 잘 되지 않고 발산할 가능성이 존재합니다.

> 📑 학습률 같은 매개변수를 **하이퍼파라미터**(hyper parameter)라고 합니다. 이는 가중치나 편향 같은 **신경망의 매개변수**와는 성질이 다른 매개변수입니다. 신경망 매개변수는 훈련 데이터와 학습 알고리즘에 의해서 **자동**으로 획득되는 반면, 하이퍼파라미터는 사람이 **직접** 설정해야 하기 때문입니다. 일반적으로 이 하이퍼파라미터들은 여러 후보 값 중에서 시험을 통해 가장 잘 학습하는 값을 찾는 과정을 거쳐야 합니다.
