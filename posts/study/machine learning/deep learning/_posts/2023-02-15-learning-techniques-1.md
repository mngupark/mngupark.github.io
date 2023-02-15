---
layout: post
title: Learning-techniques-1
category: deep learning
post-order: 16
---

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/backpropation-5)까지는 오차역전파법에 대해서 배웠습니다. 이번 post부터는 신경망 학습의 **핵심 개념**들에 대해서 알아보겠습니다. 신경망 학습에서 다루는 중요한 주제에는 가중치 매개변수의 최적값을 탐색하는 **최적화 방법**, **가중치 매개변수 초깃값**, **하이퍼파라미터** 설정 방법 등이 있습니다. 또한 오버피팅의 대응책인 가중치 감소와 드롭아웃 등의 **정규화 방법**과 **배치 정규화**에 대해서도 알아보겠습니다.

---

# 매개변수 갱신

신경망 학습의 목적은 **손실 함수**의 값을 가능한 한 **낮추는** 매개변수를 찾는 것이었습니다. 이는 곧 매개변수의 **최적값**을 찾는 문제이며 이러한 문제를 푸는 것을 **최적화**(optimization)라고 합니다. 하지만 신경망 최적화는 굉장히 어려운 문제입니다. 매개변수 공간이 매우 넓고 복잡해서 최적의 솔루션은 쉽게 찾을 수 없기 때문입니다.

## 확률적 경사 하강법(SGD)

지금까지 최적의 매개변수 값을 찾는 단서로 매개변수의 **기울기**(미분)를 이용했습니다. 매개변수의 기울기를 구해서 **기울어진 방향**으로 매개변수 값을 **갱신**하는 일을 몇 번이고 반복해서 점점 최적의 값에 다가갔습니다. 이것이 **확률적 경사 하강법**(Stochastic Gradient Descent, SGD)입니다. SGD는 단순하지만 매개변수 공간을 **무작정** 찾는 것보다는 '똑똑한' 방법입니다. 하지만 문제에 따라서는 SGD보다 똑똑한 방법도 있습니다. 지금부터 SGD의 단점을 알아본 후 SGD와는 다른 최적화 기법에 대해서 알아보겠습니다.

SGD는 수식으로는 다음과 같이 쓸 수 있습니다.

$$
\boldsymbol{W} \leftarrow \boldsymbol{W} - \eta \frac{\partial{L}}{\partial{\boldsymbol{W}}} \label{sgd} \tag{1}
$$

식 $(\ref{sgd})$에서 $\boldsymbol{W}$는 갱신할 가중치 매개변수 행렬이고 $\frac{\partial{L}}{\partial{\boldsymbol{W}}}$은 $\boldsymbol{W}$에 대한 손실 함수의 기울기 행렬입니다. $\eta$는 학습률을 의미하는데, 실제로는 $0.01$이나 $0.001$과 같은 값을 미리 정해서 사용합니다(하이퍼파라미터). $\leftarrow$는 우변의 값으로 좌변의 값을 갱신한다는 뜻입니다. 식 $(\ref{sgd})$에서 볼 수 있듯이 SGD는 **기울어진 방향**으로 **일정 거리**만 가겠다는 단순한 방법입니다. 이를 Python으로 구현한 코드는 다음과 같습니다.

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.key():
            params[key] -= self.lr * grads[key]
```

초기화 때 받는 인수인 lr은 **학습률**(learning rate)을 뜻합니다. update(params, grads) method는 SGD 과정에서 반복해서 불립니다. 이때 인수인 params와 grads는 딕셔너리 변수임을 가정하고 있습니다. 이렇게 최적화를 담당하는 클래스를 분리해서 구현하면 기능을 **모듈화**하기 좋습니다. 다른 최적화 기법에도 역시 **update(params, grad)**라는 공통의 메서드를 갖도록 구현해서 실제 학습할 때 optimizer=SGD()가 아닌 optimizer=$(또다른 최적화 기법)으로만 변경하면 되는 것입니다.

### SGD의 단점

SGD는 단순하고 구현도 쉽지만, ***문제에 따라서는*** 비효율적일 때가 있습니다. 한 함수를 통해 예시를 들어보겠습니다.

$$
f(x,y)=\frac{1}{30}x^2+\frac{1}{3}y^2 \{sgd_example} \tag{2}
$$

식 $(\ref{sgd_example})$을 3차원 그래프를 통해 나타내면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/learning_techniques_1.jpg"
          title="3-dimensional plot of f(x,y)"
          alt="Image of 3-dimensional plot of f(x,y)"
          class="img_center"
          style="width: 75%"/>
     <figcaption>f(x,y)=\frac{1}{30}x^2+\frac{1}{3}y^2의 그래프</figcaption>
</figure>

식 $(\ref{sgd_example})$의 기울기를 수식으로 나타내면 아래와 같습니다.

$$
\frac{\partial{f(x,y)}}{\partial{x}}=\frac{1}{15}x,\ 
\frac{\partial{f(x,y)}}{\partial{y}}=\frac{2}{3}y \{diff_sgd_example} \tag{3}
$$

식 $(\ref{diff_sgd_example})$을 그림으로 나타내면 아래와 같습니다. 이 기울기는 y축 방향은 크고 x축 방향은 작다는 것이 특징입니다. 말하자면 y축 방향은 가파른데 x축 방향은 완만한 것입니다. 또한, 식 $(\ref{diff_sgd_example})$이 최솟값이 되는 장소는 $(x,y)=(0,0)$이지만, 아래의 그림에서의 기울기 대부분은 $(0,0)$ 방향을 직접적으로 가리키지 않는다는 것입니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/learning_techniques_2.jpg"
          title="Plot of gradient of the f(x,y)"
          alt="Image of plot of gradient of the f(x,y)"
          class="img_center"
          style="width: 75%"/>
     <figcaption>f(x,y)=\frac{1}{30}x^2+\frac{1}{3}y^2의 기울기</figcaption>
</figure>