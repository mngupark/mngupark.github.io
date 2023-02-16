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

## 확률적 경사 하강법 (SGD)

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
f(x,y)=\frac{1}{30}x^2+y^2 \label{sgd_example} \tag{2}
$$

식 $(\ref{sgd_example})$를 3차원 그래프를 통해 나타내면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/learning_techniques_1.png"
          title="3-dimensional plot of f(x,y)"
          alt="Image of 3-dimensional plot of f(x,y)"
          class="img_center"
          style="width: 60%"/>
     <figcaption>$f(x,y)=\frac{1}{30}x^2+y^2$의 그래프</figcaption>
</figure>

식 $(\ref{sgd_example})$의 기울기를 수식으로 나타내면 아래와 같습니다.

$$
\frac{\partial{f(x,y)}}{\partial{x}}=\frac{1}{15}x,\ 
\frac{\partial{f(x,y)}}{\partial{y}}=2y \label{diff_sgd_example} \tag{3}
$$

식 $(\ref{diff_sgd_example})$을 그림으로 나타내면 아래와 같습니다. 이 기울기는 y축 방향은 크고 x축 방향은 작다는 것이 특징입니다. 말하자면 y축 방향은 가파른데 x축 방향은 완만한 것입니다. 또한, 식 $(\ref{diff_sgd_example})$이 최솟값이 되는 장소는 $(x,y)=(0,0)$이지만, 아래의 그림에서의 기울기 대부분은 $(0,0)$ 방향을 직접적으로 가리키지 않는다는 것입니다. 그림에 포함된 등곡선을 보면 어떤 의미인지 잘 이해가 되실겁니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/learning_techniques_2.png"
          title="Plot of contour and gradient of the f(x,y)"
          alt="Image of plot of contour gradient of the f(x,y)"
          class="img_center"
          style="width: 60%"/>
     <figcaption>$f(x,y)=\frac{1}{30}x^2+y^2$의 등곡선과 기울기</figcaption>
</figure>

그러면 함수 $f(x,y)$의 최솟값 탐색을 위해서 SGD를 적용해보겠습니다. 탐색을 시작하는 장소(초깃값)은 $(x,y)=(-7.5, -5.0)$으로 하겠습니다. 결과는 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/learning_techniques_3.png"
          title="Path of SGD"
          alt="Image of path of SGD"
          class="img_center"
          style="width: 60%"/>
     <figcaption>SGD의 최적화 갱신 경로</figcaption>
</figure>

[Fig. 3.]에서 SGD의 최적화 갱신을 진행할 때 학습률은 1, 반복 횟수는 20으로 진행했습니다. SGD는 [Fig. 3.]과 같이 심하게 굽어진 움직임을 보여줍니다. 상당히 비효율적으로 움직임입니다. SGD의 단점은 **비등방성**(anisotropy) 함수(방향에 따라 성질, 즉 여기에서는 기울기가 달라지는 함수)에서는 탐색 경로가 비효율적이라는 것입니다. 이럴 때는 SGD 같이 무작정 기울어진 방향으로 진행하는 단순한 방식보다 더 영리한 묘안이 간절해집니다. 또한, SGD가 지그재그로 탐색하는 근본적인 원인은 기울어진 방향이 본래의 최솟값과 다른 방향을 가리켜서라는 점도 생각해볼 필요가 있습니다.

> 🔖 비등방성에 대한 자세한 설명은 [여기](https://en.wikipedia.org/wiki/Anisotropy)를 참고해주세요.

## 모멘텀 (Momentum)

**모멘텀**은 운동량을 뜻하는 단어로, 물리학에서 주로 사용됩니다. 모멘텀 기법은 수식으로는 다음과 같이 쓸 수 있습니다.

$$
\boldsymbol{v} \leftarrow \alpha \boldsymbol{v} - \eta \frac{\partial{L}}{\partial{\boldsymbol{W}}} \\
\boldsymbol{W} \leftarrow \boldsymbol{W} + \boldsymbol{v} \label{momentum} \tag{4}
$$

식 $(\ref{momentum})$는 SGD에서의 수식과 같이 $\boldsymbol{W}$는 갱신할 가중치 매개변수, $\frac{\partial{L}}{\partial{\boldsymbol{W}}}$은 $\boldsymbol{W}$에 대한 손실 함수의 기울기, $\eta$는 학습률입니다. $\boldsymbol{v}$라는 새로운 변수가 나오는데, 이는 물리학에서 말하는 **속도**(velocity)입니다. 식 $(\ref{momentum})$는 **기울기 방향**으로 힘을 받아 물체가 **가속**된다는 물리 법칙을 나타냅니다.

그리고 식 $(\ref{momentum})$의 $\alpha \boldsymbol{v}$항은 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할을 합니다($\alpha=0.9$ 등의 값으로 초기화합니다). 물리학에서의 지면 마찰이나 공기 저항에 해당하는 겁니다. 이를 Python으로 구현한 코드는 다음과 같습니다.

```python
class Momentum:
     def __init__(self, lr=0.01, momentum=0.9):
          self.lr = lr
          self.momentum = momentum
          self.v = None

     def update(self, params, grads):
          if self.v is None:
               self.v = {}
               for key, val in params.items():
                    self.v[key] = np.zeros_like(val)
          
          for key in params.key():
               self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
               params[key] += self.v[key]
```

이번에는 함수 $f(x,y)$의 최솟값 탐색을 위해서 Momentum를 적용해보겠습니다. 탐색을 시작하는 장소(초깃값)은 SGD와 똑같이 $(x,y)=(-7.5, -5.0)$으로 하겠습니다. 결과는 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/learning_techniques_4.png"
          title="Path of Momentum"
          alt="Image of path of Momentum"
          class="img_center"
          style="width: 60%"/>
     <figcaption>Momentum의 최적화 갱신 경로</figcaption>
</figure>

[Fig. 4.]에서 SGD와 Momentum 최적화 갱신을 진행할 때 학습률은 1, 반복 횟수는 20으로 진행했습니다. Momentum의 $\alpha$는 0.9로 설정했습니다. [Fig. 4.]에서 보듯 Momentum의 갱신 경로는 마치 공이 그릇 바닥을 **구르듯** 움직입니다. SGD와 비교하면 '**지그재그의 정도**'가 덜한 것을 알 수 있습니다. 이는 $x$축의 힘은 상대적으로 작고 중심점을 향하고 있기 때문에 $y$축에 비하면 안정적이게 가속하는 것을 알 수 있습니다. 반면에 $y$축의 힘은 상대적으로 크고 방향이 위아래로 계속 바뀌기에 속도가 안정적이지 않습니다. 전체적으로 보면 SGD보다 Momentum이 최적값에 더 근접하고 $x$축 방향으로 빠르게 다가가 지그재그 움직임이 줄어드는 것을 확인할 수 있습니다.

## AdaGrad

신경망 학습에서는 학습률$(\eta)$값이 중요합니다. 이 값이 너무 작으면 학습시간이 너무 길어지고, 반대로 크면 발산하여 학습이 제대로 이뤄지지 않습니다. 이 학습률을 정하는 효과적 기술로 **학습률 감소**(learning rate decay)가 있습니다. 이는 <ins>학습을 진행하면서 학습률을 점차 줄여가는 방법</ins>입니다. 처음에는 크게 학습하다가 조금씩 작게 학습한다는 얘기로, 실제 신경망 학습에 자주 쓰입니다.

학습률을 서서히 낮추는 가장 간단한 방법은 매개변수 '**전체**'의 학습률 값을 *일괄적으로* 낮추는 것이겠죠. 이를 더욱 발전시킨 것이 **AdaGrad**입니다. AdaGrad는 '**각각의**' 매개변수에 **맞춤형** 값을 만들어줍니다.

> :books: AdaGrad에 관한 자세한 내용은 <cite>Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7).</cite>를 참고해 주시기 바랍니다.

