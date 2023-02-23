---
layout: post
title: Learning-Techniques-2
category: deep learning
post-series: Deep learning from scratch
post-order: 17
---

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/learning-techniques-1)에서는 **최적화 기법**에 대해서 배웠습니다. 이번 post에서는 신경망 학습의 가중치 매개변수의 **초깃값**을 선정하는 기술에 대해서 알아보겠습니다.

---

# 가중치의 초깃값

신경망 학습에서 특히 중요한 것이 바로 **가중치의 초깃값**입니다. 가중치의 초깃값을 무엇으로 설정하느냐가 종종 신경망 학습의 성패를 가르기도 합니다. 먼저 권장 초깃값에 대해서 설명하고 이에 대한 학습 결과를 알아보겠습니다.

## 권장 초깃값

우선 **오버피팅**을 억제해서 신경망의 범용 성능을 높이는 기술인 **가중치 감소**(weight decay) 기법에 대해서 알아보겠습니다. 가중치 감소는 간단히 말하자면 가중치 매개변수의 값이 *작아지도록* 학습하는 방법입니다. 가중치 값을 작게 하여 오버피팅이 일어나지 않게 하는 것입니다.

가중치를 작게 만들고 싶다면 초깃값도 최대한 **작은 값**에서 시작하면 되지 않을까요? 사실 그래서 지금까지 가중치 초깃값을 표준편차가 **0.01**인 **정규분포**에서 생성되는 값을 사용했었습니다. 그렇다면 가중치의 초깃값을 **0**으로 설정하면 더 좋은 결과가 있지 않을까요? 답부터 얘기하면 0으로 초기화하는 것은 별로 좋은 선택이 **아닙니다**. 실제로 가중치 초깃값을 0으로 하면 학습이 올바로 이루어지지 않습니다.

초깃값을 모두 0으로 설정해서는 안되는 이유(정확히는 가중치를 균일한 값으로 설정해서는 안 됩니다)는 바로 **오차역전파법**에서 모든 가중치의 값이 **똑같이** 갱신되기 때문입니다. 예를 들어 2층 신경망에서 첫 번째와 두 번째 층의 가중치가 모두 **0**이라고 가정하겠습니다. 그럼 순전파 때는 입력층의 가중치가 0이기 때문에 두 번째 층의 뉴런에 모두 **같은 값**이 전파됩니다. 두 번째 층의 모든 뉴런에 같은 값이 입력된다는 것은 역전파 때 두번째 층의 가중치가 모두 **똑같이 갱신**된다는 말이 됩니다(곱셈 노드의 역전파를 떠올려보면 *순전파 때의 입력 신호*를 활용합니다). 그래서 가중치들은 같은 초깃값에서 시작하고 갱신을 거쳐도 <ins>여전히 같은 값</ins>을 유지하는 것입니다. 이는 가중치를 여러 개 갖는 의미를 사라지게 합니다. 이러한 <ins>가중치가 고르게 되어버리는 상황</ins>(정확히는 가중치의 대칭적인 구조를 이루고 있는 상황)을 막으려면 초깃값을 **무작위로** 설정해야 합니다.

### 은닉층의 활성화값 분포

**은닉층의 활성화값**(활성화 함수의 출력 데이터)의 분포를 관찰하면 중요한 정보를 얻을 수 있습니다. 가중치의 초깃값을 바꿔가면서 은닉층의 활성화값들이 어떻게 변화하는지 관찰해보겠습니다. 신경망의 층은 총 5개이며, 각 층의 뉴런은 100개로 설정하겠습니다. 입력 데이터로서 1,000개의 데이터를 정규분포로 무작위로 생성하여 이 5층 신경망에 흘립니다. 활성화 함수로는 **시그모이드** 함수를 사용하겠습니다.

#### 정규분포

입력으로 **표준편차가 1**인 정규분포에서 추출된 난수를 사용했을때의 히스토그램은 아래와 같습니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_11.png"
         title="Histogram of activation value when the weight is initialized to normal distribution with standard deviation 1"
         alt="Image of histogram of activation value when the weight is initialized to normal distribution with standard deviation 1"
         class="img_center"
         style="width: 100%"/>
    <figcaption>가중치가 표준편차가 1인 정규분포를 따르는 난수로 초기화되었을 때 각 층의 활성화값 분포</figcaption>
</figure>

[Fig. 1.]을 보면 각 층의 활성화값들이 0과 1에 치우쳐서 분포되어 있는 것을 확인할 수 있습니다. 활성화 함수로 사용된 **시그모이드** 함수는 출력이 0이나 1에 가까워지면 미분은 거의 0에 가깝습니다. 그래서 데이터가 0과 1에 치우쳐서 분포하게 되면 역전파의 **기울기** 값이 점점 작아지다가 *사라집니다*. 이것이 **기울기 소실**(gradient vanishing)이라 알려진 문제입니다. 층을 깊게 하여 복잡한 함수를 표현하는 딥러닝에서 기울기 손실은 매우 심각한 문제라고 할 수 있겠습니다.

그렇다면 이번에는 가중치의 표준편차를 0.01로 바꿔 같은 신경망에 흘려보겠습니다. 결과는 다음과 같습니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_12.png"
         title="Histogram of activation value when the weight is initialized to normal distribution with standard deviation 0.01"
         alt="Image of histogram of activation value when the weight is initialized to normal distribution with standard deviation 0.01"
         class="img_center"
         style="width: 100%"/>
    <figcaption>가중치가 표준편차가 0.01인 정규분포를 따르는 난수로 초기화되었을 때 각 층의 활성화값 분포</figcaption>
</figure>

[Fig. 2.]를 보면 이번에는 각 층의 활성화값들이 0.5에 치우친 것을 확인할 수 있습니다. 활성화 함수의 그래프를 생각해보면 앞의 예처럼 기울기 소실은 발생하지 않습니다만, 활성화값들이 치우쳤다는 것은 표현력 관점에서는 큰 문제가 있는 것입니다. 즉, 다수의 뉴런이 거의 같은 값을 출력하고 있으니 뉴런을 여러 개 둔 의미가 사라집니다. 그래서 활성화값들이 치우치면 **표현력을 제한**한다는 관점에서 문제가 됩니다.

> 💁 각 층의 활성화값은 따라서 **적당히 고루** 분포되어야 합니다. 층과 층 사이에 적당하게 다양한 데이터가 흐르게 해야 신경망 학습이 효율적으로 이뤄지기 때문입니다. 반대로 치우친 데이터가 흐르면 **기울기 소실**이나 **표현력 제한** 문제에 빠져서 학습이 잘 이뤄지지 않는 경우가 생깁니다.

---

#### Xavier 초깃값

이어서 사비에르 글로로트와 요수아 벤지오의 논문[^fn-xavier-initialization]에서 권장하는 가중치 초깃값인, 일명 **Xavier 초깃값**을 사용해보겠습니다. Xavier 초깃값은 많은 딥러닝 프레임워크에서 표준적으로 사용하고 있습니다. 해당 논문은 각 층의 활성화값들을 **광범위하게** 분포시킬 목적으로 가중치의 적절한 분포를 찾고자 했습니다. 그리고 앞 계층의 노드가 $\boldsymbol{n}$개라면 표준편차가 $\sqrt{\frac{\boldsymbol{1}}{\boldsymbol{n}}}$인 분포를 사용하면 된다는 결론을 이끌었습니다. Xavier 초깃값을 사용하면 앞 층에 노드가 많을수록 대상 노드의 초깃값으로 설정하는 가중치가 **좁게** 퍼집니다. Xavier 초깃값을 사용한 신경망에 대한 활성화값들의 분포는 다음과 같습니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_13.png"
         title="Histogram of activation value when the weight is initialized with Xavier initialization"
         alt="Image of histogram of activation value when the weight is initialized with Xavier initialization"
         class="img_center"
         style="width: 100%"/>
    <figcaption>가중치가 Xavier 초깃값으로 초기화되었을 때 각 층의 활성화값 분포</figcaption>
</figure>

[Fig. 3.]을 보면 층이 깊어지면서 형태가 다소 일그러지지만, 이전의 방식들과 비교해보면 활성화값들이 훨씬 더 **넓게** 분포됨을 알 수 있습니다. 각 층에 흐르는 데이터들이 적당히 넓게 분포되어 있으므로, 시그모이드 함수의 표현력도 제한받지 않고 학습이 효율적으로 이뤄질 것으로 기대됩니다.

> [Fig. 3.]에서 층이 깊어질수록 종 모양의 histogram이 찌그러지는 것을 볼 수 있습니다. 이는 활성화 함수로 시그모이드 함수가 아닌 **쌍곡선** 함수(tanh 함수)를 이용하면 개선됩니다. 

쌍곡선 함수의 공식은 다음과 같습니다.

$$
\begin{matrix}
\tanh{z}&=&\frac{\sinh{z}}{\cosh{z}}, \\
&=&\frac{\exp{z}\ -\ \exp{-z}}{\exp{z}\ +\ \exp{-z}}, \\
&=&\frac{\exp{2z}\ -\ 1}{\exp{2z}\ +\ 1}. \\
\end{matrix} \tag{1}
$$

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_14.png"
          title="Comparison of sigmoid and hyperbolic tangent activation function "
          alt="Image of graph comparing sigmoid and hyperbolic tangent activation function"
          class="img_center"
          style="width: 60%"/>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_15.png"
         title="Comparison of histogram when using activation function with sigmoid and hyperbolic tangent"
         alt="Image of histogram when compared activation function with sigmoid and hyperbolic tangent"
         class="img_center"
         style="width: 100%"/>
    <figcaption>활성화함수를 시그모이드와 쌍곡선 함수를 사용했을 때 활성화 값들의 분포</figcaption>
</figure>

> 쌍곡선 함수도 시그모이드 함수와 같은 'S자 모양' 곡선 함수입니다만, 쌍곡선 함수는 **원점**에서 대칭인 S자 곡선인 반면에 시그모이드 함수는 $(x,y)=(\boldsymbol{0},\boldsymbol{0.5})$에서 대칭인 S자 곡선입니다. 활성화 함수용으로는 **원점에서 대칭**인 함수가 바람직하다고 알려져 있습니다.

#### He 초깃값

Xavier 초깃값은 활성화 함수가 **선형**인 것을 전제로 이끈 결과입니다. 시그모이드와 쌍곡선 함수는 좌우 대칭이라서 중앙 부근이 선형인 함수로 볼 수 있습니다. 그래서 Xavier 초깃값이 적절하게 사용될 수 있습니다. 반면 **ReLU** 함수를 활성화 함수로 사용할 때에는 ReLU에 특화된 초깃값을 이용하라고 권장합니다. 이 특화된 초깃값을 찾아낸 카이밍 히의 이름을 따서 **He 초깃값**[^fn-He-initialization]이라 합니다.

He 초깃값은 앞 계층의 노드가 $\boldsymbol{n}$개일 때, 표준편차가 $\sqrt{\frac{\boldsymbol{2}}{\boldsymbol{n}}}$인 정규분포를 사용합니다. Xavier 초깃값의 표준편차가 $\sqrt{\frac{\boldsymbol{1}}{\boldsymbol{n}}}$이었던 것을 기억해보면 **ReLU**는 **음의 영역이 0**이라서 더 넓게 분포시키기 위해서 **2배의** 계수가 필요하다고 직감적으로 해석할 수 있겠습니다. 활성화 함수를 시그모이드 함수가 아닌 ReLU 함수로 바꿨기에 초깃값들도 각각에 대해서 관찰해보겠습니다. 결과는 다음과 같습니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_16.png"
         title="Histogram of activation values using activation function of ReLU and initial weight of normal distribution of standarad deviation 0.01"
         alt="Image of histogram of activation values using activation function of ReLU and initial weight of normal distribution of standarad deviation 0.01"
         class="img_center"
         style="width: 100%"/>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_17.png"
         title="Histogram of activation values using activation function of ReLU and initial weight of Xavier initialization"
         alt="Image of histogram of activation values using activation function of ReLU and initial weight of Xavier initialization"
         class="img_center"
         style="width: 100%"/>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_18.png"
         title="Histogram of activation values using activation function of ReLU and initial weight of He initialization"
         alt="Image of histogram of activation values using activation function of ReLU and initial weight of He initialization"
         class="img_center"
         style="width: 100%"/>
    <figcaption>ReLU 활성화 함수와 표준편차가 0.01인 정규분포, Xavier 초깃값, He 초깃값을 따르는 활성화값들의 분포</figcaption>
</figure>

[Fig. 5.]를 보면 첫번째 histogram의 각 층의 활성화값들은 아주 작은 값들입니다. 신경망에 **아주 작은** 데이터들이 흐른다는 것은 역전파 때 **가중치의 기울기** 역시 작아진다는 것입니다. 그래서 거의 학습이 이뤄지지 않을 것입니다. 두번째 histogram을 보면 층이 깊어지면서 치우침이 조금씩 커집니다. 그렇다면 **기울기 소실**이 발생할 가능성도 역시 커지겠죠. 마지막 histogram을 보면 모든 층에서 균일하게 분포하는 것을 확인할 수 있습니다. 층이 깊어져도 분포가 균일하게 유지되기에 역전파 때도 적절한 값이 나올 수 있을것으로 예상할 수 있습니다.

결론을 내보면, 활성화 함수로 **ReLU** 함수를 사용할 때는 **He 초깃값**을, **시그모이드나 쌍곡선** 함수 등의 <ins>S자 모양 곡선</ins>일 때는 **Xavier 초깃값**을 쓰는 것이 권장 초깃값이라고 말할 수 있겠습니다. MNIST 데이터셋을 통해서 각 초깃값이 신경망 학습에 주는 영향에 대해서도 알아보았습니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_19.png"
         title="Comparison of each initialization value using MNIST dataset"
         alt="Image of graph comparing each initialization value using MNIST dataset"
         class="img_center"
         style="display: inline-block; width: 50%"/>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_20.png"
         title="Accuracy of each initialization value using MNIST datset"
         alt="Image of accuracy of each initialization value using MNIST dataset"
         class="img_center"
         style="display: inline-block; width: 80%"/>
    <figcaption>각 초깃값의 MNIST 데이터셋에 대한 학습 진도와 정확도 비교</figcaption>
</figure>

[Fig. 6.]에서는 층별 뉴런 수가 100개인 5층 신경망에서 활성화 함수로 **ReLU** 함수를 사용하고 최적화 기법으로 **SGD**를 사용했습니다. <span style="color: red">*표준편차가 0.01인 정규분포*</span>로 초기화한 신경망은 거의 학습이 이뤄지지 않습니다. 순전파에서 너무 **작은 값**(0 근처로 밀접한 데이터)이 흐르기 때문입니다. 그로 인해 역전파 때의 기울기도 작아져 가중치가 거의 갱신되지 않는 것입니다. 반면 <span style="color: green">*Xavier 초깃값*</span>(sigmoid)과 <span style="color: blue">*He 초깃값*</span>(relu)의 경우는 학습이 순조롭게 잘 이뤄지고 있습니다. 학습 진도는 He 초깃값이 좀 더 빠른 모습을 볼 수 있습니다.

지금까지 살펴보았듯이 **가중치의 초깃값**은 신경망 학습에 아주 중요한 포인트입니다. 가중치의 초깃값에 따라 신경망 학습의 성패가 갈리는 경우가 많습니다. 초깃값의 중요성은 간과하기 쉬운 부분이지만, 어떤 일이든 시작이 중요한 법입니다. 🚩

---

# 배치 정규화

이전에는 **활성화값 분포**를 집중관찰하면서, 각 층의 가중치의 초깃값을 적절히 설정하면 각 층의 활성화값 분포가 **적당히** 퍼지면서 학습이 원활하게 수행됨을 배웠습니다. 그렇다면 각 층이 활성화를 적당히 퍼뜨리도록 '**강제**'해보면 어떨까요? **배치 정규화**(Batch Normalization)[^fn-batch-normalization]가 그런 아이디어에서 출발한 방법입니다. 배치 정규화는 2015년에 제안된 방법입니다. 비교적 최근에 나온 기법임에도 불구하고 많은 연구자와 기술자들이 즐겨 사용하고 있습니다.

배치 정규화가 주목받는 이유는 다음과 같습니다.
* 학습을 빨리 진행할 수 있다(학습 속도 개선).
* 초깃값에 크게 의존하지 않는다(초깃값 의존성 감소).
* 오버피팅을 억제한다(추가적인 기법의 필요성 감소).

딥러닝의 학습 시간이 길다는 것을 생각하면 첫 번째 이점은 매우 반가운 일입니다. 초깃값에 크게 신경 쓸 필요가 없고, 오버피팅 억제 효과가 있다는 점도 딥러닝의 두통거리를 덜어줍니다. 🤸‍♂️

배치 정규화의 기본 아이디어는 앞에서 말했듯이 각 층에서의 활성화값이 **적당히** 분포되도록 조정하는 것입니다. 그래서 아래 그림처럼 데이터 분포를 정규화하는 **배치 정규화 계층**(Batch Norm Layer)을 신경망에 삽입합니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_21.png"
         title="Example of neural network using batch normalization"
         alt="Image of example of neural network using batch normalization"
         class="img_center"
         style="width: 60%"/>
    <figcaption>배치 정규화를 사용한 신경망의 예</figcaption>
</figure>

배치 정규화는 그 이름과 같이 학습 시 **미니배치**를 단위로 정규화합니다. 구체적으로는 데이터 분포가 **평균이 0, 분산이 1**이 되도록 정규화합니다. 수식으로는 다음과 같습니다.

$$
\begin{matrix}
\mu_B &\leftarrow& \frac{1}{m} \sum_{i=1}^m x_i, \\
\sigma_B^2 &\leftarrow& \frac{1}{m} \sum_{i=1}^m (x_i-\mu_B)^2, \\
\hat{x_i} &\leftarrow& \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}}. \label{batch_norm} \tag{2}
\end{matrix}
$$

식 $(\ref{batch_norm})$에서는 미니배치 $B=\begin{Bmatrix} x_1,&x_2,&\cdots,&x_m\end{Bmatrix}$이라는 $m$개의 입력 데이터의 집합에 대해 평균 $\mu_B$와 분산 $\sigma_B^2$을 구합니다. 그리고 입력 데이터를 평균이 0, 분산이 1이 되게(적절한 분포가 되게) 정규화합니다. 또한, 식 $(\ref{batch_norm})$에서 $\varepsilon$은 작은 값(예컨대 10e-7 등)으로, 0으로 나누는 사태를 예방하는 역할입니다.

이러한 정규화 처리를 **활성화 함수의 앞**(혹은 **뒤**)에 삽입함으로써 데이터 분포가 덜 치우치게 할 수 있습니다. 또, 배치 정규화 계층마다 이 정규화된 데이터에 고유한 **확대**(scale)와 **이동**(shift) 변환을 수행합니다. 수식으로는 다음과 같습니다.

$$
y_i \leftarrow \gamma \hat{x_i}+\beta. \label{scale_shift} \tag{3}
$$

식 $(\ref{scale_shift})$에서 $\gamma$가 확대를, $\beta$가 이동을 담당합니다. 두 값은 처음에는 $\gamma=1,\ \beta=0$부터 시작하고, 학습하면서 적합한 값으로 조정해갑니다.

> 💡 $\gamma=1$은 1배 확대를 뜻하고, $\beta=0$은 이동하지 않음을 뜻합니다. 즉, 처음에는 원본 그대로에서 시작한다는 이야기입니다.

이상이 배치 정규화의 알고리즘입니다. 이 알고리즘이 신경망에서 **순전파** 때 적용되죠. 순전파에 적용된다는 말은 **역전파**에서도 적용되어야 하겠죠? 그렇다면 계산 그래프를 통해서 나타내 보겠습니다.

## 계산 그래프

결과적인 그래프부터 보여드리면 다음과 같습니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_22.png"
         title="Computational graph of batch normalization layer"
         alt="Image of computational graph of batch normalization layer"
         class="img_center"
         style="width: 75%"/>
    <figcaption>배치 정규화의 계산 그래프</figcaption>
</figure>

[Fig. 8.]에서는 설명의 간략화와 실제적인 구현을 위해서 수식과는 표현이 조금 다르게 묘사되어 있을 수 있습니다. 순전파에 대해서 설명한 다음에 역전파에 대해서 차례대로 설명하겠습니다.

### 순전파

먼저 배치 정규화의 순전파입니다. 미니배치 $B=\begin{Bmatrix} x_1,&x_2,&\cdots,&x_m\end{Bmatrix}$는 $m$개의 $x$가 포함되어 있고, $x$에는 $n$개의 데이터가 포함되어 있다고 가정하겠습니다. 그렇다면 입력 데이터 $X$의 크기는 $(m,n)$이라고 할 수 있겠습니다.

식 $(\ref{batch_norm})$와 $(\ref{scale_shift})$을 통해 [Fig. 8.]의 순전파를 이해하는 데에는 큰 어려움이 없을 것입니다. 다만 계산 과정에서 유의할 점은 $\gamma$와 $\beta$가 **scalar**가 아닌 **vector**라는 사실과 미니배치 $B$의 데이터 $x$의 평균을 열별로 구한다는 사실입니다. 또한 순전파의 노드 중에서 지금까지 없었던 노드는 제곱근 노드와 평균 노드, 뺄셈 노드 입니다. 하나씩 차례대로 역전파를 계산해보겠습니다.

### 역전파

#### 제곱근 노드

먼저 제곱근 노드의 역전파입니다. 식 $f(x)=\sqrt{x}$에 대한 미분은 다음과 같습니다.

$$
y=f(x)=\sqrt{x}, \\
\frac{df(x)}{dx}=\frac{1}{2\sqrt{x}}=\frac{1}{2y}. \label{diff_sqrt} \tag{4}
$$

식 $(\ref{diff_sqrt})$를 보시면 제곱근 함수를 미분하면 $\frac{1}{2}$과 순전파 때의 출력이 뒤집혀서 곱해진 것을 확인할 수 있습니다.

#### 평균 노드

다음은 평균 노드입니다. 크기가 $(m,n)$인 $\boldsymbol{X}$에 대한 평균 $\boldsymbol{y}\in\mathbb{R}^{1\times n}$을 산출하는 식과 그에 대한 미분은 다음과 같습니다.

$$
\boldsymbol{y}=\begin{bmatrix}
     y_1 & y_2 & \cdots & y_n
\end{bmatrix}=f(\boldsymbol{X}),\\
y_j=\frac{1}{m} \sum_{i=1}^m x_{ij}=\frac{x_{1j}+x_{2j}+\cdots+x_{mj}}{m}, \\
\text{where}\ \boldsymbol{X}=\begin{bmatrix}
     x_{11} & x_{12} & \cdots & x_{1n} \\
     \vdots & \vdots & \ddots & \vdots \\
     x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}, \\
\begin{matrix}
     \frac{\partial{L}}{\partial{\boldsymbol{X}}}&=&
\begin{bmatrix}
     \frac{\partial{L}}{\partial{x_{11}}}&\cdots&\frac{\partial{L}}{\partial{x_{m1}}}\\
     \vdots&\ddots&\vdots\\
     \frac{\partial{L}}{\partial{x_{1n}}}&\cdots&\frac{\partial{L}}{\partial{x_{mn}}}
\end{bmatrix}, \\
\frac{\partial{L}}{\partial{x_{ij}}}
&=&\frac{\partial{\boldsymbol{y}}}{\partial{x_{ij}}}\frac{\partial{L}}{\partial{\boldsymbol{y}}} \\
&=&
\sum_{k=1}^n \frac{\partial{y_k}}{\partial{x_{ij}}}\frac{\partial{L}}{\partial{y_k}}, \\
\frac{\partial{y_k}}{\partial{x_{ij}}}&=&
\begin{cases}
     \frac{1}{m}\ (j=k)\\
     0\ (j \neq k)
\end{cases}\ (k=1,2,\cdots,n), \\
\frac{\partial{L}}{\partial{x_{ij}}}&=&
\sum_{k=1}^n \frac{\partial{y_k}}{\partial{x_{ij}}}\frac{\partial{L}}{\partial{y_k}} \\
&=&\sum_{k=1}^n \frac{1}{m}\frac{\partial{L}}{\partial{y_k}}\ (\frac{\partial{y_k}}{\partial{x_{ij}}}=\frac{1}{m}\ \text{only}\ k=j) \\
&=&\frac{1}{m}\frac{\partial{L}}{\partial{y_j}}, \\
\frac{\partial{L}}{\partial{\boldsymbol{X}}}&=&
\begin{bmatrix}
     \frac{1}{m}\frac{\partial{L}}{\partial{y_1}}&\cdots&\frac{1}{m}\frac{\partial{L}}{\partial{y_1}} \\
     \vdots&\ddots&\vdots\\
     \frac{1}{m}\frac{\partial{L}}{\partial{y_n}}&\cdots&\frac{1}{m}\frac{\partial{L}}{\partial{y_n}}
\end{bmatrix}&=&
\frac{\partial{L}}{\partial{\boldsymbol{y}}}
\frac{1}{m}
\begin{bmatrix}
     1&\cdots&1 
\end{bmatrix} \\
&=&\frac{\partial{L}}{\partial{\boldsymbol{y}}}\frac{1}{m}\boldsymbol{o} 
(\boldsymbol{o}=\begin{bmatrix}
     1 & \cdots & 1
\end{bmatrix}\in\mathbb{R}^{1\times m}).
\end{matrix} \\
\therefore \frac{\partial{L}}{\partial{\boldsymbol{X}}}=\frac{\partial{L}}{\partial{\boldsymbol{y}}}\frac{\partial{\boldsymbol{y}}}{\partial{\boldsymbol{X}}}
=\frac{\partial{L}}{\partial{\boldsymbol{y}}}\frac{1}{m}\boldsymbol{o}. \\
\therefore \frac{\partial{\boldsymbol{y}}}{\partial{\boldsymbol{X}}}=\frac{1}{m}\boldsymbol{o}. \label{mean_gate} \tag{5}
$$

이를 계산 그래프로 나타내면 아래와 같습니다.

<figure>
    <img src="/posts/study/machine learning/deep learning/images/learning_techniques_23.png"
         title="Computational graph of mean gate"
         alt="Image of computational graph of mean gate"
         class="img_center"
/>
    <figcaption>평균 노드의 계산 그래프</figcaption>
</figure>

[Fig. 9.]에서의 역전파 결과와 식 $(\ref{mean_gate})$에서 유도한 결과가 다른 것을 확인할 수 있습니다. 이는 Python의 numpy 모듈을 통해 계산하는 과정에서 broadcast 기능을 이용해서 계산하다 보니 생긴 문제입니다. 따라서 수식적 유도 과정과 실제 구현에서 발생할 수 있는 표기의 차이라고 이해해주시면 감사하겠습니다.

#### 뺄셈 노드

마지막으로 뺼셈 노드의 역전파입니다. 식 $f(x,y)=x-y$에 대한 미분은 다음과 같습니다.

$$
y=f(x,y)=x-y, \\
\frac{\partial{f(x,y)}}{\partial{x}}=1,\ \frac{\partial{f(x,y)}}{\partial{y}}=-1. \label{diff_subtraction} \tag{6}
$$

식 $(\ref{diff_subtraction})$을 보시면 덧셈 노드처럼 역전파때의 값에 1을 곱해서 그대로 흘리기에, 크기는 **그대로**라는 점은 유사하지만, 순전파때 뺄셈 노드에 의해서 **뺄셈 연산이 수행된** 입력값에 대해서는 역전파때에도 '**-**'를 곱하는 것을 확인할 수 있습니다.

배치 정규화 계층을 Python을 이용해서 구현해보겠습니다. 코드가 길어지니 자세한 코드는 여기[^fn-batch-normalization-python]를 참고해 주시면 감사하겠습니다.
```python
class BatchNormLayer:
     def __init__(self, gamma, beta, momentum=0.9, rolling_mean=None, rolling_var=None):
          self.gamma = gamma
          self.beta = beta
          self.momentum = momentum
          self.input_shape = None # 4-D for convolution layer, 2-D for affine layer

          # used when the network is run for testing, not learning
          self.rolling_mean = rolling_mean
          self.rolling_var = rolling_var

          # used when backpropagation
          self.batch_size = None
          self.xd = None # deviation between data and mean
          self.std = None # standarad deviation
          self.xhat = None # normalized data
          self.dgamma = None
          self.dbeta = None

     def forward(self, x, train_flag=True):
          self.input_shape = x.shape
          if x.ndim != 2:
               N, C, H, W = x.shape # batch, channel, height, width
               x = x.reshape(N, -1)
          
          out = self.__forward(x,train_flag)
          return out.reshape(*self.input_shape)

     def __forward(self, x, train_flag):
          if self.rolling_mean == None:
               N, D = x.shape
               self.rolling_mean = np.zeros(D)
               self.rolling_var = np.zeros(D)

          if train_flag:
               mu = np.mean(x, axis=0)
               xd = x - mu
               var = np.mean(xd**2, axis=0)
               std = np.sqrt(var + 10e-7)
               xhat = xd / std

               self.batch_size = x.shape[0]
               self.xd = xd
               self.std = std
               self.xhat = xhat

               # Exponential Moving Average (EMA)
               self.rolling_mean = self.momentum * self.rolling_mean (1 - self.momentum) * mu
               self.rolling_var = self.momentum * self.rolling_var (1 - self.momentum) * var
          else:
               xd = x - self.rolling_mean
               xhat = xd / (np.sqrt(self.running_var + 10e-7))
          
          out = self.gamma * xhat + self.beta
          return out

     def backward(self, dout):
          if dout.ndim != 2:
               N, C, H, W = dout.shape # batch, channel, height, width
               dout = dout.reshape(N, -1)
          
          dx = self.__backward(dout)
          dx = dx.reshape(*self.input_shape)
          return dx

     def __backward(self, dout):
          dbeta = np.sum(dout, axis=0)
          dgamma = np.sum(dout * self.xhat, axis=0)
          dxhat = self.gamma * dout
          dxd = dxhat / self.std
          dstd = -np.sum((dxhat * self.xd) / (self.std**2), axis=0)
          dvar = 0.5 * dstd / self.std
          dxd += (2.0 / self.batch_size) * self.xd * dvar
          dmu = np.sum(dxd, axis=0)
          dx = dxd - dmu / self.batch_size

          self.dgamma = dgamma
          self.dbeta = dbeta
          return dx
```
---

[^fn-xavier-initialization]: 📚 Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.

[^fn-He-initialization]: :books: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE international conference on computer vision. 2015.

[^fn-batch-normalization]: :books: Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." International conference on machine learning. pmlr, 2015.

[^fn-batch-normalization-python]: [여기]()에서 batch normalization에 관한 Python 코드를 확인하실 수 있습니다.