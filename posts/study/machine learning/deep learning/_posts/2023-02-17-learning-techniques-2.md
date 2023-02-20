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
\tanh{z}&=&\frac{\sinh{z}}{\cosh{z}} \\
&=&\frac{\exp{z}\ -\ \exp{-z}}{\exp{z}\ +\ \exp{-z}} \\
&=&\frac{\exp{2z}\ -\ 1}{\exp{2z}\ +\ 1} \\
\end{matrix}
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

[^fn-xavier-initialization]: 📚 Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.

[^fn-He-initialization]: :books: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE international conference on computer vision. 2015.