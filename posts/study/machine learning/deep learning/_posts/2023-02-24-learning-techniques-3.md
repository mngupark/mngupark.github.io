---
layout: post
title: Learning-Techniques-3
category: deep learning
post-series: Deep learning from scratch
post-order: 18
---

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/learning-techniques-2)에서는 **가중치 초깃값**과 **배치 정규화**에 대해서 배웠습니다. 이번 post에서는 신경망 학습의 **오버피팅**을 억제하는 기술에 대해서 알아보겠습니다.

---

# 오버피팅

기계학습에서는 **오버피팅**이 문제가 되는 일이 많습니다. 오버피팅이란 신경망이 **훈련 데이터**에만 지나치게 적응되어 그 외의 데이터에는 제대로 대응하지 못하는 상태를 말합니다. 기계학습은 범용적인 성능을 지향합니다. 훈련 데이터에는 포함되지 않는, **아직 보지 못한 데이터**가 주어져도 바르게 식별해내는 모델이 바람직합니다. 복잡하고 표현력이 높은 모델을 만들 수는 있지만, 그만큼 오버피팅을 억제하는 기술이 중요해지는 것입니다.

오버피팅은 주로 다음의 두 경우에 일어납니다.
* 매개변수가 **많고** 표현력이 **높은** 모델
* 훈련 데이터가 **적은** 경우

위 두 조건을 충족하면 어떤 일이 발생할까요? 실제로 두 조건을 충족하여 오버피팅을 **일부러** 일으켜보겠습니다. 원래 60,000개의 훈련 데이터로 이루어진 MNIST 데이터셋에서 **500개**만 사용해서 훈련 데이터를 줄이고, **7층** 네트워크를 사용해서 네트워크의 복잡성을 높이겠습니다. 각 층의 뉴런은 100개, 활성화 함수는 ReLU를 사용하겠습니다. 결과는 다음과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/learning_techniques_28.png"
         title="Comparison of accuracy between train and test dataset when overfitting occurred"
         alt="Image of comparison of accuracy between train and test dataset when overfitting occurred"
         class="img_center"
         style="width: 60%"/>
    <figcaption>오버피팅이 발생하는 조건을 충족했을때 훈련 데이터와 시험 데이터의 정확도 비교</figcaption>
</figure>

[Fig. 1.]을 보시면 훈련 데이터의 경우 약 75 데폭을 지나는 무렵부터 거의 100%의 정확도를 기록했습니다. 반면에 시험 데이터의 경우에는 시간이 흘러도 80% 이상의 정확도는 넘어서지 못하는 모습을 보입니다. 이처럼 **훈련 데이터**에만 적응해버리는 것이 바로 **오버피팅**입니다.

## 가중치 감소

오버피팅을 억제하기 위한 방법으로 예로부터 많이 이용해온 **가중치 감소**(weight decay)라는 것이 있습니다. 이는 학습 과정에서 **큰** 가중치에 대해서는 그에 상응하는 **큰 페널티**를 부과하여 오버피팅을 억제하는 방법입니다. 원래 오버피팅은 가중치 매개변수의 *값이 커서* 발생하는 경우가 많기 때문입니다.

신경망 학습의 목적은 <ins>손실 함수의 값을 줄이는 것</ins>입니다. 이때, 예를 들어 가중치의 **제곱 노름**(L2 노름, L2 norm)을 손실 함수에 더합니다. 그렇다면 가중치가 커지는 것을 억제할 수 있겠죠. 가중치를 $\boldsymbol{W}$라 하면 L2 노름에 따른 가중치 감소는 $\frac{1}{2}\lambda\boldsymbol{W}^2$이 되고, 이 $\frac{1}{2}\lambda\boldsymbol{W}^2$을 손실 함수에 더합니다. 여기에서 $\lambda$는 정규화의 세기를 조절하는 **하이퍼파라미터**입니다. $\lambda$를 **크게** 설정할수록 **큰 가중치**에 대한 페널티가 커집니다. $\frac{1}{2}\lambda\boldsymbol{W}^2$에서 $\frac{1}{2}$은 $\frac{1}{2}\lambda\boldsymbol{W}^2$의 미분 결과인 $\lambda\boldsymbol{W}$를 조정하는 역할의 상수입니다.

가중치 감소는 모든 가중치 각각의 손실 함수에 $\frac{1}{2}\lambda\boldsymbol{W}^2$을 더합니다. 따라서 가중치의 기울기를 구하는 계산에서는 그동안의 오차역전파법에 따른 결과에 정규화 항을 미분한 $\lambda\boldsymbol{W}$를 더합니다.

> 💡 L2 노름은 **각 원소의 제곱**을 더한 것에 해당합니다. 가중치 $W=\begin{pmatrix} w_1 & w_2 & \cdots & w_n \end{pmatrix}$ 이 있다면, L2 노름에서는 $\sqrt{w_1^2+w_2^2+\cdots+w_n^2}$으로 계산할 수 있습니다. L2 노름 외에 L1 노름과 L$\infty$ 노름도 있습니다. L1 노름은 절댓값의 합, 즉 $\left \vert w_1 \right \vert+\left \vert w_2 \right \vert+\cdots+\left \vert w_n \right \vert$에 해당합니다. L$\infty$ 노름은 Max 노름이라고도 하며, 각 원소의 절댓값 중 가장 큰 것에 해당합니다. 정규화 항으로 L2 노름, L1 노름, L$\infty$ 노름 중 어떤 것도 사용할 수 있습니다.

그러면 가중치 감소를 한번 적용해보겠습니다. [Fig. 1.]에 $\lambda=0.1$로 가중치 감소를 적용합니다. 결과는 다음과 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/learning_techniques_29.png"
         title="Comparison of accuracy between train and test dataset when applied weight decay method"
         alt="Image of comparison of accuracy between train and test dataset when applied weight decay method"
         class="img_center"
         style="width: 60%"/>
    <figcaption>가중치 감소를 적용한 훈련 데이터와 시험 데이터의 정확도 비교</figcaption>
</figure>
