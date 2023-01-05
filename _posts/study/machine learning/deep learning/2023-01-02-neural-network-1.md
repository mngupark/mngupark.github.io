---
layout: post
title: Neural-Network-1
category: deep learning
post-order: 3
---
# 신경망

이전에 배운 퍼셉트론의 특징은 다음과 같습니다.

- 장점
-- 복잡한 함수도 다층 퍼셉트론을 통해 표현할 수 있다.
- 단점
-- 사용자가 직접 각 매개변수의 값을 정해야 한다. ex) $b$, $w_1$, $w_2$의 값을 조절해서 AND, NAND 그리고 OR 게이트를 만들 수 있습니다.

이러한 퍼셉트론의 단점을 신경망을 통해 해결할 수 있습니다. 바로 가중치 $w$와 편향 $b$의 값을 데이터로부터 자동으로 학습하는 것입니다.
신경망을 이용한 학습을 알아보기 전에 우선 신경망에 대해서 좀 더 깊게 파고들어 보겠습니다.

## 신경망의 정의

> 신경망(Neural Network)은 신경회로 또는 신경의 망으로, 현대적 의미에서는 인공 뉴런이나 노드로 구성된 인공 신경망을 의미한다.<br>
> 인공신경망은 기계학습과 인지과학에서 생물학의 신경망(동물의 중추신경계중 특히 뇌)에서 영감을 얻은 통계학적 학습 알고리즘이다.<br>
> 인공신경망은 시냅스의 결합으로 네트워크를 형성한 인공 뉴런(노드)이 학습을 통해 시냅스의 결합 세기를 변화시켜, 문제 해결 능력을 가지는 모델 전반을 가리킨다.<br>
> <ins>출처: [위키백과](https://ko.wikipedia.org/wiki/%EC%8B%A0%EA%B2%BD%EB%A7%9D)</ins>

## 신경망의 예시

신경망은 기본적으로 퍼셉트론과 공통점이 많습니다. 신경망을 그림으로 나타낸 예시는 아래와 같습니다.

<figure>
     <img src="/assets/images/study/machine_learning/deep_learning/2023-01-02-neural_network_1.jpg" 
          title="Example of neural network"
          alt="Example of neural network"
          class="img_center"
          style="width: 50%"/>
     <figcaption>신경망의 예시</figcaption>
</figure>

여기에서 가장 왼쪽 줄을 **입력층**(input layer), 오른쪽 줄을 **출력층**(output layer), 중간 층을 **은닉층**(hidden layer)이라고 합니다.
은닉층(hidden layer)은 입력층과 출력층과 달리 사람 눈에는 보이지 않기에 *은닉층*이라고 부릅니다. 그림에 잘 보이는데 보이지 않는다고 표현한 것이 헷갈릴 수 있습니다.
그 이유는 바로 입력층과 출력층은 사람이 그 역할을 쉽게 결정할 수 있습니다.

예를 들면 사람의 손글씨 사진을 학습하여 0에서 9사이 범위의 숫자를 **분류**하는 신경망이 있다고 해봅시다. 이 신경망의 입력은 픽셀(pixel)단위로 나누어진 사람의 손글씨 사진의 데이터일 것이고 출력은 0에서 9사이 숫자 중 신경망이 추측한 정답이 포함된 배열일 것입니다. 하지만 은닉층의 값들은 어떤 역할을 하는지 쉽게 결정할 수 없습니다. 오로지 신경망 스스로만 알고 있고(private) 외부에서는 알 수 없기 때문에 ***숨겨져있다***(hidden)라고 표현하는 것이죠.

위 그림에서 신경망은 모두 3층(입력층, 은닉층, 출력층)으로 구성되어 있지만, 가중치를 갖는 층은 총 2개뿐이기 때문에 2층 신경망이라고 부릅니다.

---

## 퍼셉트론 vs 신경망

뉴런(노드)이 연결되는 방식은 퍼셉트론과 신경망 모두 큰 차이가 없습니다. 퍼셉트론이 신호를 전달하는 방식은 아래와 같았습니다.

<figure>
     <img src="/assets/images/study/machine_learning/deep_learning/2022-12-26-perceptron_1.jpg" 
          title="Review of perceptron"
          alt="Review of perceptron"
          class="img_center"
          style="width: 50%"/>
     <figcaption>퍼셉트론이 신호를 전달하는 방식</figcaption>
</figure>

위 그림은 입력이 $x_1$, $x_2$이고 출력이 $y$인 퍼셉트론입니다. 수식으로 나타내면 아래와 같습니다.

$$
y=\begin{cases}
0 & (b+w_1 x_1 + w_2 x_2 \le 0) \\
1 & (b+w_1 x_1 + w_2 x_2 > 0)
\end{cases}
$$

하지만 위의 수식에는 편향 **b**가 보이지 않습니다. 이를 눈에 보이게 명시한다면 아래와 같이 표현할 수 있습니다.

<figure>
     <img src="/assets/images/study/machine_learning/deep_learning/2023-01-02-neural_network_2.jpg" 
          title="Biased perceptron"
          alt="Biased perceptron"
          class="img_center"
          style="width: 50%"/>
     <figcaption>편향이 추가된 퍼셉트론</figcaption>
</figure>

위의 그림은 이전의 퍼셉트론에 입력이 $1$이고 가중치가 $b$인 뉴런을 새로 추가한 그림입니다. 편향 $b$에 대한 입력 신호는 항상 1로 고정되어 있기에 회색으로 구분했습니다.
따라서 위의 퍼셉트론에는 $x_1, x_2, 1$이라는 3개의 신호가 입력되고 가중치가 각각 곱해진 뒤 그 값들의 합이 0을 넘으면 1이, 그렇지 않으면 0이라는 $y$가 출력됩니다.
이러한 퍼셉트론을 수식으로 간결하게 표현하면 아래와 같습니다.

$$
y=h(b+w_1 x_1+w_2 x_2) \\
h(x)=\begin{cases}
0 & (x \le 0) \\
1 & (x > 0)
\end{cases}
$$

여기서 $h(x)$는 $x$라는 입력이 들어오면 그 값이 0을 넘으면 1을 출력하고 그렇지 않으면 0을 출력하는 하나의 함수입니다. 이처럼 입력 신호의 총합을 출력 신호로 변환하는 함수를 **활성화 함수**(activation function)이라고 합니다.
<mark>활성화</mark>라는 이름이 말해주듯이 활성화 함수는 입력 신호의 총합이 출력 뉴런의 활성화를 일으키는지 결정하는 역할을 합니다.

이런 과정을 두 단계로 나눠서 생각해보겠습니다. 가중치와 입력 신호가 각각 곱해진 총합을 $a$라 하고, 그 합을 활성화 함수 $h(x)$에 입력해서 결과를 확인하는 것입니다.
이런 활성화 함수의 처리 과정을 식과 그림으로 나타내면 아래와 같습니다.

---

### 활성화 함수의 처리 과정

$$
a=b+w_1 x_1+w_2 x_2 \\
y=h(a)
$$

<figure>
     <img src="/assets/images/study/machine_learning/deep_learning/2023-01-02-neural_network_3.jpg" 
          title="Procedure of activation function"
          alt="Procedure of activation function"
          class="img_center"
          style="width: 50%"/>
     <figcaption>활성화 함수의 처리 과정</figcaption>
</figure>

이러한 활성화 함수가 퍼셉트론에서 신경망으로 확장되는 출발점이라고 생각할 수 있습니다. 그 이유는 어떤 활성화 함수를 사용하느냐에 따라서 다양한 신경망을 구축할 수 있기 때문입니다.