---
layout: post
title: Introduction to Reinforcement Learning-1
category: reinforcement learning
post-series: Reinforcement learning from scratch
post-order: 2
---

# 머릿말

이 시리즈에서는 강화 학습의 기초에 대해서 배워볼 예정입니다. 만약 기초가 잘 잡혀 있다면 수많은 논문과 책들이 그 자체로 훌륭한 학습 자료가 되겠지만, 초심자에게는 무작정 논문을 읽기 시작하더라도 이해가 잘 가지도 않을 뿐더러 개념들 사이에 구멍이 뚫려서 꼬리에 꼬리를 무는 의문점들이 생길 것입니다. 특히 강화 학습이라는 분야는 동물의 **행동심리학**으로부터 역사가 시작되기 때문에, 공학도에게는 해당 용어나 개념이 익숙하지 않아서 그 어려움을 더 증가시킬수도 있습니다.

따라서 이번 시리즈를 통해서 강화 학습의 기초 개념들을 구멍 없이 탄탄하게 채워나가면서 **쉽고 기본에 충실한** 학습이 될 수 있도록 노력할 예정입니다. 시리즈 대부분의 설명의 근간이 될 책  **바닥부터 배우는 강화 학습**을 적극적으로 인용하면서 부족한 부분이나 추가적인 설명이 필요하다 싶은 부분이 생기면 언제든지 추가할 예정입니다.

해당 시리즈를 통해서 강화 학습이라는 분야가 **큰 틀**안에서 어떻게 <ins>구성되고</ins>, 나아가 **각 알고리즘**이 어느 <ins>흐름안에</ins> 위치하는지 집중해주시면 감사하겠습니다.

# 강화 학습의 역사

강화 학습에 대해서 설명하기 전에, 강화 학습의 **역사**에 대해서 먼저 간략하게 설명해보겠습니다.

강화 학습의 역사는 아주 오래되고 깊은 역사를 가진 서로 다른 두 갈래에서 출발했습니다. 바로 **Trial & Error** 와 **Optimal Control**의 두 갈래입니다. Trial & Error는 위에서 언급했듯이 동물의 학습과 행동의 원리를 다루는 심리학인 **행동심리학**에서 영감을 받아 출발했습니다. 이 행동심리학에서는 보편적으로 동물의 학습을 "**강화이론**"으로 설명합니다. 이러한 학습의 이론을 가장 간결하게 표현한 사람이 바로 *Edward Thorndike*입니다.

<figure>
    <img src="/posts/study/machine learning/reinforcement learning/images/introduction_to_rl_1.png"
         title="Edward Thorndike"
         alt="Image of Edward Thorndike"
         class="img_center"
         style="width: 20%"/>
    <figcaption>Edward Lee Thorndike</figcaption>
</figure>

Trial & Error 학습에서는 "<ins>좋거나 나쁜 결과에 따라서 동물의 행동은 새롭게 재선택되어서 뒤따른다</ins>"라는 생각을 기반으로 합니다. 이를 Thorndike는 **효과의 법칙**(Law of Effect)이라고 불렀는데, 그 이유는 행동을 고르는 경향성에 따라서 해당 사건이 강화되는 효과를 묘사하기 때문입니다. Law of Effect는 Trial & Error 학습에서 **두가지** 중요한 면이 있습니다. 첫번째로 여러 행동 중 하나를 **선택**할 수 있다는 것과 두번째로 선택한 행동이 특정한 상황과 **연관**되어 있다는 점입니다. 다른 말로 표현하자면, Law of Effect는 **탐색**(search)과 **기억**(memory)을 합치는 기초적인 방법이라고 말할 수 있겠습니다. 탐색이란 각각의 상황에 대해서 취할 수 있는 수많은 행동들 중 하나를 고르는 것이고, 기억은 어떤 행동이 가장 좋은 상황에 도달할 수 있게 하는가를 기억하는 것입니다. 이렇게 탐색과 기억을 **합치는** 것이 강화 학습에서는 **필수적**입니다.

한편, **Optimal Control**이라는 용어는 1950년대 후반에 동역학 시스템의 시간에 따른 출력을 최소화하는 제어기를 설계하는 문제에서 등장했습니다. 이 문제에 대한 하나의 접근 방법으로 1950년대에 *Richard Bellman*이라는 사람이 제시한 **벨만 방정식**(Bellman equation)이 등장했습니다.

<figure>
    <img src="/posts/study/machine learning/reinforcement learning/images/introduction_to_rl_2.png"
         title="Richard Bellman"
         alt="Image of Richard Bellman"
         class="img_center"
         style="width: 20%"/>
    <figcaption>Richard Ernest Bellman</figcaption>
</figure>

벨만 방정식은 19세기의 *Hamilton*과 *Jacobi*의 이론을 확장해서 동역학 시스템의 상태와 **가치 함수**(value function), 혹은 **최적 리턴 함수**(optimal return function),라는 개념을 도입한 방정식입니다. 이 벨만 방정식을 풀어서 Optimal Control 문제를 해결하려는 방법론으로 나온 것이 바로 **Dynamic Programming**입니다. 이러한 것들이 모두 1980년대 후반에 합쳐지면서 현대 강화 학습의 알고리즘과 이론의 근간을 이루는 요소들이 되었습니다.

---

# 지도 학습과 강화 학습

**지도 학습**(supervised learning) 또한 강화 학습의 역사 속에서 항상 비교의 대상이 되며 같이 발전해왔습니다. 즉, 지도 학습과 강화 학습은 **유사한** 점이 많은 동시에 뚜렷한 **차이점**도 존재합니다. 먼저 두 학습 모두 다 **기계 학습**의 범주에 포함되어 있습니다. **기계 학습**이란 문자 그대로 기계에게 무언가를 배우게 하는 것을 가리킵니다. 정확히는 "학습"하는 방법을 이해하고 구축하는 데 전념하는 탐구 분야입니다. 그래서 **인공 지능**(artificial intelligence)의 한 부분으로 여겨지기도 합니다. 그럼 인공 지능은 기계 학습과 어떤 관계에 있을까요? 사실 인공 지능은 학술적으로 엄밀히 정의되지 않는 대중적인 용어입니다. 인위적으로 만들어진 지능을 넓게 이르는 말이라고도 할 수 있겠습니다. 기계 학습은 인공 지능을 **구현**하는 하나의 **방법론**입니다. 꼭 인공 지능을 기계 학습으로 구현하지 않아도 되지만 최근 화제가 되고 있는 인공 지능은 **대부분** 기계 학습을 통해 만들어 집니다. 지도 학습과 강화 학습은 모두 이러한 기계 학습이라는 큰 틀에 포함되는 **방법론**입니다.

<figure>
    <img src="/posts/study/machine learning/reinforcement learning/images/introduction_to_rl_3.png"
         title="Classification of machine learning"
         alt="Image of classification of machine learning"
         class="img_center"
         style="width: 50%"/>
    <figcaption>기계 학습의 분류</figcaption>
</figure>

[Fig. 3.]에서 확인하실 수 있듯이 기계 학습의 분류에는 지도 학습과 강화 학습 말고도 **비지도 학습**(unsupervised learning)이라는 방법론이 있습니다. 지도 학습은 지도자(혹은 정답)이 있는 상태에서 배우는 것이고, 강화 학습은 홀로 시행착오(trial & error)를 통해 배우는 것이지만 비지도 학습은 둘 다 아닙니다. 비지도 학습은 따로 정답이 주어지지 않은 데이터에 대해서 **패턴**을 배우는 것입니다. 비지도 학습은 주어진 데이터에서 데이터의 확률 밀도나 비슷한 데이터의 특징을 군집화해서 패턴을 스스로 구축하는 것입니다.

> 비지도 학습에 대한 자세한 설명은 [여기](https://en.wikipedia.org/wiki/Unsupervised_learning)를 참고해 주세요 😃.

## 지도 학습

지도 학습에는 **지도자** 혹은 **정답**이 주어져 있습니다. 예시를 하나 들어보겠습니다.

<figure>
    <img src="/posts/study/machine learning/reinforcement learning/images/introduction_to_rl_4.png"
         title="Dataset of creatures"
         alt="Image of dataset of creatures"
         class="img_center"
         style="width: 50%"/>
    <figcaption>생명체의 사진과 해당 생명체의 분류가 주어진 데이터</figcaption>
</figure>

[Fig. 4.]와 같이 각각 동물과 식물의 사진이 1만 장씩 있다고 가정하겠습니다. 이 데이터를 기반으로 지도 학습을 이용하면 일반적인 생물의 사진이 주어지면 해당 생물의 분류를 동물과 식물 중 하나로 출력하는 모델을 학습할 수 있습니다. 이때 학습에 사용되는 데이터를 **학습 데이터**(training data)라고 합니다. 지도 학습을 통해서 학습 데이터안의 입력과 정답 레이블 사이의 관계를 학습하는 것입니다. 학습하는 방법에 대해서는 수많은 방법론들이 있습니다(SVM, KNN, etc.). 최근에는 정말 입력과 정답 값만 가지고도 학습을 진행할 수 있는 **딥러닝**(deep learning) 방식을 많이 사용합니다.

> 딥러닝에 관한 자세한 설명은 [게시글](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/)을 참고해주세요 😁.

여하튼 이와 같이 모델이 학습되면 정답을 **모르는** 데이터를 입력으로 받아도 그에 해당하는 정답을 맞출 수 있습니다. 궁극적인 목적은 한번도 모델이 본 적 없는 데이터인 **테스트 데이터**(test data)를 입력 받아도 정답을 잘 출력하는 모델을 만드는 것입니다. 정리하자면 **지도 학습**은 학습 데이터를 이용해서 <ins>입력과 정답 사이의 관계</ins>를 학습해서 테스트 데이터를 입력 받아도 정답을 잘 출력하는 **범용적인** 모델을 학습시키는 방식입니다.

## 강화 학습

강화 학습에는 지도 학습과는 달리 **정답**이 주어지지 않습니다. 마찬가지로 예시를 하나 들어보겠습니다.

<figure>
    <img src="/posts/study/machine learning/reinforcement learning/images/introduction_to_rl_5.png"
         title="Maze escape game"
         alt="Image of maze escape game"
         class="img_center"
         style="width: 40%"/>
    <figcaption>미로 탈출 게임</figcaption>
</figure>

[Fig. 5.]와 같은 간단한 미로를 탈출하는 게임이 있다고 가정하겠습니다. 컴퓨터로 프로그래밍된 가상의 미로라면 사람은 간단하게 화면을 보고 키보드를 조작해서 미로를 탈출할 수 있을 것입니다. 하지만 기계의 경우에는 어떻게 탈출하는 지에 대한 구체적인 **방법**(**정답**)이 주어지지 않는다면 [Fig. 5.]와 같은 간단한 미로도 탈출하지 못할 수 있습니다. 이러한 상황 속에서 **시행착오**를 통해서 좀 더 **좋은 결과**를 낼 수 있는 선택지를 찾아서 **발전**하는 과정이 바로 **강화 학습**의 본질과 같다고 말할 수 있습니다.

다시 표현하자면 강화 학습은
* 쉽지만 추상적인 설명
  * 시행착오를 통해 발전해 나가는 과정
* 어렵지만 좀 더 정확한 설명
  * 순차적 의사결정 문제에서 누적 보상을 최대화 하기 위해 시행착오를 통해 행동을 교정하는 학습 과정
이라고 말할 수 있습니다.

쉽지만 추상적인 설명은 예시를 통해 이해해보면 금방 와닿을 것입니다. 하지만 설명이 몹시 **추상적이고** **비유적이기에** 본질적인 이해를 했다고 말하기에는 어려울 것 같습니다. 그래서 이제부터 <ins>어렵지만 좀 더 정확한 설명</ins>의 각 개념을 차례대로 짚어가면서 본격적으로 강화 학습에 대해서 파고들어 보겠습니다.