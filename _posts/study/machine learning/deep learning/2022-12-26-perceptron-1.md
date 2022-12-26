---
layout: post
title: Perceptron-1
category: deep learning
---

퍼셉트론(Perceptron)은 프랑크 로젠블라트가 1957년에 고안한 알고리즘이다.

<img src="/images/study/machine_learning/deep_learning/2022-12-26-frank-rosenblatt.jpg" 
     title="Frank Rosenblatt"
     alt="Frank Rosenblatt"
     class="img_center"/>

퍼셉트론은 신경망(딥러닝)의 기원이 되는 알고리즘이기에 이 구조를 배우는 것은 신경망과 딥러닝으로 나아가는 중요한 시작점이라고 할 수 있다.

# 퍼셉트론이란?
퍼셉트론은 다수의 **신호**를 입력으로 받아 하나의 **신호**를 출력한다. 여기서 **신호**란 전류나 강물처럼 흐름이 있는 것을 비유할 수 있다. 하지만 퍼셉트론은 전류와 달리 '흐른다/안 흐른다(1이나 0)'의 두 가지 값을 가질 수 있다. 앞으로도 1을 '신호가 흐른다', 0을 '신호가 흐르지 않는다'라는 의미로 해석하겠다.