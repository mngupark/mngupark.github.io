---
layout: post
title: Backpropagation-4
category: deep learning
post-order: 14
---

이전 [post](https://gyuhub.github.io/posts/study/machine%20learning/deep%20learning/backpropation-3)에서는 활성화 함수의 역전파에 대해서 배웠습니다. 이번 post에서는 Affine 계층에 대해서 설명해보겠습니다.

---

# Affine 계층

신경망의 순전파에서는 가중치 매개변수의 총합을 계산하기 때문에 **행렬의 곱**을 사용했습니다. 그 과정에서 Python의 numpy 모듈의 np.dot() method를 사용했었습니다. 가중치 매개변수(**W**)와 입력 신호(**X**)를 곱하고 편향(**B**)을 더해서 이를 활성화 함수의 입력(**Y**)으로 주었습니다. 그리고 활성화 함수의 입력을 "Y=np.dot(X,W)+B"와 같이 계산할 수 있었습니다. 이때 행렬간의 곱셈이 행해지기 때문에 행렬간의 차원을 잘 맞춰줘야만 했었습니다.

> 📑 신경망의 순전파에서 수행하는 행렬의 곱을 기하학에서는 **어파인 변환**(affine transformation)이라고 합니다. 어파인 변환에 대한 자세한 정의는 [여기](https://en.wikipedia.org/wiki/Affine_transformation)을 참고해 주세요.

Affine transformation의 순전파에 대한 계산을 계산 그래프로 나타내보겠습니다. 행렬간의 곱셈을 '**dot**'노드로 나타내고, 각 행렬의 형상을 numpy의 shape 함수와 같이 표현하겠습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_19.jpg"
          title="Forward propagation of affine transformation"
          alt="Image of forward propagation of affine transformation"
          class="img_center"/>
     <figcaption>Affine 계층의 순전파</figcaption>
</figure>

[Fig. 1.]에서 각 입력 신호들은 행렬이기에 **Bold**로 굵게 표시하고 괄호안에 그 행렬의 형상을 적었습니다. 일반적인 경우를 계산하기 위해서 입력 신호의 차원을 **n**, 출력 신호의 차원을 **m**으로 계산했습니다. 이를 수식으로 표현한다면 아래와 같습니다.

$$
\boldsymbol{X}_{1\times n}=\begin{bmatrix} x_1 & x_2 & \cdots & x_n \end{bmatrix},\ 
\boldsymbol{W}_{n\times m}=\begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1m} \\ 
w_{21} & w_{22} & \cdots & w_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
w_{n1} & w_{n2} & \cdots & w_{nm} \end{bmatrix}, \\
\boldsymbol{X}\cdot \boldsymbol{W}=\begin{bmatrix} \alpha_1 & \alpha_2 & \cdots & \alpha_m \end{bmatrix}\ 
(\alpha_i = \sum_{k=1}^n x_k \cdot w_{ki},\ i=1,2,\cdots,m) \label{dot_product_matrix} \tag{1}
$$

식 $(\ref{dot_product_matrix})$에서 두 행렬간의 내적을 계산할 수 있습니다. 그럼 이 Affine 계층에 대한 역전파를 계산해보겠습니다. 그런데 이 내적의 편미분은 어떻게 계산할까요? 하나는 크기가 $1\times n$인 벡터이고 하나는 크기가 $n\times m$인 행렬인데 말입니다. 이는 이전에 배웠었던 Chain rule을 이용해서 증명할 수 있습니다. 이를 위해서 우선 오른쪽에서 왼쪽으로 하나씩 계산해보겠습니다.

## 덧셈 노드

먼저 $\frac{\partial{L}}{\partial{Y}}$가 역전파의 입력으로 들어온 뒤 덧셈 노드를 지나치게 됩니다. 덧셈 노드의 역전파는 상류의 신호를 그대로(1을 곱해서) 하류로 흘립니다. 따라서 아래의 계산 그래프와 같이 내적 노드의 입력으로 그대로 흘러가게 됩니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_20.jpg"
          title="Backpropagation of affine transformation"
          alt="Image of backpropagation of affine transformation"
          class="img_center"/>
     <figcaption>Affine 계층의 덧셈 노드의 역전파</figcaption>
</figure>

## 내적 노드

먼저 $\frac{\partial{L}}{\partial{\boldsymbol{X}}}$에 대한 수식은 아래와 같습니다.

$$
\begin{matrix}
\frac{\partial{\boldsymbol{X}\cdot \boldsymbol{W}}}{\partial{\boldsymbol{X}}}
&=&\frac{\partial{\begin{bmatrix} \alpha_1 & \alpha_2 & \cdots & \alpha_m \end{bmatrix}}}
{\partial{\begin{bmatrix} x_1 & x_2 & \cdots & x_n \end{bmatrix}}} \\
&=& \begin{bmatrix}
\frac{\partial{\alpha_1}}{\partial{x_1}} & \frac{\partial{\alpha_1}}{\partial{x_2}} & \cdots & \frac{\partial{\alpha_1}}{\partial{x_n}} \\
\frac{\partial{\alpha_2}}{\partial{x_1}} & \frac{\partial{\alpha_2}}{\partial{x_2}} & \cdots & \frac{\partial{\alpha_2}}{\partial{x_n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial{\alpha_m}}{\partial{x_1}} & \frac{\partial{\alpha_m}}{\partial{x_2}} & \cdots & \frac{\partial{\alpha_m}}{\partial{x_n}}
\end{bmatrix}
&=& \begin{bmatrix}
w_{11} & w_{21} & \cdots & w_{n1} \\
w_{12} & w_{22} & \cdots & w_{n2} \\
\vdots & \vdots & \ddots & \vdots \\
w_{1m} & w_{2m} & \cdots & w_{nm}
\end{bmatrix}\ (\frac{\partial{\alpha_i}}{\partial{x_j}}=w_{ji}) \\
&=& \boldsymbol{W}^T
\end{matrix} \\
\therefore \frac{\partial{L}}{\partial{\boldsymbol{X}}}=\frac{\partial{L}}{\partial{\boldsymbol{Y}}}\frac{\partial{\boldsymbol{Y}}}{\partial{\boldsymbol{X}}}
=\frac{\partial{L}}{\partial{\boldsymbol{Y}}}\boldsymbol{W}^T \label{L_X} \tag{2}
$$

$\frac{\partial{L}}{\partial{\boldsymbol{W}}}$에 대한 수식은 아래와 같습니다.  

$$
\begin{matrix}
\frac{\partial{\boldsymbol{X}\cdot \boldsymbol{W}}}{\partial{\boldsymbol{W}}}
&=&\frac{\partial{\begin{bmatrix} \alpha_1 & \alpha_2 & \cdots & \alpha_m \end{bmatrix}}}
{\partial{\begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1m} \\
w_{21} & w_{22} & \cdots & w_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
w_{n1} & w_{n2} & \cdots & w_{nm} \end{bmatrix}}}
\end{matrix} \label{L_W} \tag{3}
$$

식 $(\ref{L_X})$에서는 $\frac{\partial{L}}{\partial{\boldsymbol{X}}}$를 계산하니 2차원 행렬이 나왔습니다. 하지만 식 $(\ref{L_W})$에서의 $\frac{\partial{L}}{\partial{\boldsymbol{W}}}$를 계산하면 3차원 텐서가 나옵니다. 이를 직관적으로 이해하기 힘들기에 다른 방법을 사용해서 증명하겠습니다. 바로 **Chain rule**을 사용해서 증명하겠습니다.

$$
\begin{matrix}
\frac{\partial{L}}{\partial{\boldsymbol{W}}}&=&\frac{\partial{L}}{\partial{\boldsymbol{Y}}}\frac{\boldsymbol{Y}}{\partial{\boldsymbol{W}}} \\
&=&\begin{bmatrix}
\frac{\partial{L}}{\partial{w_{11}}} & \frac{\partial{L}}{\partial{w_{12}}} & \cdots & \frac{\partial{L}}{\partial{w_{1m}}} \\
\frac{\partial{L}}{\partial{w_{21}}} & \frac{\partial{L}}{\partial{w_{22}}} & \cdots & \frac{\partial{L}}{\partial{w_{2m}}} \\
\vdots & \vdots & \cdots & \vdots \\
\frac{\partial{L}}{\partial{w_{n1}}} & \frac{\partial{L}}{\partial{w_{n2}}} & \cdots & \frac{\partial{L}}{\partial{w_{nm}}} \end{bmatrix}, \\
\frac{\partial{L}}{\partial{w_{ij}}}&=&\sum_{k=1}^m \frac{\partial{L}}{\partial{y_{k}}}\frac{\partial{y_k}}{\partial{w_{ij}}} \\
&=&\frac{\partial{L}}{\partial{\boldsymbol{Y}}}\frac{\boldsymbol{Y}}{\partial{w_{ij}}}, \\
\frac{\partial{y_k}}{\partial{w_{ij}}}&=&\begin{cases} x_i\ (k=j) \\ 0\ (k\neq j) \end{cases}, \\
\frac{\partial{L}}{\partial{w_{ij}}}&=&\sum_{k=1}^m \frac{\partial{L}}{\partial{y_{k}}}\frac{\partial{y_k}}{\partial{w_{ij}}}
&=&\sum_{k=1}^m \frac{\partial{L}}{\partial{y_{k}}}x_i \\
&=&x_i \frac{\partial{L}}{\partial{y_j}}, \\
\frac{\partial{L}}{\partial{\boldsymbol{W}}}&=&
\begin{bmatrix}
x_1 \frac{\partial{L}}{\partial{y_1}} & x_1 \frac{\partial{L}}{\partial{y_2}} & \cdots x_1 \frac{\partial{L}}{\partial{y_m}} \\
x_2 \frac{\partial{L}}{\partial{y_1}} & x_2 \frac{\partial{L}}{\partial{y_2}} & \cdots x_2 \frac{\partial{L}}{\partial{y_m}} \\
\vdots & \vdots & \cdots & \vdots \\
x_n \frac{\partial{L}}{\partial{y_1}} & x_n \frac{\partial{L}}{\partial{y_2}} & \cdots x_n \frac{\partial{L}}{\partial{y_m}} \end{bmatrix} \\
&=& \boldsymbol{X}^T\frac{\partial{L}}{\partial{\boldsymbol{Y}}}
\end{matrix} \\
\therefore \frac{\partial{L}}{\partial{\boldsymbol{W}}}=\frac{\partial{L}}{\partial{\boldsymbol{Y}}}\frac{\partial{\boldsymbol{Y}}}{\partial{\boldsymbol{W}}}
=\boldsymbol{X}^T\frac{\partial{L}}{\partial{\boldsymbol{Y}}} \tag{4}
$$

증명 과정이 많이 길었는데, 요약하면 아래 수식과 같습니다.

$$
\begin{matrix}
\frac{\partial{L}}{\partial{\boldsymbol{X}}}&=&\frac{\partial{L}}{\partial{\boldsymbol{Y}}}\boldsymbol{W}^T \\
\frac{\partial{L}}{\partial{\boldsymbol{W}}}&=&\boldsymbol{X}^T\frac{\partial{L}}{\partial{\boldsymbol{Y}}} \end{matrix} \label{affine_backpropagation} \tag{5}
$$

식 $(\ref{affine_backpropagation})$를 계산 그래프로 표현하면 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_21.jpg"
          title="Backpropagation of affine transformation"
          alt="Image of backpropagation of affine transformation"
          class="img_center"/>
     <figcaption>Affine 계층의 역전파</figcaption>
</figure>

---

# 배치용 Affine 계층

지금까지 설명한 Affine 계층은 입력 데이터로 $\boldsymbol{X}$ 하나만을 고려한 것이었습니다. 이번에는 데이터 $\boldsymbol{N}$개를 묶어 순전파하는 경우인 **배치용 Affine 계층**을 설명해보겠습니다. 계산 그래프로는 아래와 같습니다.

<figure>
     <img src="/posts/study/machine learning/deep learning/images/backpropagation_22.jpg"
          title="Backpropagation of batch affine transformation"
          alt="Image of backpropagation of batch affine transformation"
          class="img_center"/>
     <figcaption>배치용 Affine 계층의 역전파</figcaption>
</figure>

[Fig. 4.]에서 기존과 달라진 부분은 입력인 $\boldsymbol{X}$의 형상이 $(n,)$에서 $(N,n)$으로 바뀐 것 뿐입니다. 이에 따라서 가중치나 편향 매개변수, 출력과 역전파 등이 행렬의 형상에 맞게 크기가 바뀌었습니다. 크기가 바뀌었지만 이전과 비슷하게 증명하는 과정을 통해서 식 $(\ref{affine_backpropagation})$와 같은 역전파를 사용할 수 있습니다.

> 이에 대한 증명 과정은 생략하겠습니다. 😅

주의해야할 점은 이를 Python으로 구현할 때 편향을 더하는 과정입니다. 순전파 때의 편향 덧셈은 $\boldsymbol{X}\cdot\boldsymbol{W}$에 대한 편향이 **각 데이터**에 더해집니다. 즉, 편향은 $\boldsymbol{N}$개의 데이터 각각에 다 더해집니다. 예를 들면 아래와 같습니다.

```python
>>> import numpy as np
>>> X_dot_W = np.array([[1, 2, 3],[6, 8, 10]])
>>> B = np.array([1, -1, 5])
>>> X_dot_W
array([[ 1,  2,  3],
       [ 6,  8, 10]]) 
>>> X_dot_W + B
array([[ 2,  1,  8],
       [ 7,  7, 15]])
```

순전파의 편향 덧셈은 각각의 데이터(1번째 데이터, 2번째 데이터, $\cdots$)에 더해집니다. 그래서 **역전파** 때는 각 데이터의 역전파 값이 **편향의 원소**에 모여야 합니다. 코드로는 아래와 같습니다.

```python
>>> import numpy as np
>>> dY = np.array([[7, 5, 3], [2, 4, 8]])
>>> dY
array([[7, 5, 3],
       [2, 4, 8]])
>>> dB = np.sum(dY, axis=0)
>>> dB
array([ 9,  9, 11])
```

편향의 역전파는 그 $\boldsymbol{N}$개의 데이터에 대한 미분을 데이터마다 더해서 구합니다. 그럼 이제 Python을 이용해서 Affine 계층을 구현해보겠습니다.
```python
class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape # for tensor input
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return (np.dot(self.x, self.W) + self.b)

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape) # for tensor input
        return dx
```