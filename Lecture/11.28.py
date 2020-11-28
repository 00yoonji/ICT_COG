# %%
# 평균, 분산 구하기(국어, 수학, 영어)
# M, m, M_idx, m_idx
# MSE error revisited
# 2D Convolution
# BE(integer format, one-hot vector format)
# sigmoid, tanh, ReLU

''' 수학
1. derivatives
2. chain rule
3. linear regression
4. logistic regression
'''

# linear regression
# logistic regression
# %%
import numpy as np

def get_random_scores(n_student):
    scores = np.random.uniform(low=0, high=100.0, size=(n_student, ))
    scores = scores.astype(np.int)
    return scores

def get_mean(scores):
    sum_ = 0
    for cnt, score in enumerate(scores):
        sum_ += score
    mean = sum_ / (cnt+1)
    return mean


def get_variance(scores):
    mean = get_mean(scores)
    
    square_sum = 0
    for cnt, score in enumerate(scores):
        square_sum += score**2
        
    variance = square_sum/(cnt+1) - mean**2
    return variance

n_student = 100
scores = get_random_scores(n_student)
mean = get_mean(scores)
var = get_variance(scores)


def get_mean_variance(scores):
    scores_sum = 0
    scores_squared_sum = 0
    
    for score in scores:
        scores_sum += score
        scores_squared_sum += score**2
    
    scores_mean = scores_sum / n_student
    scores_var = (scores_squared_sum / n_student) / (scores_mean**2)
    return scores_mean, scores_var

mean, var = get_mean_variance(scores)

# %%
n_student = 100
scores = get_random_scores(n_student)

def get_M_m(scores, M, m):
    max_score, min_score = None, None
    for idx, score in enumerate(scores):
        if max_score == None or max_score < score:
            max_idx = idx
            max_score = score
        
        if min_score == None or min_score > score:
            min_idx = idx
            min_score = score
            
    if M == True and m == True:
        return max_score, max_idx, min_score, min_idx
    if M == True and m == False:
        return max_score, max_idx
    if M == False and m == True:
        return min_score, min_idx

M, M_idx, m, m_idx = get_M_m(scores, M=True, m=True)
M, M_idx = get_M_m(scores, M=True, m=False)
m, m_idx = get_M_m(scores, M=False, m=True)


def calculator(input1, input2, operand):
    if operand == '+':
        return input1 + input2
    elif operand == '-':
        return input1 - input2
    elif operand == '*':
        return input1 * input2
    elif operand == '/':
        return input1 / input2
    else:
        print("Unknown Operand")

result = calculator(10, 20, '+')
result = calculator(10, 20, '-')
result = calculator(10, 20, '*')
result = calculator(10, 20, '/')
result = calculator(10, 20, '^')

# %%
from termcolor import colored

print(colored('Hello World!', 'red', 'on_white', attrs=['blink']))

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + 'Dataset is Loading')

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + 'Model is Loading')

# %%
from tqdm import tqdm
import time

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + 'Dataset is Loading')
for i in tqdm(range(10000)):
    time.sleep(0.01)

# %% MSE error revisited
n_point = 100
ground_truths = np.random.normal(0, 1, (n_point, ))
predictions = np.random.normal(0, 1, (n_point, ))

def get_mse(ground_truths, predictions):
    mse_error = np.mean((ground_truths - predictions)**2)
    return mse_error

# %% 2D Convolution
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def conv2d(img_gray, filter_):
    filter_len = filter_.shape[0]
    H, W = img_gray.shape
    
    img_convolved = np.zeros(shape=(H - filter_len + 1, W - filter_len + 1))
    
    for row_idx in range(H - filter_len):
        for col_idx in range(W - filter_len):
            img_segment = img_gray[row_idx : row_idx + filter_len, col_idx : col_idx + filter_len]
            convolution = np.sum(img_segment * filter_)
            img_convolved[row_idx, col_idx] = convolution
    
    return img_convolved

img = plt.imread('./test_image.jpg')
img_gray = rgb2gray(img)

filter_ = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

img_convolved = conv2d(img_gray, filter_)

fig, axes = plt.subplots(1, 2, figsize=(20, 12))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'gray')

# %%
def get_random_scores(n_student, n_subject):
    scores = np.random.uniform(low=0, high=100.0, size=(n_student, n_subject))
    scores = scores.astype(np.int)
    return scores

def get_mean(scores):
    #scores : 1-D(Vector) or 2-D(Matrix)
    if len(scores.shape) == 1:  #vector
        mean = np.mean(scores)
    else:   #matrix
        mean = np.mean(scores, axis = 0)
    return mean

def get_mean(scores):
    mean = np.mean(scores, axis=0)
    return mean

def get_mean(scores):
    sum_ = 0
    for cnt, score in enumerate(scores):
        sum_ += score
    mean = sum_ / (cnt+1)
    return mean

scores = get_random_scores(100, 5)
print(scores.shape)
mean = get_mean(scores)
print(mean)

# %% BCE(integer format, one-hot vector format)
def get_bce(labels, predictions):
    if len(labels.shape) == 1:
        tmp_list = []
        for label in labels:
            if label == 0:
                tmp_list.append([1, 0])
            elif label == 1:
                tmp_list.append([0, 1])
        labels = np.array(tmp_list)
    
    losses = -1*np.sum(labels*np.log(predictions), axis=1)
    loss = np.mean(losses)
    return loss

labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])

bce = get_bce(labels, predictions)
print(bce)

labels = np.array([[0, 1], [1, 0],
                   [0, 1], [1, 0],
                   [0, 1], [1, 0],
                   [0, 1]])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])

bce = get_bce(labels, predictions)
print(bce)

# %% sigmoid, tanh, relu

# np.exp, np.maximum
affine = np.array([-5, 2, 6, 8, 1])

def sigmoid(affine):
    return 1/(1 + np.exp(affine))

def tanh(affine):
    return (np.exp(affine) - np.exp(-affine)) / (np.exp(affine) + np.exp(-affine))

def relu(affine):
    return np.maximum(0, affine)

x_range = np.linspace(-10, 10, 300)
a_sigmoid = sigmoid(x_range)
a_tanh = tanh(x_range)
a_relu = relu(x_range)

plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(x_range, a_sigmoid, label='Sigmoid')
ax.plot(x_range, a_tanh, label='Tanh')
ax.plot(x_range, a_relu, label='ReLU')

ax.tick_params(labelsize=20)
ax.legend(fontsize=40)

# %%
import numpy as np
import matplotlib.pyplot as plt
'''
x = np.linspace(0.001, 2, 300)
y = np.log(x)
dy_dx = 1/x

# y = np.exp(x)
# dy_dx = np.exp(x)

# y = (x-1)*(x-2)
# dy_dx = 2x - 3

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].plot(x, y)
axes[1].plot(x, dy_dx)

axes[0].tick_params(labelsize=20)
axes[1].tick_params(labelsize=20)
'''
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)

X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + X2**2

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection = '3d')

ax.plot_wireframe(X1, X2, Z)

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

n_point = 100

x = np.random.normal(0, 1, size=(n_point, ))
y = 3*x + 0.1*np.random.normal(0, 1, size=(n_point, ))

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].scatter(x, y)
axes[0].axvline(x=0, linewidth=3, color='black', alpha=0.5)
axes[0].axhline(y=0, linewidth=3, color='black', alpha=0.5)

a = -10
n_iter = 200
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []

for i in range(n_iter):
    predictions = a*x
    
    mse = np.mean((y - predictions)**2)
    
    dl_da = -2*np.mean(x*(y - predictions))
    a = a - learning_rate * dl_da
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range
    ax.plot(x_range, y_range, color=cmap(i))
    
    losses.append(mse) 

axes[1].plot(losses)

# %%
n_point = 1000

x = np.random.normal(0, 1, size=(n_point, ))
y = 3*x + 2 + 0.5*np.random.normal(0, 1, size=(n_point, ))

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes[0].scatter(x, y)
axes[0].axvline(x=0, linewidth=3, color='black', alpha=0.5)
axes[0].axhline(y=0, linewidth=3, color='black', alpha=0.5)

a, b = np.random.normal(0, 1, size=(2, ))
n_iter = 300
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
a_list, b_list = [], []

for i in range(n_iter):
    a_list.append(a)
    b_list.append(b)
    
    predictions = a*x + b
    mse = np.mean((y - predictions)**2)
    
    dl_da = -2*np.mean(x*(y - predictions))
    dl_db = -2*np.mean(y - predictions)
    
    a = a - learning_rate * dl_da
    b = b - learning_rate * dl_db
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range
    ax.plot(x_range, y_range, color=cmap(i))
    
    losses.append(mse)

axes[1].plot(losses)
axes[2].plot(a_list, label='a')
axes[2].plot(b_list, label='b')
axes[2].tick_params(labelsize=20)
axes[2].grid(axis='y')
axes[2].legend(fontsize=20)

# %%
n_point = 1000

x1 = np.random.normal(0, 1, size=(n_point, ))
x2 = np.random.normal(0, 1, size=(n_point, ))
y = 3*x1 + 2*x2 - 1

fig, axes = plt.subplots(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y)

fig.tight_layout()
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

a, b, c = np.random.normal(0, 1, size=(2, ))
n_iter = 300
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
a_list, b_list, c_list = [], [], []

for i in range(n_iter):
    a_list.append(a)
    b_list.append(b)
    c_list.append(c)
    
    predictions = a*x1 + b*x2 + c
    mse = np.mean((y - predictions)**2)
    
    dl_da = -2*np.mean(x*(y - predictions))
    dl_db = -2*np.mean(y - predictions)
    dl_dc = -2*np.mean(y - predictions)
    
    a = a - learning_rate * dl_da
    b = b - learning_rate * dl_db
    c = c - learning_rate * dl_dc
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range
    ax.plot(x_range, y_range, color=cmap(i))
    
    losses.append(mse)

axes[1].plot(losses)
axes[2].plot(a_list, label='a')
axes[2].plot(b_list, label='b')
axes[2].tick_params(labelsize=20)
axes[2].grid(axis='y')
axes[2].legend(fontsize=20)

# %%
n_point = 100
x = np.random.normal(0, 1, size=(n_point, ))
x_noise = x + 0.2*np.random.normal(0, 1, size=(n_point, ))

y = (x >= 0).astype(np.int)

fig, ax = plt.subplots(figsize=(20, 10))
ax.scatter(x, y)

x_range = np.linspace(x.min(), x.max(), 100)
y_range = 1 / (1 + np.exp(-10*x_range))
ax.plot(x_range, y_range, color='r')
ax.grid()

# %%
n_point = 100
x = np.random.normal(0, 1, size=(n_point, ))
x_noise = x + 0.2*np.random.normal(0, 1, size=(n_point, ))

y = (x >= 0).astype(np.int)

fig, ax = plt.subplots(figsize=(20, 10))
ax.scatter(x, y)

a, b = np.random.normal(0, 1, size=(2, ))
n_iter = 300
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
for i in range(n_iter):
    affine = a*x + b
    predictions = 1 / (1 + np.exp(-affine))
    
    loss = -1*np.mean(y*np.log(predictions) + (1-y)*np.log(1-predictions))
    
    dl_da = -np.mean(x*(y - predictions))
    dl_db = -np.mean(y - predictions)
    
    a = a - learning_rate * dl_da
    b = b - learning_rate * dl_db
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range + b
    y_range = 1 / (1 + np.exp(-y_range))
    ax.plot(x_range, y_range, color=cmap(i))
    
    losses.append(mse)

ax.grid()

# %%
n_point = 100
x = np.random.normal(0, 1, size=(n_point, ))
x_noise = x + 0.2*np.random.normal(0, 1, size=(n_point, ))
    
y = (x_noise >= 0).astype(np.int)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].scatter(x, y)

a, b = np.random.normal(0, 1, size=(2, ))
n_iter = 1000
learning_rate = 0.1
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []

for i in range(n_iter):
    
    bce = 

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range + b
    y_range = 1/(1 + np.exp(-y_range))
    axes[0].plot(x_range, y_range, color=cmap(i))

    losses.append(bce)

axes[0].grid()
axes[1].plot(losses)
























