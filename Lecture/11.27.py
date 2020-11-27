# %% Binary Cross Entropy Loss 구하기
import math
ground_truth = 0.7
prediction1 = 0.5
prediction2 = 0.9
prediction3 = 0.4

loss1 = -(ground_truth*math.log(prediction1) + (1 - ground_truth)*math.log(1-prediction1))
loss2 = -(ground_truth*math.log(prediction1) + (1 - ground_truth)*math.log(1-prediction2))
loss3 = -(ground_truth*math.log(prediction1) + (1 - ground_truth)*math.log(1-prediction3))

# %% Binary Cross Entropy Lostt 구하기(평균)
import math
ground_truth = 0.7
prediction1 = 0.5
prediction2 = 0.9
prediction3 = 0.4

N = 3

loss1 = -(ground_truth*math.log(prediction1) + (1 - ground_truth)*math.log(1-prediction1))
loss2 = -(ground_truth*math.log(prediction1) + (1 - ground_truth)*math.log(1-prediction2))
loss3 = -(ground_truth*math.log(prediction1) + (1 - ground_truth)*math.log(1-prediction3))

bce_loss = (loss1 + loss2 + loss3)/N
print(bce_loss)

# %% BCE + Numpy
import numpy as np

labels = np.array([0.9, 0.7, 0.4])
predictions = np.array([0.7, 0.5, 0.7])

pos_val = labels*np.log(predictions)
neg_val = (1-labels)*np.log(1-predictions)

sum_ = pos_val + neg_val
bce_loss = -1*np.mean(sum_)

bce_loss = -1*np.mean(labels*np.log(predictions) + (1-labels)*np.log(1-predictions))

print(bce_loss)

# %% BCE in Deep Learning
import numpy as np

#2 classes
labels = np.array([1, 0, 1])
labels = np.array([[0, 1], [1, 0], [0, 1]])

#3 classes
labels = np.array([1, 0, 2])
labels = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

# %%
import numpy as np

labels = np.array([1, 0, 1])
predictions = np.array([0.9, 0.1, 0.6])
bce_loss = -1*np.mean(labels*np.log(predictions) + (1-labels)*np.log(1-predictions))

labels = np.array([[0, 1], [1, 0], [0, 1]])
predictions = np.array([[0.1, 0.9], [0.9, 0.1], [0.4, 0.6]])

inner = labels*np.log(predictions)
inner = np.sum(inner, axis=1)
bce_loss = -1*np.mean(inner)

# %%
labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([0, 1, 0, 0, 0, 1, 1])

accuracy = np.sum((labels == predictions).astype(np.int))/labels.shape[0]*100
accuracy = np.around(accuracy, 2)
print(accuracy, '%')

labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.6, 0.7])

predictions = (predictions >= 0.5).astype(np.int)
accuracy = np.sum((labels == predictions).astype(np.int))/labels.shape[0]*100
accuracy = np.around(accuracy, 2)
print(accuracy, '%')

# %% 함수
def test_funct(input1, input2):
    result = input1 + input2
    return result

def get_mean(input_arr):
    sum_ = 0
    for cnt, val in enumerate(input_arr):
        sum_ += val
    return sum_ / (cnt+1)

def get_accuracy(labels, predictions):
    predictions = (predictions >= 0.5).astype(np.int)
    accuracy = np.sum((labels == predictions).astype(np.int))/labels.shape[0]*100
    accuracy = np.around(accuracy, 2)
    return accuracy

# %%
def test_function(a, b):
    result = a + b
    return result

def addition(a, b):
    return a + b

def subtraction(a, b):
    return a - b

c = addition(10, 20) + subtraction(10, 20)

# %%
#input x, output x
def say_hello():
    print('Hello World!')
    
#input x, output o
def get_random_number():
    random_number = np.random.normal(0, 1, size=(1, ))
    return random_number

#input o, output x
def say_hello2(name):
    print('Hello ', name)
    
#input o, output o
def get_mean(score_list):
    sum_ = 0
    for cnt, score in enumerate(score_list):
        sum_ = score
    return sum_ / (cnt+1)

# %% Namespace
#global namespace
a = 10
#print(locals(), '\n')

def test_function():
    #local namespace
    a = 20
    print(a)

def test_function2():
    #local namespace
    a = 30
    print(a)

test_function()
test_function2()