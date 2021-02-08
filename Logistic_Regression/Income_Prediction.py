import sys
import numpy as np


X_TRAIN_PATH = sys.argv[1]
Y_TRAIN_PATH = sys.argv[2]

print("Running the File", sys.argv[0])
print("Directory 1: ", X_TRAIN_PATH)
print("Directory 2: ", Y_TRAIN_PATH)


'''
For Testing
'''
'''

X_TRAIN_PATH = 'X_train'
Y_TRAIN_PATH = 'Y_train'
'''
X_train = np.genfromtxt(X_TRAIN_PATH, delimiter=',', skip_header=1)
Y_train = np.genfromtxt(Y_TRAIN_PATH, delimiter=',', skip_header=1)


"""Do the normalization of the data"""
def normalizeColumn(X, specifiedColumns = None, X_mean = None, X_stdev = None):
    if specifiedColumns == None:
        specifiedColumns = np.arange(X.shape[1])
        
    length = len(specifiedColumns)
    X_mean = np.reshape(np.mean(X[:,specifiedColumns], 0), (1, length))
    X_stdev = np.reshape(np.std(X[:,specifiedColumns], 0), (1, length))
        
    X[:,specifiedColumns] = np.divide(np.subtract(X[:, specifiedColumns], X_mean), X_stdev)
    
    return X, X_mean, X_stdev

'''Shuffle the data in a random order'''
def shuffle(X, Y):
    randomIndex = np.arange(len(X))
    np.random.shuffle(randomIndex)
    return (X[randomIndex], Y[randomIndex])

'''Split the data into training data and validation data'''
def splitTrainAndValidationData(X, Y, validation_size = 0.1):
    train_size = int(round(len(X) * (1 - validation_size)))
    
    return X[0:train_size], Y[0:train_size], X[train_size:None], Y[train_size:None]

def sigmoid(Z):
    
    return np.clip(1 / (1 + np.exp(-Z)), 1e-6, 1-1e-6)

def getY(X,w,b):
    
    return sigmoid(np.add(np.matmul(X, w),b))

def getRoundY(y):
    for i in range(len(y)):
        if y[i] < 0.5:
            y[i] = 0
        else:
            y[i] = 1
    return y

def computeCrossEntropy(y, y_label):
    
    return -np.dot(y_label, np.log(y)) - np.dot(1 - y_label, np.log(1 - y))

def getLoss(y, y_label):
    return computeCrossEntropy(y, y_label)

def getGradient(X, y_label, w, b):
    
    y = getY(X, w, b)
    loss = y_label - y
    w_grad = -np.mean(np.multiply(loss.T, X.T), axis = 1)
    b_grad = -np.mean(loss)
    
    return w_grad, b_grad

def getAccuracy(y, y_label):
    return np.sum(y == y_label) / len(y)


def train(X, Y, method = 'GRADIENT_ADAM'):
    validation_size = 0.1
    X_train, y_label, X_validation, y_validation = splitTrainAndValidationData(X, Y, validation_size)
    print(X_train.shape)
    print(y_label.shape)
    print(X_validation.shape)
    print(y_validation.shape)
    '''Initialize the weight and bias'''
    
    w = np.zeros(X_train.shape[1])
    b = np.zeros([1])
    
    eipsilon = 1e-8
    
    if method == 'GRADIENT_ADAM':
        beta1 = 0.9
        beta2 = 0.999
        
        v_w = np.zeros(w.shape)
        s_w = np.zeros(w.shape)
        v_b = np.zeros(b.shape)
        s_b = np.zeros(b.shape)
    
    
    max_interation = 41
    batch_size = 25
    learningRate = 0.0001
    
    step = 1
    
    trainAccuracy_list = []
    trainLoss_list = []
    validationAccuracy_list = []
    validationLoss_list = []
    
    
    
    for epoch in range(max_interation):
        X_train_epoch, y_train_epoch = shuffle(X_train, y_label)
        
        for i in range(int(np.floor(len(X_train)) / batch_size)):
            X_train_batch = X_train_epoch[i * batch_size: (i + 1) * batch_size]
            y_train_batch = y_train_epoch[i * batch_size: (i + 1) * batch_size]
            
            if method == 'GRADIENT':
                w_grad, b_grad = getGradient(X_train_batch, y_train_batch, w, b) 
                w = w - learningRate / np.sqrt(step) * w_grad
                b = b - learningRate / np.sqrt(step) * b_grad
                
            elif method == 'GRADIENT_ADAM':
                w_grad, b_grad = getGradient(X_train_batch, y_train_batch, w, b) 
                v_w = beta1 * v_w + (1 - beta1) * w_grad
                s_w = beta2 * s_w + (1 - beta2) * w_grad ** 2
                v_b = beta1 * v_b + (1 - beta1) * b_grad
                s_b = beta2 * s_b + (1 - beta2) * b_grad ** 2
                
                v_w_correction = v_w / (1 - beta1 ** step)
                s_w_correction = s_w / (1 - beta2 ** step)
                v_b_correction = v_b / (1 - beta1 ** step)
                s_b_correction = s_b / (1 - beta2 ** step)
                
                w = w - learningRate * v_w_correction / (np.sqrt(s_w_correction) + eipsilon)
                b = b - learningRate * v_b_correction / (np.sqrt(s_b_correction) + eipsilon)
            
            
            
            step += 1
            
        y_train_predicted = getY(X_train, w, b)
        trainLoss_list.append(getLoss(y_train_predicted, y_label) / len(y_train_predicted))
        
        y_train_predicted = getRoundY(y_train_predicted)        
        trainAccuracy_list.append(getAccuracy(y_train_predicted, y_label))
        
        
        y_validation_predicted = getY(X_validation, w, b)
        validationLoss_list.append(getLoss(y_validation_predicted, y_validation) / len(y_validation_predicted))
        
        y_validation_predicted = getRoundY(y_validation_predicted)
        validationAccuracy_list.append(getAccuracy(y_validation_predicted, y_validation))
        
        
        print("Epoch", epoch, " Training Accuracy: ", (getAccuracy(y_train_predicted, y_label)), " Validation Accuracy: ", (getAccuracy(y_validation_predicted, y_validation)))
        
    return w, b, trainAccuracy_list, validationAccuracy_list, trainLoss_list, validationLoss_list

X_train, X_mean, X_stdev = normalizeColumn(X_train)

weight, bias, trainAccList, validationAccList, trainLossList, validationLossList = train(X_train, Y_train, method = 'GRADIENT_ADAM')


import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(trainAccList)
plt.plot(validationAccList)

plt.figure(2)

plt.plot(trainLossList)
plt.plot(validationLossList)
plt.legend(['train', 'validation'])

plt.show()

        

    
    


