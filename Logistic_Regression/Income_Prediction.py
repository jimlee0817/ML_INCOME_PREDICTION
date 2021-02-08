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
def _normalizeColumn(X, train = True, specifiedColumns = None, xMean = None, xStdev = None):
    if train:
        if specifiedColumns == None:
            specifiedColumns = np.arange(X.shape[1])
        length = len(specifiedColumns)
        xMean = np.reshape(np.mean(X[:,specifiedColumns], 0), (1, length))
        xStdev = np.reshape(np.std(X[:,specifiedColumns], 0), (1, length))
        
    X[:,specifiedColumns] = np.divide(np.subtract(X[:, specifiedColumns], xMean), xStdev)
    
    return X, xMean, xStdev

'''Shuffle the data in a random order'''
def _shuffle(X, Y):
    randomIndex = np.arange(len(X))
    np.random.shuffle(randomIndex)
    return X[randomIndex], Y[randomIndex]

'''Split the data into training data and validation data'''
def _splitTrainAndValidation(X, Y, valSize = 0.1):
    trainSize = round(len(X) * (1 - valSize))
    
    return X[0:trainSize], Y[0:trainSize], X[trainSize:None], Y[trainSize:None]

def _sigmoid(Z):
    
    return np.clip(1 / (1 + np.exp(-Z)), 1e-6, 1-1e-6)

def _yPredicted(X,w,b):
    
    return _sigmoid(np.add(np.matmul(X, w),b))

def _infer(X,w,b):
    
    return np.round(_yPredicted(X, w, b))

def _crossEntropy(y, yLabel):
    
    return -np.dot(yLabel, np.log(y)) - np.dot(1 - yLabel, np.log(1 - y))

def _gradient(X, yLabel, w, b):
    
    y = _yPredicted(X, w, b)
    L = yLabel - y
    wGrad = -np.mean(np.multiply(L.T, X.T), axis = 1)
    bGrad = -np.mean(L)
    
    return wGrad, bGrad

def _gradient_regularization(X, yLabel, w, b, lamda):
    # return the mean of the graident
    y = _yPredicted(X, w, b)
    L = yLabel - y
    wGrad = -np.mean(np.multiply(L.T, X.T), axis = 1) + lamda * w
    bGrad = -np.mean(L)
    return wGrad, bGrad

def _accuracy(y, yLabel):
    return np.sum(y == yLabel) / len(y)


def _train(X, Y, method = 'GRADIENT_ADAM'):
    validation_size = 0.1
    X_train, y_label, X_validation, y_validation = _splitTrainAndValidation(X, Y, validation_size)
    print(X_train.shape)
    print(y_label.shape)
    print(X_validation.shape)
    print(y_validation.shape)
    '''Initialize the weight and bias'''
    
    w = np.zeros(X_train.shape[1])
    b = np.zeros([1])
    
    eipsilon = 1e-8
    
    beta1 = 0.9
    beta2 = 0.999
    
    v_w = np.zeros(w.shape)
    s_w = np.zeros(w.shape)
    v_b = np.zeros(b.shape)
    s_b = np.zeros(b.shape)
    
    maxIteration = 41
    batchSize = 40
    learningRate = 0.001
    
    step = 1
    
    trainAccurancy = []
    validationAccurancy = []
    
    for epoch in range(maxIteration):
        '''Debug'''
        '''
        print(w)
        print(b)
        '''
        X_train, y_train = _shuffle(X_train, y_label)
        
        for i in range(int(np.floor(len(X_train)) / batchSize)):
            X_train_batch = X_train[i * batchSize: (i + 1) * batchSize]
            y_train_batch = y_train[i * batchSize: (i + 1) * batchSize]
            
            if method == 'GRADIENT_REGULARIZATION':
                wGrad, bGrad = _gradient_regularization(X_train_batch, y_train_batch, w, b, 0.001)
                w = w - learningRate / np.sqrt(step) * wGrad
                b = b - learningRate / np.sqrt(step) * bGrad
            elif method == 'GRADIENT':
                wGrad, bGrad = _gradient(X_train_batch, y_train_batch, w, b) 
                w = w - learningRate / np.sqrt(step) * wGrad
                b = b - learningRate / np.sqrt(step) * bGrad
            elif method == 'GRADIENT_ADAM':
                wGrad, bGrad = _gradient(X_train_batch, y_train_batch, w, b) 
                v_w = beta1 * v_w + (1 - beta1) * wGrad
                s_w = beta2 * s_w + (1 - beta2) * wGrad ** 2
                v_b = beta1 * v_b + (1 - beta1) * bGrad
                s_b = beta2 * s_b + (1 - beta2) * bGrad ** 2
                v_w_correction = v_w / (1 - beta1 ** step)
                s_w_correction = s_w / (1 - beta2 ** step)
                v_b_correction = v_b / (1 - beta1 ** step)
                s_b_correction = s_b / (1 - beta2 ** step)
                
                w = w - learningRate * v_w_correction / (np.sqrt(s_w_correction) + eipsilon)
                b = b - learningRate * v_b_correction / (np.sqrt(s_b_correction) + eipsilon)
            
            
            
            step += 1
            
        y_trainPredicted = _infer(X_train, w, b)
        trainAccurancy.append(_accuracy(y_trainPredicted, y_train))
        
        y_validationPredicted = _infer(X_validation, w, b)
        validationAccurancy.append(_accuracy(y_validationPredicted, y_validation))
        
        print("Epoch", epoch, " Accuracy: ", (_accuracy(y_trainPredicted, y_train)))
        
    return w, b, trainAccurancy, validationAccurancy


X_train, X_mean, X_stdev = _normalizeColumn(X_train)

weight, bias, trainAccList, validationAccList = _train(X_train, Y_train, method = 'GRADIENT_ADAM')

'''
import matplotlib.pyplot as plt

plt.plot(trainAccList)
plt.plot(validationAccList)
plt.legend(['train', 'validation'])
plt.show()
'''




        

    
    


