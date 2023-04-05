import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression (trainingImages, trainingLabels, epsilon, batchSize, alpha, m):
    n = np.shape(trainingImages)[1]
    m_features = np.shape(trainingImages)[0] #785
    l_labels = np.shape(trainingLabels)[1] #10
    W = .00001 * np.random.randn(m_features, l_labels)
    shuffle = np.random.permutation(n)
    m_shuffled = trainingImages[:, shuffle]
    labels_shuffled = trainingLabels[shuffle, :]
    num_epochs = 5
    for e in range(num_epochs):
        print(f"Progress: {e*20}%")
        for i in range((int(n/batchSize)) - 1):
            xtilde = m_shuffled[: , i*batchSize:(i*batchSize) + batchSize]
            ytilde = labels_shuffled[i*batchSize:(i*batchSize) + batchSize , :]
            doPrint = False
            if e == num_epochs-1 and i > (n/batchSize)-20 or e == 0 and i < 20:
                doPrint = True
            W = W - epsilon * gradfCE(W, xtilde, ytilde, alpha, doPrint, m, batchSize)
    return W

def accuracy (wtilde, Xtilde, y):
    # print (Xtilde)
    # print (wtilde)
    X = np.transpose(Xtilde)
    yhat = np.dot(X, wtilde)
    # print ("yhat")
    # print (yhat)
    # print ("y")
    # print (y)
    predictions = yhat.argmax(axis=1)
    actual = y.argmax(axis=1)
    #print (np.mean(predictions == actual))
    return np.mean(predictions == actual)

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfCE (W, Xtilde, y, alpha = 0., doPrint = False, m=10, batchSize=0):
    X = np.transpose(Xtilde)
    shape = np.shape(X)
    n = shape[0]
    wtilde_remove_bias = W.copy()
    wtilde_remove_bias[-1] = 0
    Z = np.exp(np.dot(X, W))
    # print(f"ZShape: {np.shape(Z)}\n{Z}")
    #yhat = (np.dot(Z, (1/np.sum(Z))))
    yhat_step = np.sum(Z, axis=1)
    # print(f"yhat step: {yhat_step}")
    yhat_step2 = np.reshape(np.repeat(yhat_step, m), (batchSize,m))
    # print(f"yhat step2: {yhat_step2}")
    yhat = np.divide(Z, yhat_step2) 
    # print ('yhat')
    # print (yhat)
    reg = (1/n) * np.dot(Xtilde, (yhat - y)) + (alpha/n) * wtilde_remove_bias

    if doPrint:
        # print(f"y shape {np.shape(y)}\nyhat shape {np.shape(yhat)}")
        lossP1 = (-1/n)*np.sum(np.dot(y, np.transpose(np.log(yhat))))
        lossP2 = ((alpha / (2*n))*np.sum(np.dot(np.transpose(wtilde_remove_bias), wtilde_remove_bias)))
        loss =  lossP1 + lossP2
        print(f"Loss: {loss}")
    return reg

def reshapeAndAppend1s (faces):
    shape = np.shape(faces)
    N = shape[0]
    #print (N) #60000
    M = shape[1]
    #print (M) #784
    transposed = np.array(np.transpose(faces))
    reshaped = np.array(transposed.reshape(M, N))
    new_row = np.ones(N)
    xtilde = np.array(np.vstack([reshaped, new_row]))
    return xtilde

def reshapeLabelVectors (labels, m):
    shape = np.shape(labels)
    N = shape[0]
    oneHotMatrix = np.zeros(shape= (N,m))#10 for hw, 2 for kaggle
    oneHotMatrix[np.arange(N), labels] = 1
    return oneHotMatrix

def visualize(img):
    X = np.reshape(img, (28, 28))
    # Plot the image
    plt.imshow(X, cmap='gray')
    plt.show()

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...
    trainImages = reshapeAndAppend1s(trainingImages)
    print (np.shape(trainImages))
    testImages = reshapeAndAppend1s(testingImages)

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance,
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    # ...
    trainLabels = reshapeLabelVectors(trainingLabels, 10)
    testLabels = reshapeLabelVectors(testingLabels, 10)
    print (np.shape(trainLabels))
    # Train the model
    Wtilde = softmaxRegression(trainImages, trainLabels, epsilon=0.1, batchSize=100, alpha=.1, m=10)
    testAcc = accuracy(Wtilde, testImages, testLabels)
    acc = accuracy(Wtilde,trainImages, trainLabels)
    print(f"Train Acc: {acc}\nTest Acc: {testAcc}")

    wtilde_remove_bias = Wtilde.copy()[:-1]
    for i in range(10):
        img = wtilde_remove_bias[:, i].reshape(28,28)
        visualize(img)
    # Visualize the vectors
    # ...


