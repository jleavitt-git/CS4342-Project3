import pandas
import numpy as np
import homework3_jleavitt as hw

if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    sibSP = d.SibSp.to_numpy()
    # Train model using part of homework 3.
    # ...

    X = np.asarray([sex, Pclass, sibSP]).T

    testImages = hw.reshapeAndAppend1s(X)
    testLabels = hw.reshapeLabelVectors(y, 2)

    Wtilde = hw.softmaxRegression(testImages, testLabels, epsilon=0.1, batchSize=100, alpha=.1, m=2)

    trainAcc = hw.accuracy(Wtilde, testImages, testLabels)
    print(f"Train Acc: {trainAcc}")


    #TEST ###############

    d = pandas.read_csv("test.csv")
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    sibSP = d.SibSp.to_numpy()


    X = np.asarray([sex, Pclass, sibSP]).T

    testImages = hw.reshapeAndAppend1s(X)
    testLabels = hw.reshapeLabelVectors(y, 2)
    
    yhat = np.dot(testImages.T, Wtilde)
    # print ("yhat")
    # print (yhat)
    # print ("y")
    # print (y)
    predictions = yhat.argmax(axis=1)

    # Compute predictions on test set
    # ...

    # Write CSV file of the format:
    # PassengerId, Survived
    # ..., ...
    pid = d.PassengerId.to_numpy()
    df = pandas.DataFrame({"PassengerId" : pid, "Survived" : predictions})
    df.to_csv("submission.csv", index=False)