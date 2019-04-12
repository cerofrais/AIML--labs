import numpy as np
import matplotlib.pyplot as plt

dataset = [[2.7810836, 2.550537003, -1],
           [1.465489372, 2.362125076, -1],
           [3.396561688, 4.400293529, -1],
           [1.38807019, 1.850220317, -1],
           [3.06407232, 3.005305973, -1],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

dataset = np.array(dataset)


plt.figure(figsize=(10, 10))
plt.xlim((0, 10))
plt.ylim((0, 6))
plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])

X = dataset[:, :2]
y = dataset[:, 2]

"""full batch"""
def my_preceptron1(train_data, labels):
    W = []
    eta = 0.0002  # decrease learninng rate
    add1 = np.ones((train_data.shape[0], 1))
    train_data = np.hstack((train_data, add1))
    # w=np.random.rand((train_data.shape[1]))
    w = np.zeros(train_data.shape[1])
    print(w)
    delw = np.zeros((train_data.shape[1]))
    epochs = 10
    for epoch in range(epochs):
        error = 0
        for i in range(train_data.shape[0]):
            wTx = np.dot(train_data[i], w)
            error = error + (labels[i] - wTx)**2
            delw = delw + 2 * (labels[i] - wTx) * (train_data[i].T)
        print(error)
        w = w + eta * delw
        W.append(w)
    return W


W = my_preceptron1(X, y)


dom = np.arange(0, 10, 0.1)
plt.figure(figsize=(10, 10))
plt.xlim((0, 10))
plt.ylim((0, 6))
plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])
for w in W:
    ran = (-w[0] / w[1]) * dom + w[2] / w[1]
    plt.xlim((0, 10))
    plt.ylim((0, 6))
    plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])
    plt.plot(dom, ran)
    plt.pause(0.1)
    plt.clf()
plt.plot(dom, ran)



"""########################################3"""
import sklearn.linear_model as perceptron
cls=perceptron.Perceptron(n_iter=100,eta0=0.02)
cls.fit(X,y)
skw=cls.coef_
skc=cls.intercept_
#skw[0][1]

plt.scatter(dataset[:,0],dataset[:,1],c=dataset[:,2])
ran=(-skw[0][0]/skw[0][1])*dom +skc[0]/skw[0][1]
plt.plot(dom,ran)
plt.show()


