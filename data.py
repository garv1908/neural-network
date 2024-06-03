from numpy import load

data = load('data/mnist.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])