import numpy as np
import pickle


CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_batch(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
        data = dict['data']
        labels = dict['labels']
        data = data.astype(float)
        labels = np.array(labels)
    return data, labels


def load_data():
    data = []
    labels = []
    for i in range(1, 6):
        filename = './dataset/data_batch_' + str(i)
        x, y = load_batch(filename)
        data.append(x)
        labels.append(y)

    X_train = np.concatenate(data)
    Y_train = np.concatenate(labels)
    X_test, Y_test = load_batch('./dataset/test_batch')

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    return {
        'images_train': X_train,
        'labels_train': Y_train,
        'images_test': X_test,
        'labels_test': Y_test,
        'classes': CLASSES
    }

def main():
    load_data()


if __name__ == '__main__':
    main()