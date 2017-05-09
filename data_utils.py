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

def get_image_pixels(image_path, IMAGE_PIXELS):
    import PIL
    from PIL import Image
    im = Image.open(image_path)
    im.thumbnail((32, 32), PIL.Image.ANTIALIAS)
    im = (np.array(im))
    r = im[:, :, 0].flatten()
    g = im[:, :, 1].flatten()
    b = im[:, :, 2].flatten()
    return np.array(list(r) + list(g) + list(b)).reshape((1, IMAGE_PIXELS))

def gen_batch(data, batch_size, num_iter):
  data = np.array(data)
  index = len(data)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data)):
      index = 0
      shuffled_indices = np.random.permutation(np.arange(len(data)))
      data = data[shuffled_indices]
    yield data[index:index + batch_size]

def main():
    load_data()


if __name__ == '__main__':
    main()