### Install

`pip install pip install --upgrade -r requirements.txt`

### Python version
Tested with 3.5.2

### Run using existing [Inception-v3](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) model

`python classify_image.py --image_file=./sample_images/panda.jpg`

`python classify_image.py --image_file=./sample_images/dog.jpg`

`python classify_image.py --image_file=./sample_images/cat.jpg`

### Build and run your own NN model

`python nn_run.py` to build and save model.

`python nn_load_model_and_run.py --image_file=./sample_images/dog.jpg` to restore and run model against given image.

### Output

```
giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.89107)
indri, indris, Indri indri, Indri brevicaudatus (score = 0.00779)
lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00296)
custard apple (score = 0.00147)
earthstar (score = 0.00117)
```

### Disclaimer

This version of `classify_image.py` was modified by adding requirements and embedding model related data to project.
