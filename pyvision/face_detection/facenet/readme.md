# FaceNet: A Unified Embedding for Face Recognition and Clustering

FaceNet is one of the most popular face recognition architectures.

## Paper
The paper can be read [here](https://arxiv.org/pdf/1503.03832.pdf).

## Summary

- The implementation uses a **Inception-Resnet-v1** architecture to generate the embeddings.

- Currently two models pretrained on *casia-webface* and *vggface2* are made available. These weights were originally made available by David Sandberg's implementation [here](https://github.com/davidsandberg/facenet).

- For usage details check **Usage**, but to summarize, the implementation essentially exposes functions for embedding generation and embedding verification as well i.e. a basic Facial Recognition pipeline has been made available

## Quick Usage

Check demo.

## Usage

- Import facenet.

```
from pyvision.face_detection.facenet import Facenet
```

- Initialize the class. Pretrained for the moment can be casia-webface or vggface2 models.

```
fc = Facenet(pretrianed="casia-webface", saveLoc="save", saveName="det.pkl")
```

- Now we gotta generate embeddings and store the embeddings for comparison. For this we use the **generate_embeddings()** function. There are two ways images can be supplied to this function:

    1. Pass a directory containing images. In that case, the individual image names will be used as image labels

    2. Pass a singular image/path. In this case, a  label has to be passed by the user. This gives the most flexibility and hence is recommended.

    Also, the *save* parameter can be used to specify a custom location for a given embedding that is different than the one specified during model init.

    Also it returns a list of dicts containing labels and their associated embeddings.


```
embeddings = fc.generate_embeddings(...)
```

- Now to run "recognition" on an image, we use the **verify_embeddings()** function. Unline the generate_embeddings() function, this function only accepts singular image or image paths i.e. no directories are allowed.
A few things to note regarding the function:

    1. Embeddings can either be passed directly as a parameter (a list of dicts) or a path to a stored embedding can be passed.
    
    2. The comparison function uses *l2_norm* to calculate distances between embeddings. Other distance calculation metrics like *cosine_similarity* can be added in the future.

    3. The *compare_embeddings()* function needs to be supplied with a label and the function will check whether the given embedding is *similar* to the previously known embeddings associated with the supplied label. 

    4. Return a tuple (True/False, prediction, min_l2_loss)

```
did_match, pred_label, l2_loss = fc.compare_embeddings(...)
```

- For more details look tests.

## Note
While implementing the pretrained models, it was found that often in many cases classifications were not accurate. So it is recommended that care is taken while using facenet.