import torch
import torch.nn as nn
import torch.nn.functional as F  

import numpy as np 
import cv2  
from PIL import Image
import os  
import pickle as pkl
import time

from .models.InceptionResnetV1 import InceptionResnetV1
from pyvision.misc.mtcnn import MTCNN
from .utils.extract_face import extract_face


__models__ = ["InceptionResnetV1"]

__pretrained__ = ["casia-webface", "vggface2"]

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))


class Facenet:
    """ The principle Facenet class. Exposes methods to run inference and
    generate Embeddings. It saves generated embeddings in a given location
    if required. It also exposes methods to compare an embedding to previously 
    stored embeddings and return whether or not the given embedding belongs to a
    known face or not.

    Note: Currently it only supports Inception-Resnet-V1 as the backbone but 
          other models will soon be added for supported
    """
    def __init__(self, pretrained="casia-webface", model="InceptionResnetV1", device="cpu", face_conf=0.7,
        saveLoc=None, saveName="embeddings.pkl"):

        if model not in __models__:
            raise NotImplementedError("{} models implemented. Got {}".format(__models__, model))
        if pretrained not in __pretrained__:
            raise NotImplementedError("{} pretrained models implemented. Got {}".format(__pretrained__, pretrained))
        if device is not "cpu" and not torch.cuda.is_available():
            raise ValueError("cuda not available but got device=",device)
        if saveLoc is not None and saveName is None:
            raise ValueError("saveName cannot be None when saveLoc is specified")
        if saveLoc is not None and not os.path.exists(saveLoc):
            os.mkdir(saveLoc)
        

        self.model_name = model
        self.device = "cuda" if device is not "cpu" else "cpu"
        self.face_conf = face_conf
        self.saveLoc = saveLoc
        self.saveName = saveName

     
        # initializing the mtcnn module
        self.mtcnn_module = MTCNN(conf_thresh=[self.face_conf]*3) 

        # initializing the Inception-Resnet-V1
        self.model = InceptionResnetV1(pretrained=pretrained, wtspath=__PREFIX__+"/weights/", classify=False, device=self.device)
        self.model = self.model.eval().to(self.device)

    def generate_embeddings(self, img, path=None, label=None, save=None):
        
        """The function that generates the embeddings of a given image(s).
        It provides the choice of saving the embedding(s) along with 
        labels in a pickle format in a given location. It also returns the
        generated embeddings in a dictionary for furthur downstream use.

        Note: The end-to-end pipeline provided here only works for single-face
              images. For more details regarding usage look at Facenet Docs

        Arguments:
        -> img: a numpy.ndarray or a PIL Image
        -> path: path to a single image or a directory containing images
        -> label: Positive label of the given image. If 'None' and path of 
                  image is specified or a directory of images is specified,
                  then positive labels are inferred from the image name.
        -> save: Location to save the embeddings in a pickle format

        Returns:
        -> A dictionary of embeddings and their labels
        """
        
        if save is None and self.saveLoc is not None:
            self.saveLoc = os.path.join(self.saveLoc, self.saveName)
        elif save is not None:
            if not os.path.exists(os.path.dirname(save)):
                os.mkdir(os.path.dirname(save))
            self.saveLoc = save
        
        imgs_list = []
        if img is not None and path is not None:
            raise ValueError("img and path cannot be both not None")
        if img is not None:
            if isinstance(img, np.ndarray):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img).convert('RGB')
            elif isinstance(img, Image.Image):
                pass
            else:
                raise TypeError("img can be only numpy.ndarray or PIL.Image.Image. Got {}. use 'path' if wish to pass a path".format(type(img)))
            if label is None:
                raise ValueError("label cannot be None")
            #imgs_list = [label]
            label = [label]
        else:                  
            # assert img is None: "img should be None if path specified"
            img = None

            if os.path.isfile(path):
                
                img = Image.open(path).convert("RGB")
                
                if label is None:
                    raise ValueError("label cannot be None when single image specified")
                #imgs_list = [label]
                label = [label]
            elif os.path.isdir(path):
                
                imgs_list = os.listdir(path)
                if len(imgs_list) == 0:
                    raise Exception("EmptyDirectoryError: {} is an empty directory".format(path))
                
                # when directory specified, img names act as labels
                label = [x.split(".")[0] for x in imgs_list]
                print("Found labels: ", label)

        aligned_face_tensors = []
        # first we run image(s) through mtcnn to get the face tensors
        if img is not None:
            face_tensors = extract_face(self.mtcnn_module, img)
            #print(face_tensors.shape)
            face_tensors = face_tensors.transpose(2, 3).transpose(1, 2)
            face_tensors = face_tensors.squeeze(0)
            aligned_face_tensors.append(face_tensors)
            #print(face_tensors.shape)
        else:
            for img_name in imgs_list:
                img = Image.open(os.path.join(path, img_name)).convert('RGB')
                face_tensors = extract_face(self.mtcnn_module, img)
                face_tensors = face_tensors.transpose(2, 3).transpose(1, 2)
                face_tensors = face_tensors.squeeze(0)
                aligned_face_tensors.append(face_tensors)
                tmp = face_tensors
                #print(face_tensors.shape)
        
        aligned_face_tensors = torch.stack(aligned_face_tensors)
        
        aligned_face_tensors = aligned_face_tensors.to(self.device)
        start_time = time.time()
        embeddings = self.model(aligned_face_tensors)
        running_time = time.time() - start_time

        embeddings_obj = []

        embeddings_obj = [{"name": l, "embedding": e.detach().cpu()} for l, e in zip(label, embeddings)]
        

        if self.saveLoc is not None:
            if not os.path.exists(self.saveLoc):
                open(self.saveLoc, "a").close()
            with open(self.saveLoc, "wb") as fp:
                pkl.dump(embeddings_obj, fp)

        return embeddings_obj
    
    def compare_embeddings(self, embeddings, img, label, thresh=0.5, embedLoc=None):
        """The function takes in an image, generates its embeddings and compares the 
        generated embeddings with the embeddings passed as a list of dicts or read from
        a pickle file. 

        Note: Unlike generate_embeddings(), compare_embeddings() dont support directory 
              paths supplying a directory of images. As in, a path to a single image needs
              to be provided.

        Arguments: 
        -> thresh: minimum similarity allowed below which the image is classified as unknown
        -> embeddings: List of dict of embeddings. Should be None if path to embeddings
                       provided. 
        -> img: A PIL.Image or numpy.ndarray image that needs to verified. Image paths are
                also supported.
        -> embedLoc: Path to embeddings stored as a pickle file. Should be None if embeddings 
                     are being directly passed.

        -> label: to check whether the supplied image conforms to the label provided by this 
                  keyword
        """
        
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            pass
        else:
            raise TypeError("img can be PIL.Image or numpy.ndarray or path(str). Got img type: ", type(img))
        
        if embeddings is not None and embedLoc is not None:
            raise ValueError("embeddings and embedLoc cannot both have values")
        elif embeddings is None and embedLoc is not None:
            with open(embedLoc, "rb") as fp:
                embeddings = pkl.load(fp)
        
        known_embeds = []
        known_labels = []
        for e in embeddings:
            known_embeds.append(e["embedding"])
            known_labels.append(e["name"])
        
        known_embeds_stack = torch.stack(known_embeds)
        
        
        # now we run the given image through the MTCNN, crop and align the faces
        # convert to tensors and run it through the model to generate facial 
        # embeddings
        face_tensors = extract_face(self.mtcnn_module, img)
        face_tensors = face_tensors.transpose(2, 3).transpose(1, 2)
        face_tensors = face_tensors.squeeze(0)

        face_tensors = torch.stack([face_tensors])
    
        face_tensors = face_tensors.to(self.device)
        img_embeddings = self.model(face_tensors).detach().cpu()
        
        """
        cosine_simi = nn.CosineSimilarity(dim=0)
        similarities = cosine_simi(known_embeds_stack.T, img_embeddings.T)

        max_similarity = similarities.max().item()
        max_idx = (similarities == max_similarity).nonzero().flatten().item()       
        """
        
        l2_loss = (known_embeds_stack.T - img_embeddings.T).norm(dim=0)

        min_loss = l2_loss.min().item()
        min_idx = torch.nonzero(l2_loss == min_loss, as_tuple=False).flatten().item()

        if min_loss > .8:
            predicted = "Unknown"
        else:
            if min_loss > thresh:
                predicted = None
            else:
                predicted = known_labels[min_idx]
        
        if label == predicted:
            return True, predicted, min_loss
        else:
            return False, predicted, min_loss


        

