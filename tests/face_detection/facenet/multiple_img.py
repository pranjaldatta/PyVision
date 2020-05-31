from pyvision.face_detection.facenet import Facenet

# In this example, we take all the imgs from the ./imgs folder and 
# generate embeddings for them. We also associate each embedding with their
# filename which act as 'true labels'. Then we use these embeddings to 'classify'
# whether a supplied image belongs to any one of given categories

# First we instantiate the facenet object. saveLoc is the path to the
# folder wherein the embeddings will be saved. By default it will be saved
# as "embeddings.pkl" but can be changed with the "saveName" param
fc = Facenet(saveLoc="save", saveName="embeddings2.pkl")

embeddings = fc.generate_embeddings(img=None, path="demo/face_detection/facenet/imgs")

did_match, preds, loss = fc.compare_embeddings(
    img="demo/face_detection/facenet/zucktest.jpeg",
    embedLoc="save/embeddings2.pkl",
    embeddings=None,
    label="MarkZuckerberg"
)
print(did_match, preds, loss)
print("For 'True' Image, we get: ", did_match)

