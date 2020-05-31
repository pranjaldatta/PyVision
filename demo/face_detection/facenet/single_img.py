from pyvision.face_detection.facenet import Facenet

# In this example we take a single image from the ./imgs folder
# Generate embeddings and store them. Then use those embeddings to 
# check whether a previously unseen image is classified accurately or not


# First we instantiate the facenet object. saveLoc is the path to the
# folder wherein the embeddings will be saved. By default it will be saved
# as "embeddings.pkl" but can be changed with the "saveName" param
fc = Facenet(saveLoc="save/") 

# generate embeds
_ = fc.generate_embeddings(img=None, path="demo/face_detection/facenet/imgs/BarackObama.jpeg", label="Barack Obama")

# now we compare it against a "False" image 
did_match, pred, loss = fc.compare_embeddings(None, img="demo/face_detection/facenet/imgs/ManojBajpayee.jpeg", label="Barack Obama", embedLoc="save/embeddings.pkl")
print(did_match, pred, loss)
print("Comparing against 'False' image, we get: ", did_match)