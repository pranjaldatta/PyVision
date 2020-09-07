from pyvision.segmentation import fcn
from glob import glob

fcn_model = fcn.FCN(device="cpu", show=False)

for idx, item in enumerate(glob("pyvision/segmentation/fcn/examples/*.jpg")):
    print(f"#### Image #{idx+1} ####")
    preds, seg_map, blend_map = fcn_model.inference(item, save=item.split(".")[0]+"_50")
    print("Prediction matrix shape: ", preds.shape)
    print("Segmentation Map shape: ", seg_map.size)
    print("Blend Map shape: ", blend_map.size)