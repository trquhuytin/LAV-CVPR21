import json
import os
import shutil
import numpy as np

# fileObject = open("D:/Drhuy/LAV/encode.json", "r")
# jsonContent = fileObject.read()
# encode = json.loads(jsonContent)
# fileObject.close()

# for f in encode['test'].keys():
#     os.makedirs(os.path.join("D:/Drhuy/LAV/data/H20/val", encode['test'][f]), exist_ok=True)
#     pth = os.path.join("D:/Drhuy/h2o_CASA",f,"cam4/rgb256")
#     for files in os.listdir(pth):
#         shutil.copy(
#             src=os.path.join(pth, files),
#             dst=os.path.join("D:/Drhuy/LAV/data/H20/val", encode['test'][f])
#         )


# for f in encode['test'].keys():
#     pth = os.path.join("D:/Drhuy/h2o_CASA",f,"cam4/action_label")
#     np_labs=[]
#     for files in os.listdir(pth):
#         with open(os.path.join(pth,files), 'r') as k:
#             np_labs.append(int(k.read()))
    
#     labels = np.array(np_labs)
#     np.save(os.path.join("D:/Drhuy/LAV/data/H20/labels/val/videos", encode['test'][f]), labels)

unique_labels = list()
vids = 0
for label in os.listdir(r'D:\Drhuy\LAV\data\H20\labels\train\videos'):
    l = np.load(os.path.join(r'D:\Drhuy\LAV\data\H20\labels\train\videos', label))
    unique_labels.extend(np.unique(l))
    vids += 1

for label in os.listdir(r'D:\Drhuy\LAV\data\H20\labels\val\videos'):
    l = np.load(os.path.join(r'D:\Drhuy\LAV\data\H20\labels\val\videos', label))
    unique_labels.extend(np.unique(l))
    vids += 1

print(np.unique(unique_labels), vids)
# print(set((unique_labels)))