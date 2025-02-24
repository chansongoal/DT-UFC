import os

img_path = '/gdata/gaocs/dataset/VOC2012/JPEGImages'
img_names = sorted(os.listdir(img_path))

img_names = img_names[:10000]
for img_name in img_names:
    if os.path.isfile(os.path.join(img_path, img_name)): 
        name, ext = os.path.splitext(img_name)
        print(name) 

