import os
import json
import cv2
import random
import time
from PIL import Image

coco_format_save_path='/data/lhm/UN_DETR/'
yolo_format_classes_path='/data/lhm/UN_DETR/eval/class.txt'
yolo_format_annotation_path='/data/lhm/UN_DETR/pesudo_label/Voc_txt_result'
img_pathDir='/data/datasets/Detection/coco/mixed'

LABLE_DICT = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog":5, "horse": 6, "sheep": 7, "airplane": 8, "bicycle": 9, "boat": 10, "bus": 11, 
              "car": 12, "motorcycle": 13, "train": 14, "bottle": 15, "chair": 16, "dining table": 17, "potted plant": 18, "couch": 19, "tv": 20}

with open(yolo_format_classes_path,'r') as fr:
    lines1=fr.readlines()
categories=[]

for i in LABLE_DICT:
    categories.append({'id':LABLE_DICT[i],'name':i,'supercategory':'None'}) 
categories.append({'id':81,'name':'unknow object','supercategory':'None'}) 

cls_name = lines1


write_json_context=dict()
write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2022, 'contributor': '纯粹ss', 'date_created': '2022-07-8'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
write_json_context['categories']=categories
write_json_context['images']=[]
write_json_context['annotations']=[]

imageFileList=os.listdir(img_pathDir)
for i,imageFile in enumerate(imageFileList):
    imagePath = os.path.join(img_pathDir,imageFile)
    image = Image.open(imagePath)
    W, H = image.size

    img_context={}
    img_context['file_name']=imageFile
    
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2022-07-8'
    img_context['id']= int(imageFile.split('.')[0])
    img_context['license']=1
    img_context['color_url']=''
    img_context['flickr_url']=''
    write_json_context['images'].append(img_context)


    txtFile=imageFile.split('.')[0]+'.txt'
    with open(os.path.join(yolo_format_annotation_path,txtFile),'r') as fr:
        lines=fr.readlines()
    for j,line in enumerate(lines):

        bbox_dict = {}

        class_id,x,y,w,h, Uscore=line.strip().split(' ')
        
        if int(class_id) != 81:
            name = cls_name[int(class_id) - 1].strip()
            for category in categories:
                  if category['name'] == name:
                      class_id = category['id']
        
     
        
        class_id,x, y, w, h, Uscore = int(class_id), float(x), float(y), float(w), float(h), float(Uscore)
   
        
        xmin=(x-w/2)*W
        ymin=(y-h/2)*H
        xmax=(x+w/2)*W
        ymax=(y+h/2)*H
        w=w*W
        h=h*H

        bbox_dict['id']=i*10000+j
        bbox_dict['image_id']= int(imageFile.split('.')[0])
        bbox_dict['category_id']=class_id
        bbox_dict['iscrowd']=0
        height,width=abs(ymax-ymin),abs(xmax-xmin)
        bbox_dict['area']=height*width
        bbox_dict['bbox']=[xmin,ymin,w,h]
        bbox_dict['complete_scores'] = Uscore
        bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
        write_json_context['annotations'].append(bbox_dict)

name = os.path.join(coco_format_save_path,"train"+ '.json')
with open(name,'w') as fw:
    json.dump(write_json_context,fw,indent=2)

