from mmdet.apis import init_detector, inference_detector
import os, zipfile
import shutil
CONFIG_FILE = '/home/lty/mmdetection-master/submit/22_4000_/cascade_rcnn_x101_64x4d_fpn_1x_coco.py'
CHECKPOINT_PATH = '/home/lty/mmdetection-master/submit/22_4000_/epoch_12.pth'
model = init_detector(CONFIG_FILE, CHECKPOINT_PATH)
path='/home/lty/mmdetection-master/data/test/'    #测试的照片路径
# path='/home/lty/dataset/comp/test/'    #测试的照片路径
key={0:"window_shielding",1:"multi_signs",2:"non_traffic_sign"}

score_thresh = 0.1
del_shuiyin = 0


def make_zip(source_dir, output_filename):
  zipf = zipfile.ZipFile(output_filename, 'w')
  pre_len = len(os.path.dirname(source_dir))
  for parent, dirnames, filenames in os.walk(source_dir):
    for filename in filenames:
      pathfile = os.path.join(parent, filename)
      arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
      zipf.write(pathfile, arcname)
  zipf.close()

import json
import os
import numpy as np
dir=os.listdir(path)
n = 0
for ii in range(len(dir)):

  # print(ii)
  img=path+dir[ii]
  name=dir[ii].split(".")[0]
  detections = inference_detector(model, img)
  # print(detections)#detections里面存储的xyxy score信息。
  data=[]
  for i in range(3):

    cate=key[i]
    x1=detections[i][:,0]
    y1=detections[i][:,1]
    x2=detections[i][:,2]
    y2=detections[i][:,3]
    w1=x2-x1
    h1=y2-y1
    scores1=detections[i][:,4]

    for j in range(len(detections[i])):
      x = int(x1[j] + w1[j] / 2.0)
      y = int(y1[j] + h1[j] / 2.0)
      w=int(w1[j])
      h=int(h1[j])
      scores2=str(scores1[j])
      if float(scores2) >= score_thresh:
        value={"category":cate,"x":x,"y":y,"w":w,"h":h,"score":float(scores2[:7])}
        data.append(value)
    path1 ='/home/lty/mmdetection-master/submit/22_4000_/result/'+name+'.json'
    # path1 ='/home/lty/下载/result/'+name+'.json'
    # path1 ='/home/lty/dataset/comp/result1/'+name+'.json'
    with open(path1, 'w')  as f:
      json.dump(data, f, indent=2)
  n += 1
  print(n)
# zip output files
# make_zip('/data/result/','result.zip')
# shutil.rmtree('/data/result/result')
# # os.mkdir('../data')
# shutil.move('result.zip','/data/result/result.zip')