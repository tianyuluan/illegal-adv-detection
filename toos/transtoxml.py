import os
import json
import cv2
import numpy as np
from xml.etree.ElementTree import Element,ElementTree


def indent( elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem

# acc = TP+TN/all
def prepare_data(testpath,jsonpath, annpath,thresh = -1):
    for rootdir,dir,files in os.walk(jsonpath):
        files = [f.split('.')[0]+'.jpg' for f in files if f.endswith('.json')]
        for imgname in files:
            #print(imgname)
            path = os.path.join(testpath, imgname)
            root = Element('annotation')
            treeroot = ElementTree(root)

            child0 = Element('folder')
            child0.text = 'down'
            root.append(child0)
            child0 = Element('filename')
            child0.text = imgname
            root.append(child0)
            child0 = Element('path')
            child0.text = './savePicture/' + imgname
            root.append(child0)

            child0 = Element('source')
            child00 =Element('database')
            child00.text = 'Unknown'
            child0.append(child00)
            root.append(child0)

            child0 = Element('size')
            img = cv2.imread(path)
            child_w = Element('width')
            child_w.text = str(np.shape(img)[1])
            child_h = Element('height')
            child_h.text = str(np.shape(img)[0])
            child_d = Element('depth')
            child_d.text = str(np.shape(img)[2])
            child0.append(child_w)
            child0.append(child_h)
            child0.append(child_d)
            root.append(child0)
            child0 = Element('segmented')
            child0.text = str(0)
            root.append(child0)

            with open(os.path.join(jsonpath,imgname.split('.')[0]+'.json'),'r') as f:
                annots = json.load(f)
                for a in annots:
                    # if thresh != -1 and thresh > a['score']:
                    #     continue
                    obj = Element('object')
                    child0 = Element('name')
                    child0.text = a['category']
                    obj.append(child0)

                    childp = Element('pose')
                    childp.text = 'Unspecified'
                    obj.append(childp)

                    childt = Element('truncated')
                    childt.text = str(0)
                    obj.append(childt)
                    childd = Element('difficult')
                    childd.text = str(0)
                    obj.append(childd)
                    child0 = Element('bndbox')
                    xmin = Element('xmin')
                    xmin.text=str(a['x']-a['w']/2)
                    ymin = Element('ymin')
                    ymin.text = str(a['y'] - a['h'] / 2)

                    xmax = Element('xmax')
                    xmax.text = str(a['x'] + a['w'] / 2)

                    ymax = Element('ymax')
                    ymax.text = str(a['y'] + a['h'] / 2)

                    child0.append(xmin)
                    child0.append(xmax)
                    child0.append(ymin)
                    child0.append(ymax)
                    obj.append(child0)

                    root.append(obj)
            indent(root)
            #fp = open(os.path.join(annpath,imgname.split('.')[0]+'.xml'), 'w')
            treeroot.write(os.path.join(annpath,imgname.split('.')[0]+'.xml'),encoding="utf-8",xml_declaration=True)
            #treeroot.write(os.path.join(annpath,imgname.split('.')[0]+'.xml'), 'UTF-8')



if __name__ == '__main__':
    jsonpath = '/home/lty/mmdetection_lower/data/hualubei_gaunggaopai/测试集/result'  # '/home/tan/dataset/comptition/SEED/test_acc/result_test'
    testpath = '/home/lty/mmdetection_lower/data/hualubei_gaunggaopai/测试集/weibiao'
    annpath = '/home/lty/mmdetection_lower/data/hualubei_gaunggaopai/测试集/temp_xml_1088/'

    if not os.path.exists(annpath):
        os.mkdir(annpath)
    prepare_data(testpath,jsonpath, annpath,thresh = 0)
