import os
import xml.dom.minidom
import xml.etree.ElementTree
 
xmldir = '/home/lty/mmdetection-master/data/fusai_train/VOCdevkit/VOC2007/Annotations/' #你的xml文件的路經，注意最后一定要有'/'
 
for xmlfile in os.listdir(xmldir):
    xmlname = os.path.splitext(xmlfile)[0]
 
    #读取 xml 文件
    dom = xml.dom.minidom.parse(os.path.join(xmldir,xmlfile))
    root = dom.documentElement
 
    #获取标签对的名字，并为其赋一个新值
    root.getElementsByTagName('folder')[0].firstChild.data = 'down'
    root.getElementsByTagName('filename')[0].firstChild.data = xmlname + '.jpg'
    root.getElementsByTagName('path')[0].firstChild.data = './savePicture/' + xmlname + '.jpg'
       
    #修改并保存文件
    xml_specific = xmldir + xmlfile 
    with open(xml_specific,'w') as fh:
        dom.writexml(fh)

