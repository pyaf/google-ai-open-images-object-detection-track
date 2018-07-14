import os
from subprocess import call

files = ['000002b66c9c498e.jpg', '000002b97e5471a0.jpg', '000002c707c9895e.jpg', '0000048549557964.jpg', '000004f4400f6ec5.jpg', '0000071d71a0a6f6.jpg', '000013ba71c12506.jpg', '000018acd19b4ad3.jpg', '00001bc2c4027449.jpg', '00001bcc92282a38.jpg', '0000201cd362f303.jpg', '000020780ccee28d.jpg', '000023aa04ab09ed.jpg', '0000253ea4ecbf19.jpg', '000025ea48cab6fc.jpg', '0000271195f2c007.jpg', '0000286a5c6a3eb5.jpg', '00002b368e91b947.jpg', '00002f4ff380c64c.jpg', '0000313e5dccf13b.jpg', '000032046c3f8371.jpg', '00003223e04e2e66.jpg', '0000333f08ced1cd.jpg']

for file in files:
    if not os.path.exists('train/' + file + '.jpg'):
        spath = "gs://open-images-dataset/train/%s " % file
        call(["gsutil", "cp", spath, 'train/'])
        print(file, 'done', 'count:')
    else:
        print(file, 'already downloaded')
