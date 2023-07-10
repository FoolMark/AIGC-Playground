import os
dirs = os.listdir('images')
dirs.sort()
with open('image_list.txt','w') as f:
    for dir in dirs:
        f.write(dir+' 0\n')