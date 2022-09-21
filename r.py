import os
from PIL import Image
 
def listallfiles(root):
    listfilenames = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name != 'Thumbs.db':
                listfilenames.append(os.path.join(path, name))
    return listfilenames

print('started')

source_path = r"C:\Users\laharika\Desktop\Downloads\kiran goat photos\goat 54"
destination_path = r'C:\Users\laharika\Desktop\Downloads\kiran goat photos\goat 54 - Copy'

source_list = listallfiles(source_path)
print(len(source_list),'source files reading completed')
dest_list = listallfiles(destination_path)
print(len(dest_list),'destination files reading completed')

foundpath = source_path +'\\' + 'found\\'
misspath = source_path +'\\' + 'missing\\'


if os.path.exists(foundpath) == False:
    os.mkdir(foundpath)

if os.path.exists(misspath) == False:
    os.mkdir(misspath)


for source in source_list:
    s = Image.open(source)
    condi = True
    for dest in dest_list:
        d = Image.open(dest)
        if s == d:
            s.save(foundpath + source.split('\\')[-1])
            dest_list.remove(dest)
            condi = False
            continue
    if condi:
        s.save(misspath + source.split('\\')[-1])
