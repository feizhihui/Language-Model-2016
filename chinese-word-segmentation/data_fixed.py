# encoding=utf-8
import os


def get_filelist(filepath):
    pathDir = os.listdir(filepath)
    print(pathDir)
    final_doc = []
    for filename in pathDir:
        with open(filepath + filename, mode='r') as file:
            newdoc = file.readlines()
            final_doc.extend(newdoc[2:-1])
    print(len(final_doc))
    with open('ctb.txt', mode='w') as file:
        for line in final_doc:
            file.write(line)



if __name__ == '__main__':
    get_filelist('F:/Datas/ctb/')
