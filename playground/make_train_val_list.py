import os
import sys
sys.path.append("..")
from utils import utils
import numpy as np

# Full NameLists
# Train Lists
# Val Lists

def check_existence(query,key):
    for query_item in query:
        if query_item in key:
            return True

    else:
        return False
    


if __name__=="__main__":
    
    datapath = "/data1/liu/OrientationLearning"
    
    lines = utils.read_text_lines("../filenames/val_list.txt")
    valid_lines_cars = []
    for idx, line in enumerate(lines):
        image_filename = line
        annotation = line.replace(".jpg",".txt")
        
        annotation_path = os.path.join(datapath,annotation)
        
        annotations = np.loadtxt(annotation_path,dtype=str).reshape(-1,5)
        classes = annotations[:,0].astype(np.int)
        exitence = check_existence([0,1,2],classes)
        
        if exitence:
            valid_lines_cars.append(line)
        else:
            pass
    
    
    with open("val_list_cars.txt",'w') as f:
        for idx, line in  enumerate(valid_lines_cars):
            if idx!=len(valid_lines_cars)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    