import json
import os


def read_json(json_file):
    with open(json_file,"r") as f:
        act_cfg = json.load(f)
    return act_cfg

def get_all_files(root_dir,suffixs):
    files = []
    for root,dirs,names in os.walk(root_dir):
        for filename in names:
            if filename.endswith(suffixs):
                files.append(os.path.join(root,filename))
    files = sorted(files,  key=lambda x: os.path.getmtime(x))
    return files

def get_all_jsons(root_dir):
    img_suffix = ".json"
    ret_list = get_all_files(root_dir,img_suffix)   
    return ret_list

def get_all_images(root_dir):
    img_suffix = ('.png', '.jpg', '.jpeg','.bmp')
    ret_list = get_all_files(root_dir,img_suffix)
    return ret_list


if __name__ == "__main__":
    imgs = get_all_images("train_data")
    print(imgs)
        
