import os
import glob
mesh_root= '/home/yu/AMirror/ICON/data/cape_raw/00096'
fit_root='/home/yu/AMirror/ICON/data/cape_raw/00096_fit'
# os.makedirs(fit_root)

for cm in os.listdir(mesh_root):
    # os.makedirs(os.path.join(fit_root,cm))
    
    with open(f'{mesh_root}/{cm}.txt', 'w') as f:
        for file in os.listdir(os.path.join(mesh_root, cm)):
            f.write(file+'\n')
            
    # with open('/home/yu/AMirror/ICON/data/cape_raw/00096.txt','w') as f:
    #     for file_path in file_paths:
    #         file_path=os.path.basename(file_path)
    #         f.write(file_path+'\n')