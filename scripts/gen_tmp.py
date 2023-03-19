import os
import glob
dataset='cape_raw'
subject='00096'

subject_path= f'/home/yu/AMirror/ICON/data/{dataset}/{subject}'
# os.makedirs(fit_root)

for seq in os.listdir(os.path.join(subject_path, 'scans_ply')):
    # os.makedirs(os.path.join(fit_root,cm))
    with open(f'{subject_path}/{seq}.txt', 'w') as f:
        for file in sorted(os.listdir(f'{subject_path}/scans_ply/{seq}')):
            frame=file.split('.')[1]
            f.write(f'{dataset}/{subject}/{seq}/{frame}\n')
            
    # with open('/home/yu/AMirror/ICON/data/cape_raw/00096.txt','w') as f:
    #     for file_path in file_paths:
    #         file_path=os.path.basename(file_path)
    #         f.write(file_path+'\n')