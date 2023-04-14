import os
import glob
dataset='cape_raw'
subject='00096'

subject_path= f'/home/yu/AMirror/ICON/data/{dataset}/{subject}'
# os.makedirs(fit_root)
def gen_all():
    for seq in os.listdir(os.path.join(subject_path, 'scans_ply')):
        # os.makedirs(os.path.join(fit_root,cm))
        with open(f'{subject_path}/{seq}.txt', 'w') as f:
            for file in sorted(os.listdir(f'{subject_path}/scans_ply/{seq}')):
                frame=file.split('.')[1]
                f.write(f'{dataset}/{subject}/{seq}/{frame}\n')
            
# generate new txt file with content sampled from the original txt file by the given interval
def gen_sampled_txt(txt_path,txt2_path, interval,offset=0):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    with open(txt2_path, 'w') as f:
        for i in range(0, len(lines), interval):
            if i+offset<len(lines):
                f.write(lines[i+offset])

if __name__=='__main__':
    gen_sampled_txt(f'{subject_path}/shortlong_hips.txt',f'{subject_path}/val.txt', 3,1)
    # gen_sampled_txt(f'{subject_path}/val.txt',f'{subject_path}/val_2.txt', 2)
    # gen_sampled_txt(f'{subject_path}/test.txt',f'{subject_path}/test_2.txt', 2)