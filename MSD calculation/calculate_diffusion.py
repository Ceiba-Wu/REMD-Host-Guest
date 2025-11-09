import subprocess
import os


folder_list = ['1-butanol', '1-propanol', 'aspirin', 'methyl_butyrate', 't-butanol']
box_list = [40,45,50]
root_dir = os.getcwd()
for folder in folder_list:
    for box_size in box_list:
        for i in range(4,11):
            filepath = os.path.join('', folder, 'box_'+str(box_size), 'md-'+str(i))
            os.chdir(filepath)
            if os.path.exists('msd.dat'):
                os.chdir(root_dir) 
                continue
            subprocess.run('sh /home/databank/zjhan/diffusion/calculation/calculate_diffusion2.sh', shell=True)
            os.chdir(root_dir)
            
            