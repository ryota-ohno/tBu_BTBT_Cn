##pbepbe+d3bjで計算
import os
import numpy as np
import pandas as pd
import subprocess
from utils import Rod, R2atom

############################汎用関数###########################
def get_monomer_xyzR(monomer_name,Ta,Tb,Tc,A2,A3,phi1,phi2):
    T_vec = np.array([Ta,Tb,Tc])
    df_mono=pd.read_csv(f'/home/ohno/Working/amber_sc_opt/tBu_BTBT_Cn/monomer/{monomer_name}.csv')
    atoms_array_xyzR=df_mono[['X','Y','Z','R']].values
    xyz_array = atoms_array_xyzR[:,:3];R_array = atoms_array_xyzR[:,3].reshape((-1,1))

    ex = np.array([1.,0.,0.]); ez = np.array([0.,0.,1.])
    xyz_array = np.matmul(xyz_array,Rod(-ex,A2).T)#
    xyz_array = np.matmul(xyz_array,Rod(ez,A3).T)#
    xyz_array = xyz_array + T_vec
    
    C0_index = 5;C1_index = 22;C2_index = 13;C3_index = 50####
    C0=xyz_array[C0_index];C1=xyz_array[C1_index];C2=xyz_array[C2_index];C3=xyz_array[C3_index]
    n1=C1-C0;n1/=np.linalg.norm(n1)
    n2=C3-C2;n2/=np.linalg.norm(n2)

    xyz_array[C1_index:C3_index] = np.matmul((xyz_array[C1_index:C3_index]-C0),Rod(n2,phi2).T) + C0
    xyz_array[C3_index:] = np.matmul((xyz_array[C3_index:]-C2),Rod(n1,phi1).T) + C2
    return np.concatenate([xyz_array,R_array],axis=1)
        
def get_xyzR_lines(xyzR_array,file_description):
    lines = [     
        '%mem=40GB\n',
        '%nproc=40\n',
        '#P TEST pbepbe/6-311G** EmpiricalDispersion=GD3BJ counterpoise=2\n',
        '\n',
        file_description+'\n',
        '\n',
        '0 1 0 1 0 1\n'
    ]
    mol_len = len(xyzR_array)//2
    atom_index = 0
    mol_index = 0
    for x,y,z,R in xyzR_array:
        atom = R2atom(R)
        mol_index = atom_index//mol_len + 1
        line = '{}(Fragment={}) {} {} {}\n'.format(atom,mol_index,x,y,z)     
        lines.append(line)
        atom_index += 1
    return lines

# 実行ファイル作成
def get_one_exe(file_name,machine_type):
    file_basename = os.path.splitext(file_name)[0]
    if machine_type=='1':
        group=1;core=40
    elif machine_type=='2':
        group=2;core=52    
    cc_list=[
        '#!/bin/sh \n',
         '#$ -S /bin/sh \n',
         '#$ -cwd \n',
         '#$ -V \n',
         f'#$ -q gr{group}.q \n',
         f'#$ -pe OpenMP {core} \n',
         '\n',
         'hostname \n',
         '\n',
         'export g16root=/home/g03 \n',
         'source $g16root/g16/bsd/g16.profile \n',
         '\n',
         'export GAUSS_SCRDIR=/home/scr/$JOB_ID \n',
         'mkdir /home/scr/$JOB_ID \n',
         '\n',
         'g16 < {}.inp > {}.log \n'.format(file_basename,file_basename),
         '\n',
         'rm -rf /home/scr/$JOB_ID \n',
         '\n',
         '\n',
         '#sleep 5 \n']
#          '#sleep 500 \n'

    return cc_list

######################################## 特化関数 ########################################

##################gaussview##################
def make_gaussview_xyz(auto_dir,monomer_name,params_dict,isInterlayer=False):
    a_ = params_dict['a']; b_ = params_dict['b']
    z = params_dict['z']; A2 = params_dict['A2']; A3 = params_dict['theta']
    phi1 = params_dict.get('phi1',0.0); phi2 = params_dict.get('phi2',0.0)
    
    a =np.array([a_,0,0])
    b =np.array([0,b_,0])
    
    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A2,A3, phi1,phi2)
    if a_>b_:
        monomer_array_p1 = get_monomer_xyzR(monomer_name,0,b_,2*z,A2,A3, phi1,phi2)
        monomer_array_p2 = get_monomer_xyzR(monomer_name,0,-b_,-2*z,A2,A3, phi1,phi2)
    else:
        monomer_array_p1 = get_monomer_xyzR(monomer_name,a_,0,0,A2,A3, phi1,phi2)
        monomer_array_p2 = get_monomer_xyzR(monomer_name,-a_,0,0,A2,A3, phi1,phi2)
    
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a_/2,b_/2,z,A2,-A3,-phi1,-phi2)
    monomer_array_t2 = get_monomer_xyzR(monomer_name,a_/2,-b_/2,z,A2,-A3,-phi1,-phi2)
    monomer_array_t3 = get_monomer_xyzR(monomer_name,-a_/2,-b_/2,z,A2,-A3,-phi1,-phi2)
    monomer_array_t4 = get_monomer_xyzR(monomer_name,-a_/2,b_/2,z,A2,-A3,-phi1,-phi2)

    monomers_array = np.concatenate([monomer_array_i,monomer_array_p1,monomer_array_t1,monomer_array_t2,monomer_array_p2,monomer_array_t3,monomer_array_t4],axis=0)
    
    file_description = 'z={}_A2={}_A3={}'.format(np.round(z,1),round(A2),round(A3))
    lines = get_xyzR_lines(monomers_array,file_description)
    lines.append('Tv {} {} {}\n'.format(a[0],a[1],a[2]))
    lines.append('Tv {} {} {}\n'.format(b[0],b[1],b[2]))
    #lines.append('Tv {} {} {}\n\n\n'.format(c[0],c[1],c[2]))
    
    os.makedirs(os.path.join(auto_dir,'gaussview'),exist_ok=True)
    output_path = os.path.join(
        auto_dir,
        'gaussview/{}_z={}_A2={}_A3={}_a={}_b={}.gjf'.format(monomer_name,round(z),round(A2),round(A3),np.round(a_,2),np.round(b_,2))
    )
            
    with open(output_path,'w') as f:
        f.writelines(lines)

def make_gjf_xyz(auto_dir,monomer_name,params_dict,isInterlayer):
    a_ = params_dict['a']; b_ = params_dict['b']
    z = params_dict['z']; A2 = params_dict['A2']; A3 = params_dict['theta']
    phi1 = params_dict.get('phi1',0.0); phi2 = params_dict.get('phi2',0.0)
    
    a =np.array([a_,0,0])
    b =np.array([0,b_,0])
    
    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A2,A3, phi1,phi2)
    if a_>b_:
        monomer_array_p1 = get_monomer_xyzR(monomer_name,0,b_,2*z,A2,A3, phi1,phi2)
        monomer_array_p2 = get_monomer_xyzR(monomer_name,0,-b_,-2*z,A2,A3, phi1,phi2)
    else:
        monomer_array_p1 = get_monomer_xyzR(monomer_name,a_,0,0,A2,A3, phi1,phi2)
        monomer_array_p2 = get_monomer_xyzR(monomer_name,-a_,0,0,A2,A3, phi1,phi2)
    
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a_/2,b_/2,z,A2,-A3,-phi1,-phi2)
    monomer_array_t2 = get_monomer_xyzR(monomer_name,a_/2,-b_/2,z,A2,-A3,-phi1,-phi2)
    monomer_array_t3 = get_monomer_xyzR(monomer_name,-a_/2,-b_/2,z,A2,-A3,-phi1,-phi2)
    monomer_array_t4 = get_monomer_xyzR(monomer_name,-a_/2,b_/2,z,A2,-A3,-phi1,-phi2)
    
    dimer_array_t1 = np.concatenate([monomer_array_i,monomer_array_t1])
    dimer_array_t4 = np.concatenate([monomer_array_i,monomer_array_t4])
    dimer_array_p1 = np.concatenate([monomer_array_i,monomer_array_p1])
    dimer_array_p2 = np.concatenate([monomer_array_i,monomer_array_p2])
    
    file_description = '{}_z={}_A2={}_A3={}'.format(monomer_name,round(z,1),int(A2),round(A3,2))
    line_list_dimer_p1 = get_xyzR_lines(dimer_array_p1,file_description+'_p1')
    line_list_dimer_p2 = get_xyzR_lines(dimer_array_p2,file_description+'_p2')
    line_list_dimer_t1 = get_xyzR_lines(dimer_array_t1,file_description+'_t1')
    line_list_dimer_t4 = get_xyzR_lines(dimer_array_t4,file_description+'_t4')
    
    gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_t1 + ['\n\n--Link1--\n'] + line_list_dimer_t4 + ['\n\n--Link1--\n'] + line_list_dimer_p1 + ['\n\n--Link1--\n'] + line_list_dimer_p2 + ['\n\n\n']#+ ['\n\n--Link1--\n'] + line_list_dimer_p2 + ['\n\n\n']
    
    file_name = get_file_name_from_dict(monomer_name,params_dict)
    inp_dir = os.path.join(auto_dir,'gaussian')
    gij_xyz_path = os.path.join(inp_dir,file_name)
    with open(gij_xyz_path,'w') as f:
        f.writelines(gij_xyz_lines)
    
    return file_name

def get_file_name_from_dict(monomer_name,paras_dict):
    file_name = ''
    file_name += monomer_name
    for key,val in paras_dict.items():
        val = np.round(val,2)
        file_name += '_{}={}'.format(key,val)
    return file_name + '.inp'
    
def exec_gjf(auto_dir, monomer_name, params_dict,machine_type,isInterlayer,isTest=True):
    inp_dir = os.path.join(auto_dir,'gaussian')
    print(params_dict)
    
    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict,isInterlayer)
    cc_list = get_one_exe(file_name,machine_type)
    sh_filename = os.path.splitext(file_name)[0]+'.r1'
    sh_path = os.path.join(inp_dir,sh_filename)
    with open(sh_path,'w') as f:
        f.writelines(cc_list)
    if not(isTest):
        subprocess.run(['qsub',sh_path])
    log_file_name = os.path.splitext(file_name)[0]+'.log'
    return log_file_name
    
############################################################################################