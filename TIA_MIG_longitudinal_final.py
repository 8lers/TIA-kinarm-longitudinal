# %% 

import sys
sys.path.append('C:/Users/user/Documents/Python Scripts')

### Loading packages
from load_participants import load_tia
from load_participants import load_migraine
from kinarmAnalysis import ts_convert
from kinarmAnalysis import ts_inverse

# import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import re
import warnings
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr
import string
# from scipy.stats import ttest_ind
# from scipy.stats import ttest_1samp

"""
Suppressing irrelevant warnings. The ones of interest
pertain to doing > or < tests that involve np.nan, which
return np.nan and therefore don't matter
"""

warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#This keeps the .SVG output from registering EVERY CHARACTER AS AN INDIVIDUAL VECTOR (gross)
plt.rcParams['svg.fonttype'] = 'none'

all_tasks_names=['VGRD', 'VGRND', 'BOB', 'OH', 'OHA', 'RVGRD', 'RVGRND', 'TM', 'SPS', 'APMD', 'APMND']

#Pre-allocating lists
task_scores_tia_stored, m_scores_tia_stored, task_scores_mig_stored, m_scores_mig_stored=([] for i in range(4))
all_scores_tia_stored=list()
all_scores_mig_stored=list()
   
#Getting data for all timepoints, parking them in the lists defined above     
for TP in range(4):
    
    all_tasks_tia, all_tasks_mig=([] for i in range(2))
    
    VGRA_TIA, VGRUA_TIA, OHA_TIA, OH_TIA, RVGRA_TIA, RVGRUA_TIA, TM_TIA,\
        SPS_TIA, APMA_TIA, APMUA_TIA, BOB_TIA, vgra_p, vgrua_p, oha_p, oh_p, rvgra_p,\
        rvgrua_p, tm_p, sps_p, apma_p, apmua_p, bob_p, subs_a=load_tia(tp=TP)
        
    VGRND_MIG, VGRD_MIG, OHA_MIG, OH_MIG, RVGRND_MIG, RVGRD_MIG, TM_MIG,\
        SPS_MIG, APMD_MIG, APMND_MIG, BOB_MIG, vgrnd_mig_p, vgrd_mig_p, oha_mig_p, oh_mig_p, rvgrnd_mig_p,\
        rvgrd_mig_p, tm_mig_p, sps_mig_p, apmnd_mig_p, apmd_mig_p, bob_mig_p, subs_m=load_migraine(tp=TP)
   
    all_tasks_tia=[VGRUA_TIA, VGRA_TIA, BOB_TIA, OH_TIA, OHA_TIA, RVGRUA_TIA, RVGRA_TIA, TM_TIA, SPS_TIA, APMUA_TIA, APMA_TIA]
    all_tasks_mig=[VGRD_MIG, VGRND_MIG, BOB_MIG, OH_MIG, OHA_MIG, RVGRD_MIG, RVGRND_MIG, TM_MIG, SPS_MIG, APMD_MIG, APMND_MIG]
    all_params=[vgrnd_mig_p, vgrd_mig_p, bob_mig_p, oh_mig_p, oha_mig_p, rvgrnd_mig_p, rvgrd_mig_p, tm_mig_p, sps_mig_p, apmnd_mig_p, apmd_mig_p]
    all_scores_tia_stored.append(all_tasks_tia)
    all_scores_mig_stored.append(all_tasks_mig)
    
    task_scores_tia=np.zeros((len(subs_a), len(all_tasks_tia)),dtype=float)
    for i, task_TIA in enumerate(all_tasks_tia):
        task_scores_tia[:, i]=task_TIA[:,1]
        
    m_scores_tia=np.zeros((len(subs_a), len(all_tasks_tia)),dtype=float)
    for i, task_TIA in enumerate(all_tasks_tia):
        m_scores_tia[:, i]=task_TIA[:,0]
        
    task_scores_mig=np.zeros((len(subs_m), len(all_tasks_mig)),dtype=float)
    for i, task_MIG in enumerate(all_tasks_mig):
        task_scores_mig[:, i]=task_MIG[:,1]

    m_scores_mig=np.zeros((len(subs_m), len(all_tasks_mig)),dtype=float)
    for i, task_MIG in enumerate(all_tasks_mig):
        m_scores_mig[:, i]=task_MIG[:,0]

    
    task_scores_tia_stored.append(task_scores_tia)
    m_scores_tia_stored.append(m_scores_tia)
    
    task_scores_mig_stored.append(task_scores_mig)
    m_scores_mig_stored.append(m_scores_mig)

# %% Getting HC data for IRR study
import scipy

hc_dir = 'C:/Users/user/Documents/Python Scripts/Healthy Control IRR/'
for r, d, hc_file in os.walk(hc_dir):
    hc_file

hc_dir_sprintf = 'C:/Users/user/Documents/Python Scripts/Healthy Control IRR/%s'
hc_dfs = {f: pd.read_csv(hc_dir_sprintf % f) for f in hc_file}

## Just to make sure the diffs are in the same order as the clincial data (makes life 10000000000x easier):
#all_tasks_names=['VGRD', 'VGRND', 'BOB', 'OH', 'OHA', 'RVGRD', 'RVGRND','TM', 'SPS', 'APMD', 'APMND']

for hcdf in hc_dfs:
    hc_dfs[hcdf][hc_dfs[hcdf]==-10] = np.nan

hc_p_diffs = {'VGRD': hc_dfs[hc_file[20]] - hc_dfs[hc_file[18]],
              'VGRND': hc_dfs[hc_file[21]] - hc_dfs[hc_file[19]],
              'BOB': hc_dfs[hc_file[1]] - hc_dfs[hc_file[0]],
              'OH': hc_dfs[hc_file[5]] - hc_dfs[hc_file[4]],
              'OHA': hc_dfs[hc_file[3]] - hc_dfs[hc_file[2]],
              'RVGRD': hc_dfs[hc_file[12]] - hc_dfs[hc_file[10]],
              'RVGRND': hc_dfs[hc_file[13]] - hc_dfs[hc_file[11]],
              'TM': hc_dfs[hc_file[17]] - hc_dfs[hc_file[16]],
              'SPS': hc_dfs[hc_file[15]] - hc_dfs[hc_file[14]],
              'APMD': hc_dfs[hc_file[8]] - hc_dfs[hc_file[6]],
              'APMND': hc_dfs[hc_file[9]] - hc_dfs[hc_file[7]],
              }

np.nanmean(hc_p_diffs['SPS'].values, axis=0)
hc_delta_zts2 = {'VGRD': hc_dfs[hc_file[20]]['Z_TaskScore'] - hc_dfs[hc_file[18]]['Z_TaskScore'],
                'VGRND': hc_dfs[hc_file[21]]['Z_TaskScore'] - hc_dfs[hc_file[19]]['Z_TaskScore'],
                'BOB': hc_dfs[hc_file[1]]['Z_TaskScore'] - hc_dfs[hc_file[0]]['Z_TaskScore'],
                'OH': hc_dfs[hc_file[5]]['Z_TaskScore'] - hc_dfs[hc_file[4]]['Z_TaskScore'],
                'OHA': hc_dfs[hc_file[3]]['Z_TaskScore'] - hc_dfs[hc_file[2]]['Z_TaskScore'],
                'RVGRD': hc_dfs[hc_file[12]]['Z_TaskScore'] - hc_dfs[hc_file[10]]['Z_TaskScore'],
                'RVGRND': hc_dfs[hc_file[13]]['Z_TaskScore'] - hc_dfs[hc_file[11]]['Z_TaskScore'],
                'TM': hc_dfs[hc_file[17]]['Z_TaskScore'] - hc_dfs[hc_file[16]]['Z_TaskScore'],
                'SPS': hc_dfs[hc_file[15]]['Z_TaskScore'] - hc_dfs[hc_file[14]]['Z_TaskScore'],
                'APMD': hc_dfs[hc_file[8]]['Z_TaskScore'] - hc_dfs[hc_file[6]]['Z_TaskScore'],
                'APMND': hc_dfs[hc_file[9]]['Z_TaskScore'] - hc_dfs[hc_file[7]]['Z_TaskScore'],
                }

hc_zts2 =    {'VGRD': hc_dfs[hc_file[20]]['Z_TaskScore'],
              'VGRND': hc_dfs[hc_file[21]]['Z_TaskScore'],
              'BOB': hc_dfs[hc_file[1]]['Z_TaskScore'],
              'OH': hc_dfs[hc_file[5]]['Z_TaskScore'],
              'OHA': hc_dfs[hc_file[3]]['Z_TaskScore'],
              'RVGRD': hc_dfs[hc_file[12]]['Z_TaskScore'],
              'RVGRND': hc_dfs[hc_file[13]]['Z_TaskScore'],
              'TM': hc_dfs[hc_file[17]]['Z_TaskScore'],
              'SPS': hc_dfs[hc_file[15]]['Z_TaskScore'],
              'APMD': hc_dfs[hc_file[8]]['Z_TaskScore'],
              'APMND': hc_dfs[hc_file[9]]['Z_TaskScore'],
              }

hc_zts1 =    {'VGRD': hc_dfs[hc_file[18]]['Z_TaskScore'],
              'VGRND': hc_dfs[hc_file[19]]['Z_TaskScore'],
              'BOB': hc_dfs[hc_file[0]]['Z_TaskScore'],
              'OH': hc_dfs[hc_file[4]]['Z_TaskScore'],
              'OHA': hc_dfs[hc_file[2]]['Z_TaskScore'],
              'RVGRD': hc_dfs[hc_file[10]]['Z_TaskScore'],
              'RVGRND': hc_dfs[hc_file[11]]['Z_TaskScore'],
              'TM': hc_dfs[hc_file[16]]['Z_TaskScore'],
              'SPS': hc_dfs[hc_file[14]]['Z_TaskScore'],
              'APMD': hc_dfs[hc_file[6]]['Z_TaskScore'],
              'APMND': hc_dfs[hc_file[7]]['Z_TaskScore'],
              }

# %% New MEGAFIGURE1 (2?)

# all_tasks_names = ['VGR-D/UA', 'VGR-ND/A', 'BOB', 'OH', 'OHA', 'RVGR-D/UA', 'RVGR-ND/A', 'TMT', 'SPS', 'APM-UA', 'APM-A']
all_tasks_names = ['VGRD', 'VGRND', 'BOB', 'OH', 'OHA', 'RVGRD', 'RVGRND', 'TM', 'SPS', 'APMD', 'APMND']
CI=[2.19,2.61,1.59,1.97,1.69,1.59,2.14,1.03,1.76,2.16,2.05] 
LE=[0, 0, 0, 0, 0, -0.78, -0.67, -0.23, -0.39, 0, 0]

'''
I think everything is scaled by 1.42 (sqrt(2)) because BOTH the TIAs and the 
controls have a sqrt(2) to account for, because of the repeated assessments?
Hence only dividing SC by [1.96] as opposed to 1.96*1.42 = [2.77]

Edit - This works. I'm not insane!'
'''

initial_tia=task_scores_tia_stored[0]
initial_mig=task_scores_mig_stored[0]

final_tia=task_scores_tia_stored[3]
final_mig=task_scores_mig_stored[1]

#Setting zeros to np.nan
initial_tia[initial_tia==0]=np.nan
initial_mig[initial_mig==0]=np.nan

final_tia[final_tia==0]=np.nan
final_mig[final_mig==0]=np.nan

#Setting max values at 4 to make sure subtleties/values close to barriers can be seen
row, col, z_max, counter, ax_max = 0, 0, 10, 0, 6

nsubplots=len(all_tasks_tia)

z_taskScores=np.linspace(-((z_max+0.5)),(z_max+0.5),10000, dtype=float)

probs_tia = dict()
probs_mig = dict()

fig, ax = plt.subplots(2, 3, figsize = (7.5, 5))
plt.subplots_adjust(top = 0.95, bottom = 0.1, left= 0.1, right = 0.95, hspace = 0.4, wspace = 0.4)

new_counter2 = -1
ax_max0 = [6, 6, 4]

for task in [1, 6, 7]:
    
    new_counter2 += 1
    ax_max = ax_max0[new_counter2]    
    i = all_tasks_names[task]
    counter = task
    
    ###
    
    x_tia1, x_tia2 = np.sort(ts_inverse(task_scores_tia_stored[0][:,task])), np.sort(ts_inverse(task_scores_tia_stored[3][:,task]))
    x_mig1, x_mig2 = np.sort(ts_inverse(task_scores_mig_stored[0][:,task])), np.sort(ts_inverse(task_scores_mig_stored[1][:,task]))
    
    title_str = re.sub('TM', 'TM', all_tasks_names[task])
    title_str_tia = re.sub('ND', '-A', title_str)
    title_str_mig = re.sub('ND', '-ND', title_str)
    
    ax[0, row].set_title('2-week: %s' % title_str_tia, fontsize = 12, fontweight = 'regular')
    ax[0, row].plot([0, 0], [0, 1], ls = ':', c = 'grey')
    ax[0, row].plot([-1.96, -1.96], [0, 1], ls = ':', c = 'grey')
    ax[0, row].plot([+1.96, +1.96], [0, 1], ls = ':', c = 'grey')
    ax[0, row].plot(x_tia1[np.isnan(x_tia1)==0], np.arange(0+1/sum(np.isnan(x_tia1)==0), 1+1/sum(np.isnan(x_tia1)==0), 1/sum(np.isnan(x_tia1)==0)), marker = None, ms = 3, c = 'darkgrey', mfc = 'none')
    # ax[0, row].plot(x_mig1[np.isnan(x_mig1)==0], np.arange(0+1/sum(np.isnan(x_mig1)==0), 1+1/sum(np.isnan(x_mig1)==0), 1/sum(np.isnan(x_mig1)==0)), marker = None, ms = 3, c = 'b', mfc = 'none')
    ax[0, row].set_facecolor('white')
    ax[0, row].spines['bottom'].set_edgecolor('black')
    ax[0, row].spines['top'].set_edgecolor('black')
    ax[0, row].spines['left'].set_edgecolor('black')
    ax[0, row].spines['right'].set_edgecolor('black')
    ax[0, row].set_xlim(-ax_max, ax_max)
    ax[0, row].set_ylim(0, 1.01)
    
    ax[0, row].set_title('1-year: %s' % title_str_tia, fontsize = 12, fontweight = 'regular')
    ax[0, row].plot([0, 0], [0, 1], ls = ':', c = 'grey')
    ax[0, row].plot([-1.96, -1.96], [0, 1], ls = ':', c = 'grey')
    ax[0, row].plot([+1.96, +1.96], [0, 1], ls = ':', c = 'grey')
    ax[0, row].plot(x_tia2[np.isnan(x_tia2)==0], np.arange(0+1/sum(np.isnan(x_tia2)==0), 1+1/sum(np.isnan(x_tia2)==0), 1/sum(np.isnan(x_tia2)==0)), marker = None, ms = 3, c = 'dimgrey', mfc = 'none')
    # ax[0, row].plot(x_mig1[np.isnan(x_mig1)==0], np.arange(0+1/sum(np.isnan(x_mig1)==0), 1+1/sum(np.isnan(x_mig1)==0), 1/sum(np.isnan(x_mig1)==0)), marker = None, ms = 3, c = 'b', mfc = 'none')
    ax[0, row].set_facecolor('white')
    ax[0, row].spines['bottom'].set_edgecolor('black')
    ax[0, row].spines['top'].set_edgecolor('black')
    ax[0, row].spines['left'].set_edgecolor('black')
    ax[0, row].spines['right'].set_edgecolor('black')
    ax[0, row].set_xlim(-ax_max, ax_max)
    ax[0, row].set_ylim(0, 1.01)
    
    ax[1, row].set_title('2-week: %s' % title_str_mig, fontsize = 12, fontweight = 'regular')
    ax[1, row].plot([0, 0], [0, 1], ls = ':', c = 'grey')
    ax[1, row].plot([-1.96, -1.96], [0, 1], ls = ':', c = 'grey')
    ax[1, row].plot([+1.96, +1.96], [0, 1], ls = ':', c = 'grey')
    # ax[1, row].plot(x_tia2[np.isnan(x_tia2)==0], np.arange(0+1/sum(np.isnan(x_tia2)==0), 1+1/sum(np.isnan(x_tia2)==0), 1/sum(np.isnan(x_tia2)==0)), marker = None, ms = 3, c = 'r', mfc = 'none')
    ax[1, row].plot(x_mig1[np.isnan(x_mig1)==0], np.arange(0+1/sum(np.isnan(x_mig1)==0), 1+1/sum(np.isnan(x_mig1)==0), 1/sum(np.isnan(x_mig1)==0)), marker = None, ms = 3, c = 'darkgrey', mfc = 'none')
    ax[1, row].set_facecolor('white')
    ax[1, row].spines['bottom'].set_edgecolor('black')
    ax[1, row].spines['top'].set_edgecolor('black')
    ax[1, row].spines['left'].set_edgecolor('black')
    ax[1, row].spines['right'].set_edgecolor('black')
    ax[1, row].set_xlim(-ax_max, ax_max)
    ax[1, row].set_ylim(0, 1.01)
    
    ax[1, row].set_title('1-year: %s' % title_str_mig, fontsize = 12, fontweight = 'regular')
    ax[1, row].plot([0, 0], [0, 1], ls = ':', c = 'grey')
    ax[1, row].plot([-1.96, -1.96], [0, 1], ls = ':', c = 'grey')
    ax[1, row].plot([+1.96, +1.96], [0, 1], ls = ':', c = 'grey')
    # ax[1, row].plot(x_tia2[np.isnan(x_tia2)==0], np.arange(0+1/sum(np.isnan(x_tia2)==0), 1+1/sum(np.isnan(x_tia2)==0), 1/sum(np.isnan(x_tia2)==0)), marker = None, ms = 3, c = 'r', mfc = 'none')
    ax[1, row].plot(x_mig2[np.isnan(x_mig2)==0], np.arange(0+1/sum(np.isnan(x_mig2)==0), 1+1/sum(np.isnan(x_mig2)==0), 1/sum(np.isnan(x_mig2)==0)), marker = None, ms = 3, c = 'dimgrey', mfc = 'none')
    ax[1, row].set_facecolor('white')
    ax[1, row].spines['bottom'].set_edgecolor('black')
    ax[1, row].spines['top'].set_edgecolor('black')
    ax[1, row].spines['left'].set_edgecolor('black')
    ax[1, row].spines['right'].set_edgecolor('black')
    ax[1, row].set_xlim(-ax_max, ax_max)
    ax[1, row].set_ylim(0, 1.01)
    
    ax[0, row].set_xticks(np.arange(-ax_max, ax_max+2, 2))
    ax[0, row].set_xticklabels(np.arange(-ax_max, ax_max+2, 2), fontsize = 10)
    ax[0, row].set_yticks(np.arange(0, 1.2, 0.2))
    ax[0, row].set_yticklabels(np.round(np.arange(0, 1.2, 0.2), 2), fontsize = 10)
    
    ax[0, row].set_xticks(np.arange(-ax_max, ax_max+2, 2))
    ax[0, row].set_xticklabels(np.arange(-ax_max, ax_max+2, 2), fontsize = 10)
    ax[0, row].set_yticks(np.arange(0, 1.2, 0.2))
    ax[0, row].set_yticklabels(np.round(np.arange(0, 1.2, 0.2), 2), fontsize = 10)
    
    ax[1, row].set_xticks(np.arange(-ax_max, ax_max+2, 2))
    ax[1, row].set_xticklabels(np.arange(-ax_max, ax_max+2, 2), fontsize = 10)
    ax[1, row].set_yticks(np.arange(0, 1.2, 0.2))
    ax[1, row].set_yticklabels(np.round(np.arange(0, 1.2, 0.2), 2), fontsize = 10)
    
    ax[1, row].set_xticks(np.arange(-ax_max, ax_max+2, 2))
    ax[1, row].set_xticklabels(np.arange(-ax_max, ax_max+2, 2), fontsize = 10)
    ax[1, row].set_yticks(np.arange(0, 1.2, 0.2))
    ax[1, row].set_yticklabels(np.round(np.arange(0, 1.2, 0.2), 2), fontsize = 10)
    
    if row==0:
        ax[0, row].set_xlabel('Z-Task Score', fontsize = 10, fontweight = 'regular')
        ax[0, row].set_ylabel('Cumulative sum', fontsize = 10, fontweight = 'regular')
    
    row += 1
    
new_counter2 = -1
col, row = 0, 0

for txt in range(6):
    if col==3: col = 0; row += 1
    
    new_counter2 += 1
    ax[row, col].annotate(string.ascii_uppercase[new_counter2], xy = (-0.10, 1.05), xycoords = 'axes fraction',
                          fontsize = 18, fontweight = 'bold')    
    
    col += 1
    
# %% Longitudinal scatterplots

#Setting max values at 4 to make sure subtleties/values close to barriers can be seen
row, col, z_max, counter, ax_max = 0, 0, 10, 0, 6

fig, ax = plt.subplots(2, 3, figsize = (15, 10))
plt.subplots_adjust(top = 0.95, bottom = 0.1, left= 0.1, right = 0.95, hspace = 0.4, wspace = 0.4)

new_counter2 = -1
ax_max0 = [6, 6, 4]

probs_tia = dict()
probs_mig = dict()

for task in [1, 6, 7]:
    
    new_counter2 += 1
    ax_max = ax_max0[new_counter2]    
    i = all_tasks_names[task]
    counter = task
       
    title_str = re.sub('TM', 'TMT', all_tasks_names[task])
    title_str_tia = re.sub('ND', '-A', title_str)
    title_str_mig = re.sub('ND', '-ND', title_str)
        
    z_taskScores_posCI = z_taskScores+CI[counter]-LE[counter]
    z_taskScores_negCI = z_taskScores-CI[counter]-LE[counter]
    
    one_sided_ts = ts_convert(z_taskScores)
    one_sided_ts_posCI = ts_convert(z_taskScores_posCI)
    one_sided_ts_negCI = ts_convert(z_taskScores_negCI)
    
    sig_better_prc_tia = np.zeros((len(CI), 1), dtype=int)
    sig_better_prc_mig = np.zeros((len(CI), 1), dtype=int)
    sig_worse_prc_tia = np.zeros((len(CI), 1), dtype=int)
    sig_worse_prc_mig = np.zeros((len(CI), 1), dtype=int)

    tia_data_final = final_tia[np.isnan(final_tia[:,counter])==0,counter]
    tia_data_initial = initial_tia[np.isnan(final_tia[:,counter])==0,counter]
    mig_data_final = final_mig[np.isnan(final_mig[:,counter])==0,counter]
    mig_data_initial = initial_mig[np.isnan(final_mig[:,counter])==0,counter]
    
    initial_tia_over = tia_data_initial[tia_data_initial>ax_max]
    initial_mig_over = mig_data_initial[mig_data_initial>ax_max]
    final_tia_over = tia_data_final[tia_data_final>ax_max]
    final_mig_over = mig_data_final[mig_data_final>ax_max]
    
    worse_tia = tia_data_final>=ts_convert(ts_inverse(tia_data_initial)+CI[counter]+LE[counter])
    better_tia = tia_data_final<ts_convert(ts_inverse(tia_data_initial)-CI[counter]+LE[counter])
    nc_tia = (tia_data_final<ts_convert(ts_inverse(tia_data_initial)+CI[counter]+LE[counter])) & \
    (tia_data_final>=ts_convert(ts_inverse(tia_data_initial)-CI[counter]+LE[counter]))
    
    worse_mig = mig_data_final>=ts_convert(ts_inverse(mig_data_initial)+CI[counter]+LE[counter])
    better_mig = mig_data_final<ts_convert(ts_inverse(mig_data_initial)-CI[counter]+LE[counter])
    nc_mig = (mig_data_final<ts_convert(ts_inverse(mig_data_initial)+CI[counter]+LE[counter])) & \
    (mig_data_final>=ts_convert(ts_inverse(mig_data_initial)-CI[counter]+LE[counter]))
    
    tia_data_initial_copy, mig_data_initial_copy = np.copy(tia_data_initial), np.copy(mig_data_initial)
    tia_data_final_copy, mig_data_final_copy = np.copy(tia_data_final), np.copy(mig_data_final)
    tia_data_initial_inds, mig_data_initial_inds = np.where(tia_data_initial_copy>ax_max)[0], np.where(mig_data_initial_copy>ax_max)[0]
    tia_data_final_inds, mig_data_final_inds = np.where(tia_data_final_copy>ax_max)[0], np.where(mig_data_final_copy>ax_max)[0]
    
    tia_data_initial[tia_data_initial>ax_max] = ax_max
    mig_data_initial[mig_data_initial>ax_max] = ax_max
    tia_data_final[tia_data_final>ax_max] = ax_max
    mig_data_final[mig_data_final>ax_max] = ax_max
    
    for ii in tia_data_initial_inds:
        ax[0, col].text(ax_max, tia_data_final[ii]+0.1, np.round(tia_data_initial_copy[ii], 2), fontsize = 12, fontweight = 'regular') #x is initial, y is final
    for ii in tia_data_final_inds:
        ax[0, col].text(tia_data_initial[ii], ax_max+0.1, np.round(tia_data_final_copy[ii], 2), fontsize = 12, fontweight = 'regular') #x is initial, y is final
    for ii in mig_data_initial_inds:
        ax[1, col].text(ax_max, mig_data_final[ii]+0.1, np.round(mig_data_initial_copy[ii], 2), fontsize = 12, fontweight = 'regular') #x is initial, y is final
    for ii in mig_data_final_inds:
        ax[1, col].text(mig_data_initial[ii], ax_max+0.1, np.round(mig_data_final_copy[ii], 2), fontsize = 12, fontweight = 'regular') #x is initial, y is final
    
    ax[0, col].plot(one_sided_ts, one_sided_ts, '-k', linewidth=1)
    ax[0, col].plot(one_sided_ts_posCI, one_sided_ts, ':k', linewidth=1) #bottom boundary on plots
    ax[0, col].plot(one_sided_ts_negCI, one_sided_ts, ':k', linewidth=1) #top boundary on plots
    
    ax[1, col].plot(one_sided_ts, one_sided_ts, '-k', linewidth=1)
    ax[1, col].plot(one_sided_ts_posCI, one_sided_ts, ':k', linewidth=1) #bottom boundary on plots
    ax[1, col].plot(one_sided_ts_negCI, one_sided_ts, ':k', linewidth=1) #top boundary on plots
    
    ax[0, col].plot([1.96, 1.96], [-0.5,ax_max+0.5], '-k', linewidth=0.25)
    ax[0, col].plot([-0.5,ax_max+0.5], [1.96, 1.96], '-k', linewidth=0.25)
    
    ax[1, col].plot([1.96, 1.96], [-0.5,ax_max+0.5], '-k', linewidth=0.25)
    ax[1, col].plot([-0.5,ax_max+0.5], [1.96, 1.96], '-k', linewidth=0.25)
    
    ax[0, col].scatter(tia_data_initial[worse_tia], tia_data_final[worse_tia], facecolor='darkgrey', edgecolors='darkgrey')
    ax[0, col].scatter(tia_data_initial[better_tia], tia_data_final[better_tia], facecolor='darkgrey', edgecolors='darkgrey')
    ax[0, col].scatter(tia_data_initial[nc_tia], tia_data_final[nc_tia], facecolor='none', edgecolors='dimgrey', linewidth=2)
    
    ax[1, col].scatter(mig_data_initial[worse_mig], mig_data_final[worse_mig], facecolor='darkgrey', edgecolors='darkgrey', marker='o')
    ax[1, col].scatter(mig_data_initial[better_mig], mig_data_final[better_mig], facecolor='darkgrey', edgecolors='darkgrey', marker='o')
    ax[1, col].scatter(mig_data_initial[nc_mig], mig_data_final[nc_mig], facecolor='none', edgecolors='dimgrey', linewidth=2, marker='o')
    
    ax[0, col].set_title(title_str_tia, fontweight = 'regular', fontsize = 16)
    ax[0, col].set_xlim(-0.5,ax_max+0.5)
    ax[0, col].set_ylim(-0.5,ax_max+0.5)
    
    ax[1, col].set_title(title_str_mig, fontweight = 'regular', fontsize = 16)
    ax[1, col].set_xlim(-0.5,ax_max+0.5)
    ax[1, col].set_ylim(-0.5,ax_max+0.5)
    # ax[col].set_aspect('equal', adjustable='box')
    
    ax[0, col].set_facecolor('white')
    ax[0, col].spines['top'].set_color('k')
    ax[0, col].spines['left'].set_color('k')
    ax[0, col].spines['bottom'].set_color('k')
    ax[0, col].spines['right'].set_color('k')
    
    ax[1, col].set_facecolor('white')
    ax[1, col].spines['top'].set_color('k')
    ax[1, col].spines['left'].set_color('k')
    ax[1, col].spines['bottom'].set_color('k')
    ax[1, col].spines['right'].set_color('k')

    ###
    
    SD = np.asarray(CI[task])/1.96 
    
    deltaTask_tia = ts_inverse(task_scores_tia_stored[3][:, task]) - ts_inverse(task_scores_tia_stored[0][:, task])
    deltaTask_mig = ts_inverse(task_scores_mig_stored[1][:, task]) - ts_inverse(task_scores_mig_stored[0][:, task])
    
    deltaTask_tia[deltaTask_tia==-np.inf] = np.nan
    deltaTask_tia[deltaTask_tia==np.inf] = np.nan
    deltaTask_mig[deltaTask_mig==-np.inf] = np.nan
    deltaTask_mig[deltaTask_mig==np.inf] = np.nan
    
    probs_tia.update({'%s' % all_tasks_names[task] : [scipy.stats.norm(loc=LE[task], scale=SD).cdf(i) for i in deltaTask_tia]})
    probs_mig.update({'%s' % all_tasks_names[task] : [scipy.stats.norm(loc=LE[task], scale=SD).cdf(i) for i in deltaTask_mig]})
    
    min_0025 = scipy.stats.norm(loc=LE[task], scale=SD).ppf(0.025)
    max_0975 = scipy.stats.norm(loc=LE[task], scale=SD).ppf(0.975)
    
    deltaTask_tia2 = np.copy(deltaTask_tia)
    deltaTask_tia2[deltaTask_tia2 < -ax_max] = -ax_max
    deltaTask_tia2[deltaTask_tia2 > ax_max] = ax_max
    
    deltaTask_mig2 = np.copy(deltaTask_mig)
    deltaTask_mig2[deltaTask_mig2 < -ax_max] = -ax_max
    deltaTask_mig2[deltaTask_mig2 > ax_max] = ax_max

    ptia2 = probs_tia[i]
    for ind_k, k in enumerate(deltaTask_tia):
        if (k < min_0025) | (k > max_0975):
            # ax[row, 0].annotate('%.2f, %.4f' % (k, ptia2[ind_k]), (k, 0.1), rotation = 90)
            if k<-6: addon = '(%.2f)' % k
            elif k>6: addon = '(%.2f)' % k
            else: addon = ''
            
            if ptia2[ind_k]>0.975: p_addon = 1-ptia2[ind_k]
            else: p_addon = ptia2[ind_k] 
            
            ax[0, col].annotate('%s%.2E' % (addon, p_addon), 
                                xytext = (task_scores_tia_stored[0][ind_k, task]+(ax_max/8), task_scores_tia_stored[3][ind_k, task]), 
                                rotation = 0, ha = 'left', va = 'center',
                                xy = (task_scores_tia_stored[0][ind_k, task], task_scores_tia_stored[3][ind_k, task]),
                                fontsize = 12,
                                arrowprops = dict(width = 1.5, frac = 0.05, headwidth = 5, facecolor = 'k'))
    
    pmig2 = probs_mig[i]
    for ind_k, k in enumerate(deltaTask_mig):
        if (k < min_0025) | (k > max_0975):
            # ax[row, 0].annotate('%.2f, %.4f' % (k, ptia2[ind_k]), (k, 0.1), rotation = 90)
            if k<-6: addon = '(%.2f)' % k
            elif k>6: addon = '(%.2f)' % k
            else: addon = ''
            
            if pmig2[ind_k]>0.975: p_addon = 1-pmig2[ind_k]
            else: p_addon = pmig2[ind_k] 
            
            ax[1, col].annotate('%s%.2E' % (addon, p_addon), 
                                xytext = (task_scores_mig_stored[0][ind_k, task]+(ax_max/8), task_scores_mig_stored[1][ind_k, task]), 
                                rotation = 0, ha = 'left', va = 'center',
                                xy = (task_scores_mig_stored[0][ind_k, task], task_scores_mig_stored[1][ind_k, task]),
                                fontsize = 12, 
                                arrowprops = dict(width = 1.5, frac = 0.05, headwidth = 5, facecolor = 'k'))
    
    
    ax[0, col].set_xticks(np.arange(0, ax_max+2, 2))
    ax[0, col].set_xticklabels(np.arange(0, ax_max+2, 2), fontsize = 12)
    ax[0, col].set_yticks(np.arange(0, ax_max+2, 2))
    ax[0, col].set_yticklabels(np.arange(0, ax_max+2, 2), fontsize = 12)
    ax[0, col].set_xlim(-0.1, ax_max+0.1)
    ax[0, col].set_ylim(-0.1, ax_max+0.1)
    
    ax[1, col].set_xticks(np.arange(0, ax_max+2, 2))
    ax[1, col].set_xticklabels(np.arange(0, ax_max+2, 2), fontsize = 12)
    ax[1, col].set_yticks(np.arange(0, ax_max+2, 2))
    ax[1, col].set_yticklabels(np.arange(0, ax_max+2, 2), fontsize = 12)
    ax[1, col].set_xlim(-0.1, ax_max+0.1)
    ax[1, col].set_ylim(-0.1, ax_max+0.1)
    
    if col==0:
        ax[0, col].set_xlabel('Initial Task Score', fontsize = 14, fontweight = 'regular')
        ax[0, col].set_ylabel('Final Task Score', fontsize = 14, fontweight = 'regular') 
   
    col += 1

# %% Heat maps of differences

ts_CI = np.asarray(CI)[np.asarray([0,1,2,3,4,5,6,7,8,10])]
ts_LE = np.asarray(LE)[np.asarray([0,1,2,3,4,5,6,7,8,10])]

tia_1_ts=task_scores_tia_stored[0][:, [0,1,2,3,4,5,6,7,8,10]]
tia_2_ts=task_scores_tia_stored[3][:, [0,1,2,3,4,5,6,7,8,10]]
mig_1_ts=task_scores_mig_stored[0][:, [0,1,2,3,4,5,6,7,8,10]]
mig_2_ts=task_scores_mig_stored[1][:, [0,1,2,3,4,5,6,7,8,10]]

tia_1_ts[(tia_1_ts==0) | (tia_2_ts==0)]=np.nan
tia_2_ts[(tia_1_ts==0) | (tia_2_ts==0)]=np.nan
mig_1_ts[(mig_1_ts==0) | (mig_2_ts==0)]=np.nan
mig_2_ts[(mig_1_ts==0) | (mig_2_ts==0)]=np.nan

### First fail, both pass, both fail stats
first_imp_tia = np.round(np.sum((tia_1_ts>1.96), axis=0) / np.sum(np.isnan(tia_1_ts)==0, axis=0), 2)
both_imp_tia = np.round(np.sum((tia_1_ts>1.96) & (tia_2_ts>1.96), axis=0) / np.sum(np.isnan(tia_2_ts)==0, axis=0), 2)
both_pass_tia = np.round(np.sum((tia_1_ts<=1.96) & (tia_2_ts<=1.96), axis=0) / np.sum(np.isnan(tia_2_ts)==0, axis=0), 2)

first_imp_mig = np.round(np.sum((mig_1_ts>1.96), axis=0) / np.sum(np.isnan(tia_1_ts)==0, axis=0), 2)
both_imp_mig = np.round(np.sum((mig_1_ts>1.96) & (mig_2_ts>1.96), axis=0) / np.sum(np.isnan(tia_2_ts)==0, axis=0), 2)
both_pass_mig = np.round(np.sum((mig_1_ts<=1.96) & (mig_2_ts<=1.96), axis=0) / np.sum(np.isnan(tia_2_ts)==0, axis=0), 2)
###

all_tia_data_diffs=tia_2_ts-tia_1_ts
all_mig_data_diffs=mig_2_ts-mig_1_ts

ts_CI_tia=np.repeat(ts_CI.reshape(1,-1), np.shape(all_tia_data_diffs)[0], axis=0)
ts_CI_mig=np.repeat(ts_CI.reshape(1,-1), np.shape(all_mig_data_diffs)[0], axis=0)

ts_LE_tia=np.repeat(ts_LE.reshape(1,-1), np.shape(all_tia_data_diffs)[0], axis=0)
ts_LE_mig=np.repeat(ts_LE.reshape(1,-1), np.shape(all_mig_data_diffs)[0], axis=0)

ts_CI_tia[np.isnan(all_tia_data_diffs)]=np.nan
ts_CI_mig[np.isnan(all_mig_data_diffs)]=np.nan

all_tia_data_diffs[(tia_2_ts >= ts_convert(ts_inverse(tia_1_ts) - ts_CI_tia + ts_LE_tia)) &
                   (tia_2_ts < ts_convert(ts_inverse(tia_1_ts) + ts_CI_tia + ts_LE_tia))]=0

all_mig_data_diffs[(mig_2_ts >= ts_convert(ts_inverse(mig_1_ts) - ts_CI_mig + ts_LE_mig)) &
                   (mig_2_ts < ts_convert(ts_inverse(mig_1_ts) + ts_CI_mig + ts_LE_mig))]=0

# Plotting the heatmap
fig0 = plt.figure()

ax0 = fig0.add_axes([.15, .15, .30, .70])
ax1 = fig0.add_axes([.55, .15, .225, .70])
cbar_ax = fig0.add_axes([.15, .10, .30, .025])

ax_lim=4
cmap=sns.color_palette("RdBu_r", 100)
sns.set(font_scale=0.75)

sns.heatmap(all_tia_data_diffs[np.sum(np.isnan(all_tia_data_diffs)==1, axis=1)<10, :].T, ax=ax0, \
            vmin=-ax_lim, vmax=ax_lim, cmap=cmap, linecolor='k', \
            xticklabels='', \
            yticklabels=np.asarray(all_tasks_names)[np.asarray([0,1,2,3,4,5,6,7,8,10])], \
            cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"})

ax0.set_title('TIA Task Score Change', fontsize=16)

sns.heatmap(all_mig_data_diffs[np.sum(np.isnan(all_mig_data_diffs)==1, axis=1)<10, :].T, ax=ax1, \
            vmin=-ax_lim, vmax=ax_lim, cmap=cmap, linecolor='k', \
            xticklabels='', \
            yticklabels=np.asarray(all_tasks_names)[np.asarray([0,1,2,3,4,5,6,7,8,10])], \
            cbar=False)

ax0.set_yticklabels(labels=ax0.get_yticklabels(), rotation=0, fontsize=14, fontweight = 'bold', horizontalalignment='right')
ax1.set_yticklabels(labels=ax1.get_yticklabels(), rotation=0, fontsize=14, fontweight = 'bold', horizontalalignment='right')
ax1.set_title('Migraine Task Score Change', fontsize=16)

ax0.set_facecolor('lightgrey')
ax1.set_facecolor('lightgrey')

cbar_ax.set_xticks(np.arange(-ax_lim, ax_lim+1, 1))
cbar_ax.set_xticklabels(labels=np.arange(-ax_lim, ax_lim+1, 1), fontsize=12, fontweight = 'bold')

cols_tia = all_tia_data_diffs[np.sum(np.isnan(all_tia_data_diffs)==1, axis=1)<10, :].T.shape[1]
cols_mig = all_mig_data_diffs[np.sum(np.isnan(all_mig_data_diffs)==1, axis=1)<10, :].T.shape[1]
rows_both = 10

lw1, lw2 = 0.25, 2.50

for i in np.arange(cols_tia+1):
    lw = lw1
    if (i==0) | (i==cols_tia):
        lw = lw2
    ax0.plot([i, i], [0, rows_both], '-k', linewidth=lw)

for i in np.arange(cols_mig+1):
    lw = lw1
    if (i==0) | (i==cols_mig):
        lw = lw2
    ax1.plot([i, i], [0, rows_both], '-k', linewidth=lw)    
    
for i in np.arange(rows_both+1):
    lw = lw1
    if (i==0) | (i==rows_both):
        lw = lw2
    ax0.plot([0, cols_tia], [i, i], '-k', linewidth=lw)
    
for i in np.arange(rows_both+1):
    lw = lw1
    if (i==0) | (i==rows_both):
        lw = lw2
    ax1.plot([0, cols_mig], [i, i], '-k', linewidth=lw)

tia_sig_better = np.sum(all_tia_data_diffs<0, axis = 0)
tia_sig_worse = np.sum(all_tia_data_diffs>0, axis = 0)

mig_sig_better = np.sum(all_mig_data_diffs<0, axis = 0)
mig_sig_worse = np.sum(all_mig_data_diffs>0, axis = 0)

for j in np.arange(rows_both):
    xtia, xmig = 28.25, 21.25

    ax0.text(xtia, j+0.35, ('%i/%i' % (tia_sig_better[j], cols_tia)), fontsize=14, fontweight = 'bold', color='dodgerblue')
    ax0.text(xtia, j+0.65, ('%i/%i' % (tia_sig_worse[j], cols_tia)), fontsize=14, fontweight = 'bold', color='firebrick')
    
    ax1.text(xmig, j+0.35, ('%i/%i' % (mig_sig_better[j], cols_mig)), fontsize=14, fontweight = 'bold', color='dodgerblue')
    ax1.text(xmig, j+0.65, ('%i/%i' % (mig_sig_worse[j], cols_mig)), fontsize=14, fontweight = 'bold', color='firebrick')

shutUpPython=1    
    
# %% Segment-wise significant changes

probs_tia.update({'%s' % all_tasks_names[task] : [scipy.stats.norm(loc=LE[task], scale=SD).cdf(i) for i in deltaTask_tia]})

from scipy.stats import fisher_exact as fex

all_CI = CI
tia_task_names = ['VGR-UA', 'VGR-A', 'BOB', 'OH', 'OHA', 'RVGR-UA', 'RVGR-A', 'TMT', 'SPS', 'APM-UA', 'APM-A']
# all_tasks_names

task_scores_tia_stored[0][task_scores_tia_stored[0]==0] = np.nan
task_scores_tia_stored[1][task_scores_tia_stored[1]==0] = np.nan
task_scores_tia_stored[2][task_scores_tia_stored[2]==0] = np.nan
task_scores_tia_stored[3][task_scores_tia_stored[3]==0] = np.nan

task_scores_mig_stored[0][task_scores_mig_stored[0]==0] = np.nan
task_scores_mig_stored[1][task_scores_mig_stored[1]==0] = np.nan

# tia_est12, tia_true12, tia_all_num12 = get_est_true(set1=ts_inverse(scores_for_task[0][:, k]), set2=ts_inverse(scores_for_task[1][:, k]), LE[k], CI[k])
def get_est_true(set1, set2, LEf, CIf):
    from kinarmAnalysis import ts_convert
    real_diff = set2 - set1 
    mu1, mu2, sd1, sd2 = 0, LEf, 1, 1  
    est0 = list()
    len_est = 100

    for i in range(1000):
        est_1 = np.random.normal(mu1, sd1, len_est)
        est_2 = np.random.normal(mu2, sd2, len_est)
        est_diff = est_2 - est_1
        
        est0.append(sum(np.abs(est_diff)>(1.96*np.sqrt(2))))  
    
    all_num = sum(np.isnan(real_diff)==0)
    corr_fac = all_num/len_est
    est = int(np.ceil(np.mean(est0)*corr_fac))
    
    true = sum(ts_convert(set1)>=ts_convert(set2 + CIf - LEf)) + sum(ts_convert(set1)<ts_convert(set2 - CIf - LEf))
    
    return(est, true, all_num)

def get_bh_cut(p_vals, alpha_nought = 0.05, plotting = 0):
    import numpy as np
    p_vals = np.asarray(p_vals)
    sorted_p_vals = np.sort(p_vals).ravel()
    lin_pval_ests = np.linspace(0, alpha_nought, len(sorted_p_vals))
    over_vals = np.where((sorted_p_vals > lin_pval_ests))[0]
    cutpoint = lin_pval_ests[over_vals[1]]
    
    if plotting==1:
        plt.figure()
        plt.plot([0, len(lin_pval_ests)], [alpha_nought, alpha_nought], ':k', label = 'Alpha_0')
        plt.plot(sorted_p_vals, label = 'Sorted p-values')
        plt.plot((1-sorted_p_vals)[::-1], label = 'Mirrored sorted (1-p)-values')
        plt.plot(lin_pval_ests, label = 'Reference line')
        plt.legend()
    
    return cutpoint


fig, ax=plt.subplots(2,5,figsize=(10,10))
plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.10, right=0.90)

row, col, count, count2 = 0, 0, 0, -1

stored_fex_tia12 = dict()
stored_fex_tia23 = dict()
stored_fex_tia34 = dict()

deltaTask_tia_probs01 = list()
deltaTask_tia_probs12 = list()
deltaTask_tia_probs23 = list()

for task in [0,1,2,3,4,5,6,7,8,10]:
        
    SD = np.asarray(CI[task])/1.96 

    deltaTask_tia01 = ts_inverse(task_scores_tia_stored[1][:, task]) - ts_inverse(task_scores_tia_stored[0][:, task])
    deltaTask_tia01[deltaTask_tia01==-np.inf] = np.nan
    deltaTask_tia01[deltaTask_tia01==np.inf] = np.nan
    deltaTask_tia_probs01.append([scipy.stats.norm(loc=LE[task], scale=SD).cdf(i) for i in deltaTask_tia01])
    
    deltaTask_tia12 = ts_inverse(task_scores_tia_stored[2][:, task]) - ts_inverse(task_scores_tia_stored[1][:, task])
    deltaTask_tia12[deltaTask_tia12==-np.inf] = np.nan
    deltaTask_tia12[deltaTask_tia12==np.inf] = np.nan
    deltaTask_tia_probs12.append([scipy.stats.norm(loc=LE[task], scale=SD).cdf(i) for i in deltaTask_tia12])
    
    deltaTask_tia23 = ts_inverse(task_scores_tia_stored[3][:, task]) - ts_inverse(task_scores_tia_stored[2][:, task])
    deltaTask_tia23[deltaTask_tia23==-np.inf] = np.nan
    deltaTask_tia23[deltaTask_tia23==np.inf] = np.nan
    deltaTask_tia_probs23.append([scipy.stats.norm(loc=LE[task], scale=SD).cdf(i) for i in deltaTask_tia23])
     
p_vals01 = np.concatenate((np.concatenate((deltaTask_tia_probs01)), 
                           np.concatenate((deltaTask_tia_probs12)), 
                           np.concatenate((deltaTask_tia_probs23))), axis = 0)
pvals_original = np.copy(p_vals01)
p_vals01 = p_vals01[np.isnan(p_vals01)==False]
threshold_p_val = get_bh_cut(p_vals01, alpha_nought = 0.025, plotting = 0)

for k in [0,1,2,3,4,5,6,7,8,10]:
    
    count2 += 1
    task = k
    
    if col==5: col = 0; row += 1
        
    scores_for_task = np.copy(task_scores_tia_stored)
    
    ax[row, col].plot([0, 3], [1.96, 1.96], ':k', lw=2)
    
    ax[row, col].plot([0, 0], [0, 6], ':k', lw=1)
    ax[row, col].plot([1, 1], [0, 6], ':k', lw=1)
    ax[row, col].plot([2, 2], [0, 6], ':k', lw=1)
    ax[row, col].plot([3, 3], [0, 6], ':k', lw=1)
    
    for s in scores_for_task:
        s[s>6] = 6
        s[s==0] = np.nan
    
    more12, less12, nc12 = 0, 0, 0
    more23, less23, nc23 = 0, 0, 0
    more34, less34, nc34 = 0, 0, 0
    mstyle = None
    
    tia_est12, tia_true12, tia_all_num12 = get_est_true(ts_inverse(scores_for_task[0][:, k]), ts_inverse(scores_for_task[1][:, k]), LE[k], CI[k])
    stored_fex_tia12.update({'%s (OR, pval, is_sig)' % all_tasks_names[k]: (np.round(fex([[tia_true12, tia_est12], [tia_all_num12, tia_all_num12]])[0], 3),
                                                                     np.round(fex([[tia_true12, tia_est12], [tia_all_num12, tia_all_num12]])[1], 3),
                                                                     fex([[tia_true12, tia_est12], [tia_all_num12, tia_all_num12]])[1]<(0.05),
                                                                     fex([[tia_true12, tia_est12], [tia_all_num12, tia_all_num12]])[1]<(0.05/22))})
        
    tia_est23, tia_true23, tia_all_num23 = get_est_true(ts_inverse(scores_for_task[1][:, k]), ts_inverse(scores_for_task[2][:, k]), 0, CI[k])
    stored_fex_tia23.update({'%s (OR, pval, is_sig)' % all_tasks_names[k]: (np.round(fex([[tia_true23, tia_est23], [tia_all_num23, tia_all_num23]])[0], 3),
                                                                 np.round(fex([[tia_true23, tia_est23], [tia_all_num23, tia_all_num23]])[1], 3),
                                                                 fex([[tia_true23, tia_est23], [tia_all_num23, tia_all_num23]])[1]<(0.05),
                                                                 fex([[tia_true23, tia_est23], [tia_all_num23, tia_all_num23]])[1]<(0.05/22))})
    
    tia_est34, tia_true34, tia_all_num34 = get_est_true(ts_inverse(scores_for_task[2][:, k]), ts_inverse(scores_for_task[3][:, k]), 0, CI[k])
    stored_fex_tia34.update({'%s (OR, pval, is_sig)' % all_tasks_names[k]: (np.round(fex([[tia_true34, tia_est34], [tia_all_num34, tia_all_num34]])[0], 3),
                                                                 np.round(fex([[tia_true34, tia_est34], [tia_all_num34, tia_all_num34]])[1], 3),
                                                                 fex([[tia_true34, tia_est34], [tia_all_num34, tia_all_num34]])[1]<(0.05),
                                                                 fex([[tia_true34, tia_est34], [tia_all_num34, tia_all_num34]])[1]<(0.05/22))})

    for i in range(scores_for_task[0].shape[0]): 

        if deltaTask_tia_probs01[count2][i]<(threshold_p_val): #scores_for_task[0][:, k][i]>=ts_convert(ts_inverse(scores_for_task[1][:, k][i])+all_CI[k]-LE[k]):
            
             ax[row, col].plot([0, 1], [scores_for_task[0][:, k][i], scores_for_task[1][:, k][i]], 
                               color='dimgrey', lw=3, marker=mstyle) 
             if np.isnan(scores_for_task[1][:, k][i])==0:
                 less12+=1
                 
        elif deltaTask_tia_probs01[count2][i]>(1-threshold_p_val): #scores_for_task[0][:, k][i]<ts_convert(ts_inverse(scores_for_task[1][:, k][i])-all_CI[k]-LE[k]): 
             ax[row, col].plot([0, 1], [scores_for_task[0][:, k][i], scores_for_task[1][:, k][i]], 
                               color='dimgrey', lw=3, marker=mstyle)
             if np.isnan(scores_for_task[1][:, k][i])==0:
                 more12+=1
                 
        else:
             ax[row, col].plot([0, 1], [scores_for_task[0][:, k][i], scores_for_task[1][:, k][i]], 
                               color='grey', lw=0.25, marker=mstyle, mfc='none')
             if np.isnan(scores_for_task[1][:, k][i])==0:
                 nc12+=1
         
        if deltaTask_tia_probs12[count2][i]<(threshold_p_val): #scores_for_task[1][:, k][i]>=ts_convert(ts_inverse(scores_for_task[2][:, k][i])+all_CI[k]-0):           
             ax[row, col].plot([1, 2], [scores_for_task[1][:, k][i], scores_for_task[2][:, k][i]], 
                               color='dimgrey', lw=3, marker=mstyle) 
             if np.isnan(scores_for_task[2][:, k][i])==0:
                 less23+=1
                 
        elif deltaTask_tia_probs12[count2][i]>(1-threshold_p_val): #scores_for_task[1][:, k][i]<ts_convert(ts_inverse(scores_for_task[2][:, k][i])-all_CI[k]-0):     
             ax[row, col].plot([1, 2], [scores_for_task[1][:, k][i], scores_for_task[2][:, k][i]], 
                               color='dimgrey', lw=3, marker=mstyle) 
             if np.isnan(scores_for_task[2][:, k][i])==0:
                 more23+=1
                 
        else:
             ax[row, col].plot([1, 2], [scores_for_task[1][:, k][i], scores_for_task[2][:, k][i]], 
                               color='grey', lw=0.25, marker=mstyle, mfc='none')             
             if np.isnan(scores_for_task[2][:, k][i])==0:
                 nc23+=1
        
        if deltaTask_tia_probs23[count2][i]<(threshold_p_val): #scores_for_task[2][:, k][i]>=ts_convert(ts_inverse(scores_for_task[3][:, k][i])+all_CI[k]-0):         
             ax[row, col].plot([2, 3], [scores_for_task[2][:, k][i], scores_for_task[3][:, k][i]], 
                               color='dimgrey', lw=3, marker=mstyle)               
             if np.isnan(scores_for_task[3][:, k][i])==0:
                 less34+=1
                 
        elif deltaTask_tia_probs23[count2][i]>(1-threshold_p_val): #scores_for_task[2][:, k][i]<ts_convert(ts_inverse(scores_for_task[3][:, k][i])-all_CI[k]-0):    
             ax[row, col].plot([2, 3], [scores_for_task[2][:, k][i], scores_for_task[3][:, k][i]], 
                               color='dimgrey', lw=3, marker=mstyle)             
             if np.isnan(scores_for_task[3][:, k][i])==0:
                 more34+=1
        else:
             ax[row, col].plot([2, 3], [scores_for_task[2][:, k][i], scores_for_task[3][:, k][i]], 
                               color='grey', lw=0.25, marker=mstyle, mfc='none')
             if np.isnan(scores_for_task[3][:, k][i])==0:
                 nc34+=1
                 
    ax[row, col].set_title(tia_task_names[k], fontsize=16, fontweight='bold')
    
    if count == 0:
        ax[row, col].set_ylabel('Task Score', fontsize=14)
        ax[row, col].set_yticks(np.array((0,2,4,6)))
        ax[row, col].set_yticklabels(np.array((0,2,4,6)), fontsize=12)
        ax[row, col].set_xticks(range(4))
        ax[row, col].set_xticklabels(['2 weeks', '6 weeks', '3 months', '1 year'], fontsize = 12)
    else:     
        ax[row, col].set_ylabel('')
        # ax[row, col].set_yticks(np.array((0,2,4,6)))
        ax[row, col].set_yticklabels('')
        # ax[row, col].set_xticks(range(4))
        ax[row, col].set_xticklabels('')
        
    fsize = 14
    # ax[row, col].text(x=0.15, y=6.5, s='%s' % less12,  ha='left', color='b', fontsize=fsize) #10 is the default fontsize
    # ax[row, col].text(x=0.35, y=6.5, s= '%s' % more12, ha='left', color='r', fontsize=fsize)
    ax[row, col].text(x=0.35, y=6.5, s= '%s' % (less12+more12+nc12), ha='left', color='k', fontsize=fsize)
    # ax[row, col].text(x=0.15, y=6.0, s= 'P = %.2f' % stored_fex_tia12['%s (OR, pval, is_sig)' % all_tasks_names[k]][1], ha='left', color='k', fontsize=10)

    # ax[row, col].text(x=1.15, y=6.5, s= '%s' % less23, ha='left', color='b', fontsize=fsize)
    # ax[row, col].text(x=1.35, y=6.5, s= '%s' % more23, ha='left', color='r', fontsize=fsize)
    ax[row, col].text(x=1.35, y=6.5, s= '%s' % (less23+more23+nc23), ha='left', color='k', fontsize=fsize)
    # ax[row, col].text(x=1.15, y=6.0, s= 'P = %.2f' % stored_fex_tia23['%s (OR, pval, is_sig)' % all_tasks_names[k]][1], ha='left', color='k', fontsize=10)

    # ax[row, col].text(x=2.15, y=6.5, s= '%s' % less34, ha='left', color='b', fontsize=fsize)
    # ax[row, col].text(x=2.35, y=6.5, s= '%s' % more34, ha='left', color='r', fontsize=fsize)
    ax[row, col].text(x=2.35, y=6.5, s= '%s' % (less34+more34+nc34), ha='left', color='k', fontsize=fsize)
    # ax[row, col].text(x=2.15, y=6.0, s= 'P = %.2f' % stored_fex_tia34['%s (OR, pval, is_sig)' % all_tasks_names[k]][1], ha='left', color='k', fontsize=10)
    
    ax[row, col].set_ylim(0,7.5)   
    
    ax[row, col].set_facecolor('white')
    ax[row, col].spines['bottom'].set_edgecolor('black')
    ax[row, col].spines['top'].set_edgecolor('black')
    ax[row, col].spines['left'].set_edgecolor('black')
    ax[row, col].spines['right'].set_edgecolor('black')
    
    col += 1
    count += 1

plt.show()

# %% Stats for 2wk TIA and migraine vs control1 and 1yr TIA and migraine vs control2

from scipy.stats import ttest_ind as tti
from scipy.stats import ttest_1samp as tt1

tti_results, mig_cont_ax1_2 = dict(), dict()

for ind, i in enumerate(all_tasks_names):
    
    tia1_test = tt1(ts_inverse(task_scores_tia_stored[0][:,ind]), popmean = 0, nan_policy = 'omit')
    tia4_test = tt1(ts_inverse(task_scores_tia_stored[3][:,ind]), popmean = 0, nan_policy = 'omit')
    tia_14_test = tt1(ts_inverse(task_scores_tia_stored[0][:,ind]) - ts_inverse(task_scores_tia_stored[3][:,ind]), popmean = 0, nan_policy = 'omit')
    
    mig1_test = tt1(ts_inverse(task_scores_mig_stored[0][:,ind]), popmean = 0, nan_policy = 'omit')
    mig2_test = tt1(ts_inverse(task_scores_mig_stored[1][:,ind]), popmean = 0, nan_policy = 'omit')
    mig_12_test = tt1(ts_inverse(task_scores_mig_stored[0][:,ind]) - ts_inverse(task_scores_mig_stored[1][:,ind]), popmean = 0, nan_policy = 'omit')
    
    tti_results.update({'%s' % i : {'TIA_1 (p, stat, is_sig, is_sig_bonf)' : (tia1_test.pvalue.round(2), tia1_test.statistic.round(2), tia1_test.pvalue<(0.05), tia1_test.pvalue<(p_crit)),
                                    'TIA_4 (p, stat, is_sig, is_sig_bonf)' : (tia4_test.pvalue.round(2), tia4_test.statistic.round(2), tia4_test.pvalue<(0.05), tia4_test.pvalue<(p_crit)),
                                    'TIA_1x4 (p, stat, is_sig, is_sig_bonf)' : (tia_14_test.pvalue.round(2), tia_14_test.statistic.round(2), tia_14_test.pvalue<(0.05), tia_14_test.pvalue<(p_crit)),
                                    'MIG_1 (p, stat, is_sig, is_sig_bonf)' : (mig1_test.pvalue.round(2), mig1_test.statistic.round(2), mig1_test.pvalue<(0.05), mig1_test.pvalue<(p_crit)),
                                    'MIG_2 (p, stat, is_sig, is_sig_bonf)' : (mig2_test.pvalue.round(2), mig2_test.statistic.round(2), mig2_test.pvalue<(0.05), mig2_test.pvalue<(p_crit)),
                                    'MIG_1x2 (p, stat, is_sig, is_sig_bonf)' : (mig_12_test.pvalue.round(2), mig_12_test.statistic.round(2), mig_12_test.pvalue<(0.05), mig_12_test.pvalue<(p_crit))
                                    }
                        })
pd.DataFrame(tti_results).T.to_csv('C:/Users/user/Documents/PhD/TIA/TIA-migraine longitudinal/outputb.csv')

all_p_vals = list()

for k in tti_results.keys():
    for k2 in tti_results[k].keys():
        all_p_vals.append(tti_results[k][k2][0])

p_crit = get_bh_cut(all_p_vals)

# %% CSV t-stat and p-value tables (output.csv and output2.csv)

columns1 = ['Odds', 'P-val', '<0.05', '<0.05/66']
columns2 = ['Stat', 'P-val', '<0.05', '<0.05/66', 'Stat', 'P-val', '<0.05', '<0.05/66']
rows = np.asarray(all_tasks_names)[np.asarray([0,1,2,3,4,5,6,7,8,10])]

new_keys = np.asarray(list(tti_results.keys()))[np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])]
cell_text_1st_ttest = [(np.concatenate((list(tti_results[i]['TIA_4 (p, stat, is_sig, is_sig_bonf)']), 
                                        list(tti_results[i]['MIG_2 (p, stat, is_sig, is_sig_bonf)'])))) for i in new_keys] 
cell_text_2nd_ttest = [(np.concatenate((list(tti_results[i]['TIA_1 (p, stat, is_sig, is_sig_bonf)']), 
                                        list(tti_results[i]['MIG_1 (p, stat, is_sig, is_sig_bonf)'])))) for i in new_keys] 
    
import csv
RESULTS = cell_text_1st_ttest
with open('output1.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(RESULTS)

RESULTS2 = cell_text_2nd_ttest
with open('output2.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(RESULTS2)

# %% RUN LAST: NaN vs 0 indexing

impairment_counts = dict()

for task_ind, task in enumerate(all_tasks_names):
    tia_imp_dict = dict()
    mig_imp_dict = dict()
    
    for tp in range(4):
        tia_im_task = np.nansum(task_scores_tia_stored[tp][:, task_ind]>1.96)
        tia_ct_task = np.sum(np.isnan(task_scores_tia_stored[tp][:, task_ind])==0)
        tia_pr_task = tia_im_task/tia_ct_task
        
        if (tp==0):
            mig_im_task = np.nansum(task_scores_mig_stored[0][:, task_ind]>1.96)
            mig_ct_task = np.sum(np.isnan(task_scores_mig_stored[0][:, task_ind])==0)
            mig_pr_task = mig_im_task/mig_ct_task
        elif(tp==3): 
            mig_im_task = np.nansum(task_scores_mig_stored[1][:, task_ind]>1.96)
            mig_ct_task = np.sum(np.isnan(task_scores_mig_stored[1][:, task_ind])==0)
            mig_pr_task = mig_im_task/mig_ct_task
        else:
            mig_im_task = np.nan
            mig_ct_task = np.nan
            mig_pr_task = np.nan
            
        tia_imp_dict.update({'TIA_%i (ct, imp, prc)' % tp: (tia_im_task, tia_ct_task, 100*np.round(tia_pr_task, 3))})
        mig_imp_dict.update({'MIG_%i (ct, imp, prc)' % tp: (mig_im_task, mig_ct_task, 100*np.round(mig_pr_task, 3))})
        
    impairment_counts.update({task: (tia_imp_dict, mig_imp_dict)})




    
