#!/usr/bin/env python3
"""Generate application-layer vector figures from official audited rows only."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

ap=argparse.ArgumentParser();ap.add_argument('--application-data',default='application_data');ap.add_argument('--transfer-results',required=True);ap.add_argument('--out',default='application_figures');args=ap.parse_args()
ADD=Path(args.application_data);TRANSFER=Path(args.transfer_results);OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
# Bright, colorblind-oriented line palette: blue / amber / green.
COL={'field_like_stellar':'#0072B2','compact_object_newtonian':'#E69F00','young_cluster_like_stellar':'#009E73'}
MARK={'field_like_stellar':'o','compact_object_newtonian':'s','young_cluster_like_stellar':'^'}
LAB={'field_like_stellar':'Field-like','compact_object_newtonian':'Compact Newtonian','young_cluster_like_stellar':'Young-cluster-like'}
POP_ORDER=list(LAB)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif','font.size':9.4,'axes.labelsize':10.2,'axes.titlesize':10.4,'axes.titlepad':10,'legend.fontsize':7.5,'xtick.labelsize':8.7,'ytick.labelsize':8.7,'axes.linewidth':.9,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
 fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/f'{name}.png',dpi=190,bbox_inches='tight');plt.close(fig)
def finish(ax,xlabel=None,ylabel=None,title=None):
 if xlabel:ax.set_xlabel(xlabel,labelpad=5)
 if ylabel:ax.set_ylabel(ylabel,labelpad=6)
 if title:ax.set_title(title,pad=10)
 ax.grid(alpha=.20,zorder=0)

tr=json.load(open(TRANSFER));absrows=list(csv.DictReader(open(ADD/'ABSOLUTE_METRIC_INTERVALS.csv')));paired=list(csv.DictReader(open(ADD/'PAIRED_BASELINE_INTERVALS.csv')));env=list(csv.DictReader(open(ADD/'SENSITIVITY_ENVELOPES.csv')))

# Figure 1 — protocol schematic.
fig,ax=plt.subplots(figsize=(7.25,5.15));ax.set_xlim(0,10);ax.set_ylim(0,8.2);ax.axis('off')
def box(x,y,w,h,text,fc='#F3F6F8',ec='#334E68',fs=8.5):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',fc=fc,ec=ec,lw=1.15);ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs);return p
def arrow(x1,y1,x2,y2,color='#52606D'):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,lw=1.05,color=color))
box(3.2,7.15,3.6,.65,'Frozen source arrays + protocol\n(two deterministic reconstructions)',fc='#E8F1FA')
box(3.2,6.05,3.6,.58,'15/15 model, threshold, and\nsource-score hashes match',fc='#E7F5EE',ec='#16794B');arrow(5,7.15,5,6.63)
for x,t,c in [(0.25,'Field-like\nengineering proxy','#DCEEFF'),(3.75,'Compact-object-Newtonian\nengineering proxy','#FCE8DD'),(7.25,'Young-cluster-like\nengineering proxy','#DDF4E8')]:box(x,4.75,2.5,.65,t,fc=c);arrow(5,6.05,x+1.25,5.4)
box(3.1,3.65,3.8,.55,'For each target: fixed split roles + frozen source scores')
for x in [1.5,5,8.5]:arrow(x,4.75,5,4.2)
for x,t,c in [(0.25,'Zero-shot\nsource model + source cut','#EEF4FA'),(3.75,'Target-calibrated\nfrozen model + target cut','#FFF6D9'),(7.25,'Target-refit control\nfixed family/hyperparameters','#FFF6D9')]:box(x,2.35,2.5,.65,t,fc=c);arrow(5,3.65,x+1.25,3.0)
box(3.2,1.15,3.6,.58,'Untouched target test scoring\nAUROC, PR-AUC, operation, sensitivity',fc='#F3E8FF',ec='#6B3FA0')
for x in [1.5,5,8.5]:arrow(x,2.35,5,1.73)
ax.text(5,.50,'Leakage boundary: test IDs enter scoring only—never fitting or threshold selection',ha='center',va='center',fontsize=8.8,fontweight='bold',color='#6B3FA0')
fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.03);save(fig,'figure1_design')

# Figure 2 — ranking dashboard.
fig,axs=plt.subplots(2,2,figsize=(7.35,6.45));axs=axs.ravel()
for pop in POP_ORDER:
 rr=sorted([r for r in absrows if r['population']==pop and r['mode']=='zero_shot' and r['arm']=='combined' and float(r['horizon_Tout'])<=150],key=lambda r:float(r['horizon_Tout']));x=np.array([float(r['horizon_Tout']) for r in rr])
 for ax,key,lo,hi in [(axs[0],'roc_auc','roc_lo','roc_hi'),(axs[1],'pr_auc','pr_lo','pr_hi')]:
  y=np.array([float(r[key]) for r in rr]);yl=np.array([float(r[lo]) for r in rr]);yh=np.array([float(r[hi]) for r in rr]);ax.errorbar(x,y,yerr=[y-yl,yh-y],color=COL[pop],marker=MARK[pop],lw=1.7,ms=5.6,capsize=2.8,label=LAB[pop],zorder=3)
 hs=tr[pop]['horizons'][:4];xx=np.array([300*float(q['f']) for q in hs]);g=np.array([float(q['zero_shot']['combined_minus_IC']['point']) for q in hs]);gl=np.array([float(q['zero_shot']['combined_minus_IC']['lo']) for q in hs]);gh=np.array([float(q['zero_shot']['combined_minus_IC']['hi']) for q in hs]);axs[2].errorbar(xx,g,yerr=[g-gl,gh-g],color=COL[pop],marker=MARK[pop],lw=1.7,ms=5.6,capsize=2.8,zorder=3)
 prev=np.array([100*float(q['zero_shot']['combined']['metrics']['prevalence']) for q in hs]);axs[3].plot(xx,prev,color=COL[pop],marker=MARK[pop],lw=1.7,ms=5.6,zorder=3)
finish(axs[0],r'Observation horizon [$T_{\rm out}$]','AUROC','(a) Zero-shot ranking');axs[0].set_ylim(.90,1.005)
finish(axs[1],r'Observation horizon [$T_{\rm out}$]','PR-AUC','(b) Precision–recall performance');axs[1].set_ylim(.15,.98)
axs[2].axhline(0,color='.35',lw=.9,ls=':',zorder=1);finish(axs[2],r'Observation horizon [$T_{\rm out}$]',r'$\Delta$AUROC (combined $-$ IC)','(c) Increment from trajectory history')
finish(axs[3],r'Observation horizon [$T_{\rm out}$]','Future-ejection prevalence [%]','(d) Event support in the test risk set')
handles=[Line2D([0],[0],color=COL[p],marker=MARK[p],lw=1.7,label=LAB[p]) for p in POP_ORDER]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015));fig.subplots_adjust(left=.10,right=.98,top=.94,bottom=.14,hspace=.48,wspace=.36);save(fig,'figure2_transfer_performance')

# Figure 3 — paired external-baseline forest plot.
base_order=['MA01_margin','V22_formula','V22_MLP','V23_MLP'];base_label={'MA01_margin':'MA01 margin','V22_formula':'V22 formula','V22_MLP':'V22 MLP','V23_MLP':'V23 MLP'}
horiz=[15,45,90,150];hcol=['#0072B2','#E69F00','#009E73','#CC79A7'];hmark=['o','s','^','D'];offset=np.array([-.24,-.08,.08,.24])
fig,axs=plt.subplots(1,3,figsize=(7.4,3.95),sharex=False,sharey=True)
for ax,pop in zip(axs,POP_ORDER):
 sub=[r for r in paired if r['population']==pop and float(r['horizon_Tout'])<=150]
 for hi,h in enumerate(horiz):
  for bi,b in enumerate(base_order):
   r=next(x for x in sub if float(x['horizon_Tout'])==h and x['comparison'].endswith('_'+b));point=float(r['roc_point']);lo=float(r['roc_lo']);up=float(r['roc_hi']);y=bi+offset[hi]
   ax.errorbar(point,y,xerr=[[point-lo],[up-point]],fmt=hmark[hi],color=hcol[hi],ms=4.5,lw=1.1,capsize=2,zorder=3)
 ax.axvline(0,color='.3',lw=.9,ls=':');ax.set_yticks(range(4),[base_label[b] for b in base_order]);ax.invert_yaxis();finish(ax,None,None,LAB[pop]);ax.tick_params(axis='y',labelsize=8)
handles=[Line2D([0],[0],marker=hmark[i],color=hcol[i],lw=0,label=rf'${h}\,T_{{\rm out}}$') for i,h in enumerate(horiz)]
fig.supxlabel(r'$\Delta$AUROC (combined $-$ baseline)',y=.105,fontsize=10.2)
fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.012));fig.subplots_adjust(left=.17,right=.99,top=.88,bottom=.24,wspace=.28);save(fig,'figure3_baseline_forest')

# Figure 4 — threshold transport, warning operation, lead, and status composition.
fig,axs=plt.subplots(2,2,figsize=(7.35,6.55));axs=axs.ravel()
for pop in POP_ORDER:
 hs=tr[pop]['horizons'][:4];x=np.array([300*float(q['f']) for q in hs]);src=np.array([100*float(q['zero_shot']['combined']['source_threshold_operation']['0.01']['achieved_far']) for q in hs]);cal=np.array([100*float(q['zero_shot']['combined']['target_calibrated_operation']['0.01']['achieved_far']) for q in hs])
 axs[0].plot(x,src,color=COL[pop],marker=MARK[pop],lw=1.6,ms=5.5,zorder=3);axs[0].plot(x,cal,color=COL[pop],marker=MARK[pop],lw=1.15,ls='--',mfc='white',ms=5.5,zorder=3)
 w=tr[pop]['warning_operation'];modes=[('zero_shot_source_threshold_combined_1pct','Source'),('zero_shot_target_calibrated_combined_1pct','Target-cal.'),('target_refit_combined_1pct','Refit')];mks=['o','s','D'];xs=[];ys=[]
 for (key,label),mk in zip(modes,mks):
  q=w[key];xs.append(100*q['cumulative_control_alert_fraction']);ys.append(100*q['coverage']);axs[1].scatter(xs[-1],ys[-1],s=52,marker=mk,color=COL[pop],edgecolor='white',linewidth=.7,zorder=4)
 axs[1].plot(xs,ys,color=COL[pop],lw=1.1,alpha=.72,zorder=2)
# lead grouped bars by target and mode
mode_keys=[('zero_shot_source_threshold_combined_1pct','Source','#4C78A8'),('zero_shot_target_calibrated_combined_1pct','Target-cal.','#F58518'),('target_refit_combined_1pct','Refit','#54A24B')];x0=np.arange(3);width=.23
for j,(key,label,color) in enumerate(mode_keys):
 vals=[tr[p]['warning_operation'][key]['median_lead_Tout'] for p in POP_ORDER];axs[2].bar(x0+(j-1)*width,vals,width,color=color,label=label,edgecolor='white',linewidth=.5,zorder=3)
# status composition
statuses=['stable','ejected','collision','numerical_error','worker_error'];slab=['Stable','Ejected','Collision','Numerical','Worker'];scol=['#4C78A8','#F58518','#E45756','#B279A2','#79706E'];bottom=np.zeros(3)
for st,la,co in zip(statuses,slab,scol):
 vals=np.array([100*tr[p]['status_counts'].get(st,0)/10000 for p in POP_ORDER]);axs[3].bar(np.arange(3),vals,bottom=bottom,color=co,label=la,width=.68,edgecolor='white',linewidth=.45,zorder=3);bottom+=vals
axs[0].axhline(1,color='.3',lw=.9,ls=':');axs[0].set_yscale('log');axs[0].set_ylim(.035,7);finish(axs[0],r'Observation horizon [$T_{\rm out}$]','Achieved test FAR [%]','(a) Threshold transport')
finish(axs[1],'Cumulative controls alerted [%]','Eligible-event coverage [%]','(b) Five-landmark warning operation')
axs[2].set_xticks(x0,['Field','Compact','Young']);finish(axs[2],None,r'Median lead [$T_{\rm out}$]','(c) Lead at first warning');axs[2].legend(frameon=False,ncol=3,loc='upper center',fontsize=6.8)
axs[3].set_xticks(np.arange(3),['Field','Compact','Young']);axs[3].set_ylim(0,100);finish(axs[3],None,'Fraction of target ensemble [%]','(d) Terminal-status composition');axs[3].legend(frameon=False,ncol=2,fontsize=6.7,loc='lower left')
target_handles=[Line2D([0],[0],color=COL[p],marker=MARK[p],lw=1.6,label=LAB[p]) for p in POP_ORDER];style_handles=[Line2D([0],[0],color='.3',lw=1.6,label='Source cut'),Line2D([0],[0],color='.3',lw=1.2,ls='--',mfc='white',marker='o',label='Target-calibrated cut')]
fig.legend(handles=target_handles+style_handles,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.012),fontsize=6.8);fig.subplots_adjust(left=.10,right=.98,top=.94,bottom=.14,hspace=.49,wspace=.36);save(fig,'figure4_threshold_operation')

# Figure 5 — dedicated sensitivity envelopes, AUROC and PR-AUC.
fig,axs=plt.subplots(2,3,figsize=(7.45,5.45),sharex='col')
for ci,pop in enumerate(POP_ORDER):
 rr=sorted([r for r in env if r['population']==pop and float(r['horizon_Tout'])<=150],key=lambda r:float(r['horizon_Tout']));x=np.array([float(r['horizon_Tout']) for r in rr])
 absr=sorted([r for r in absrows if r['population']==pop and r['mode']=='zero_shot' and r['arm']=='combined' and float(r['horizon_Tout'])<=150],key=lambda r:float(r['horizon_Tout']))
 for ri,(pointkey,lokey,hikey,ylabel) in enumerate([('roc_auc','scoreable_scenario_roc_min','scoreable_scenario_roc_max','AUROC'),('pr_auc','scoreable_scenario_pr_min','scoreable_scenario_pr_max','PR-AUC')]):
  ax=axs[ri,ci];lo=np.array([float(r[lokey]) for r in rr]);hi=np.array([float(r[hikey]) for r in rr]);point=np.array([float(r[pointkey]) for r in absr]);ax.fill_between(x,lo,hi,color=COL[pop],alpha=.22,zorder=1);ax.plot(x,point,color=COL[pop],marker=MARK[pop],lw=1.75,ms=5.3,zorder=3);finish(ax,r'Observation horizon [$T_{\rm out}$]' if ri==1 else None,ylabel if ci==0 else None,LAB[pop] if ri==0 else None)
  if ri==0:ax.set_ylim(.65,1.01)
  else:ax.set_ylim(0,1.0)
handles=[Line2D([0],[0],color='.2',marker='o',lw=1.7,label='Primary valid cohort'),Patch(facecolor='.5',alpha=.22,label='Scoreable stress envelope')]
fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.012));fig.subplots_adjust(left=.09,right=.99,top=.91,bottom=.16,hspace=.24,wspace=.28);save(fig,'figure5_sensitivity_envelopes')
print('wrote five publication-grade application figures')
