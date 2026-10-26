"""Render verified real endpoint tables only; no synthetic or imputed results."""
import argparse,hashlib,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'lm_only':'#0077BB','fixed_lce':'#EE7733','cabg_lce':'#009988'}
LABELS={'lm_only':'LM only','fixed_lce':'Fixed LCE','cabg_lce':'CABG-LCE'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    verification=json.loads((a.results/'independent-verification.json').read_text());assert verification['status']=='PASS'
    rows=json.loads((a.results/'endpoint-summary.json').read_text());pairs=json.loads((a.results/'paired-comparisons.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
    records={name:hashlib.sha256((a.results/name).read_bytes()).hexdigest() for name in ['endpoint-summary.json','paired-comparisons.json','independent-verification.json']}
    (a.output/'data-manifest.json').write_text(json.dumps({'kind':'real verified results','sources':records,'missing_endpoints':'omitted only when explicitly terminal/failed; never imputed','outputs':['development-curves.png','development-curves.svg','final-paired-comparisons.png','final-paired-comparisons.svg']},indent=2)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'legend.frameon':False,'svg.fonttype':'none','savefig.dpi':450})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),sharex=True)
    for col,seed in enumerate([43,44]):
        for arm,color in COLORS.items():
            selected=[]
            for row in rows:
                match=re.fullmatch(f's{seed}-{arm}-step([0-9]+)-development',row['endpoint'])
                if match:selected.append((int(match[1]),row))
            selected.sort()
            for line,key in enumerate(['balanced_accuracy','auroc']):
                axes[line,col].plot([x for x,r in selected],[100*r[key] for x,r in selected],'-o',color=color,label=LABELS[arm],lw=1.6,ms=4)
        axes[0,col].set_title(f'Retained seed {seed}')
        axes[1,col].set_xlabel('Training updates (cosine horizon 1000)')
        for ax in axes[:,col]:ax.set_xscale('log');ax.set_xticks([24,100,300,1000],['24','100','300','1000']);ax.set_ylim(0,102)
    axes[0,0].set_ylabel('Generated balanced accuracy (%)');axes[1,0].set_ylabel('Anomaly score AUROC (%)')
    axes[0,0].legend(loc='lower right',fontsize=8);fig.suptitle('Fixed development checkpoints — 128 ChestX-Det images',fontsize=11)
    fig.text(.5,.015,'Reused development observations; image-level grouping; source-family pretraining exposure known.',ha='center',fontsize=7)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(a.output/'development-curves.png');fig.savefig(a.output/'development-curves.svg');plt.close(fig)
    labels=[];points=[];limits=[];zero_discordance=False
    for seed in [43,44]:
        for comparator in ['lm_only','fixed_lce']:
            key=f's{seed}-cabg_lce-step1000-final__minus__s{seed}-{comparator}-step1000-final'
            if key not in pairs:continue
            r=pairs[key];degenerate=r.get('bootstrap_degenerate_zero_discordance',False);zero_discordance|=degenerate
            labels.append(f'Seed {seed}: CABG − {LABELS[comparator]}' + (' †' if degenerate else ''));points.append(r['delta_balanced_accuracy']*100);limits.append([x*100 for x in r['paired_balanced_accuracy_95ci']])
    fig,ax=plt.subplots(figsize=(7.2,3.4))
    for i,(label,value,interval) in enumerate(zip(labels,points,limits)):
        ax.plot(interval,[i,i],color='#0077BB',lw=2);ax.scatter(value,i,color='#0077BB',s=32,zorder=3)
        ax.annotate(f'{value:+.2f} pp',(value,i),xytext=(5,7),textcoords='offset points',fontsize=8)
    ax.axvline(0,color='#666666',lw=.8);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('Generated balanced accuracy difference (percentage points)')
    ax.set_title('Step 1000 final-set paired comparisons',fontsize=11)
    fig.text(.5,.04,'Unadjusted paired image-bootstrap 95% intervals; no patient or pretraining-clean independence claim.',ha='center',fontsize=7)
    if zero_discordance:fig.text(.5,.008,'† No observed discordance: the empirical interval collapses; this does not establish equivalence.',ha='center',fontsize=7)
    if not labels:ax.text(.5,.5,'No complete paired final endpoints available',ha='center',transform=ax.transAxes)
    fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(a.output/'final-paired-comparisons.png');fig.savefig(a.output/'final-paired-comparisons.svg');plt.close(fig)
if __name__=='__main__':main()
