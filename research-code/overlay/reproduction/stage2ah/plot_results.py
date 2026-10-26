"""Local lightweight plotting of verified aggregate results only."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    source=a.record/'verified-execution-v1/results-v1'
    verification=json.loads((source/'independent-verification.json').read_text())
    assert verification['status']=='PASS', 'Actual independent results required'
    table=json.loads((source/'endpoint-summary.json').read_text())
    pairs=json.loads((source/'paired-comparisons.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'aligned':'#0077BB','shuffled':'#EE7733','fixed_lce':'#009988','lm_only':'#AA4499','cabg_lce':'#777777'}
    labels={'aligned':'Aligned transport','shuffled':'Shuffled transport','fixed_lce':'Fixed LCE','lm_only':'LM-only','cabg_lce':'CABG-LCE'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                        'savefig.dpi':450,'svg.fonttype':'none','font.family':'DejaVu Sans'})
    out=a.record/'figures';out.mkdir(exist_ok=True)
    lookup={r['endpoint']:r for r in table}
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.5),sharey=True,layout='constrained')
    for ax,seed in zip(axes,[43,44]):
        for arm in colors:
            steps=[24,100,300,1000]
            values=[lookup[f's{seed}-{arm}-step{s:04}-development']['balanced_accuracy']*100 for s in steps]
            ax.plot(steps,values,marker='o',markersize=3,lw=1.4,color=colors[arm],label=labels[arm])
        ax.set(xlabel='Optimizer updates',title=f'Seed {seed}',ylim=(40,100))
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Generated-answer balanced accuracy (%)')
    axes[1].legend(fontsize=7,loc='lower right')
    fig.suptitle('Repeated development set: 128 images (64 negative / 64 positive)',fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'development-curves.{ext}',bbox_inches='tight',dpi=450)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.5),sharex=True,layout='constrained')
    comparisons=[('aligned','fixed_lce'),('aligned','shuffled'),('aligned','lm_only'),('aligned','cabg_lce'),('shuffled','fixed_lce')]
    for ax,seed in zip(axes,[43,44]):
        for i,(left,right) in enumerate(comparisons):
            suffix='-step1000-final'
            row=pairs[f's{seed}-{left}'+suffix+'__minus__'+f's{seed}-{right}'+suffix]
            value=row['delta_balanced_accuracy']*100;low,high=[v*100 for v in row['paired_balanced_accuracy_95ci']]
            ax.plot([low,high],[i,i],color=colors[left],lw=1.5)
            ax.plot(value,i,'o',color=colors[left],ms=4)
        ax.axvline(0,color='#777777',lw=.8,ls='--')
        ax.set_yticks(range(len(comparisons)),[labels[l]+' − '+labels[r] for l,r in comparisons],fontsize=7)
        ax.invert_yaxis();ax.set(xlabel='BA difference (percentage points)',title=f'Seed {seed}')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Reused-test follow-up: 542 images; descriptive paired image 95% intervals',fontsize=10)
    for ext in ['png','svg']:fig.savefig(out/f'final-paired-comparisons.{ext}',bbox_inches='tight',dpi=450)
    plt.close(fig)
    manifest=['# Figure data manifest','','Real observed results only. Repeated development/reused-test images; no patient-level independence or seed-population inference. No synthetic values.','',
              '| Figure | Data file | Real/mock | Source | Script | Outputs |','|---|---|---|---|---|---|']
    for figure,data in [('development-curves','endpoint-summary.json'),('final-paired-comparisons','paired-comparisons.json')]:
        digest=hashlib.sha256((source/data).read_bytes()).hexdigest()
        manifest.append(f'| {figure} | ../verified-execution-v1/results-v1/{data} | Real | stage2ah plus verified stage2af; SHA256 {digest} | Medic-AD/reproduction/stage2ah/plot_results.py | {figure}.png / .svg |')
    (out/'data-manifest.md').write_text('\n'.join(manifest)+'\n')
    print(json.dumps(dict(figures=4,real_data_only=True,directory=str(out))))


if __name__=='__main__':main()
