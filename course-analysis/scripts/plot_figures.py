"""Publication assets from analysis tables. English figures, Chinese companion guide."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1];TAB=ROOT/'tables';FIG=ROOT/'figures';FD=FIG/'data'
C={'lm_only':'#778397','fixed_lce':'#00867D','cabg_lce':'#D58B27','linear_head':'#414A58',
   'aligned':'#3478B3','shuffled':'#9465A6','negative':'#557BA9','positive':'#D18C35'}
N={'lm_only':'LM-only','fixed_lce':'Fixed LCE-MIL','cabg_lce':'CABG-LCE-MIL','linear_head':'Linear head',
   'aligned':'Aligned transport','shuffled':'Shuffled transport'}
ARMS=['lm_only','fixed_lce','cabg_lce'];FOUR=ARMS+['linear_head']
CMAP=LinearSegmentedColormap.from_list('ink_teal',['#F2F6F7','#BDDCD8','#168A82','#09594F'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12.5,'axes.labelsize':11,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B7C0CA','axes.labelcolor':'#273449',
 'xtick.color':'#475569','ytick.color':'#475569','text.color':'#17283E','axes.titleweight':'bold',
 'legend.frameon':False,'legend.fontsize':9,'savefig.dpi':450,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','lines.linewidth':2,'axes.axisbelow':True})
CAT=[];BOOK=None

def read(name): return pd.read_csv(TAB/name)
def grid(ax,axis='y'):ax.grid(axis=axis,color='#DCE3E9',linewidth=.65,alpha=.8)
def canvas(rows=1,cols=2,size=(10.5,5.8),left=.095,bottom=.18,top=.80,hspace=.62,wspace=.32):
    fig,axs=plt.subplots(rows,cols,figsize=size,squeeze=False)
    fig.subplots_adjust(left=left,right=.975,bottom=bottom,top=top,hspace=hspace,wspace=wspace)
    return fig,axs
def meta(fig,title,subtitle,note):
    fig.suptitle(title,x=.055,y=.969,ha='left',fontsize=17,fontweight='bold')
    fig.text(.055,.905,subtitle,ha='left',va='top',fontsize=10.5,color='#56667A')
    fig.text(.055,.045,note,ha='left',va='bottom',fontsize=8.7,color='#56667A',linespacing=1.45)
def ep(seed,method,split='final',step=1000):
    return 'linear-head-'+split if method=='linear_head' else f's{seed}-{method}-step{step:04d}-{split}'
def publish(fig,num,slug,title_cn,question,finding,limits,placement,tables,caption,priority='正文'):
    stem=f'F{num:02d}_{slug}';dat=[]
    for text in fig.findobj(matplotlib.text.Text):
        text.set_text(re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text.get_text()))
    fig.canvas.draw()
    for text in fig.texts:
        while text.get_window_extent(fig.canvas.get_renderer()).x1 > fig.bbox.width*.985 and text.get_fontsize()>8:
            text.set_fontsize(text.get_fontsize()-.5)
            fig.canvas.draw()
    for name in tables:
        df=read(name);out=FD/(stem+'__'+name);df.to_csv(out,index=False,float_format='%.15g');dat.append(str(out.relative_to(ROOT)))
    for ext in ('png','svg','pdf'):
        fig.savefig(FIG/(stem+'.'+ext),dpi=450,facecolor='white')
    BOOK.savefig(fig,facecolor='white')
    plt.close(fig)
    CAT.append({'id':f'F{num:02d}','stem':stem,'title_cn':title_cn,'question':question,'finding':finding,
                'limits':limits,'placement':placement,'priority':priority,'tables':tables,'data':dat,
                'caption_en':caption,'outputs':[f'figures/{stem}.{x}' for x in ('png','svg','pdf')]})

def f01():
    d=read('04_cohort_counts.csv');fig,aa=canvas(size=(10.5,5.9),left=.145,bottom=.24,wspace=.50);a,b=aa[0]
    spl=['train','development','final'];labs=['Train','Development','Final source']
    mat=np.array([[d[(d.split==s)&(d['class']==c)].n.iloc[0] for c in (0,1)] for s in spl])
    for k,ax in enumerate((a,b)):
        dd=mat if k==0 else mat/mat.sum(1,keepdims=True)*100
        left=np.zeros(3)
        for c,col in enumerate(['negative','positive']):
            ax.barh(np.arange(3),dd[:,c],left=left,color=C[col],height=.55,label=['No listed finding','Listed finding'][c])
            for i in range(3):
                text=str(mat[i,c]) if k==0 else f'{dd[i,c]:.1f}%'
                ax.text(left[i]+dd[i,c]/2,i,text,ha='center',va='center',color='white',fontsize=10)
            left+=dd[:,c]
        ax.set_yticks(range(3),labs);ax.invert_yaxis();grid(ax,'x');ax.set_xlabel('Images' if k==0 else 'Within-split composition (%)')
        ax.set_xlim(0,1050 if k==0 else 100);ax.set_title(['A  Image counts','B  Class composition'][k],loc='left')
    fig.legend(*a.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.125),ncol=2,fontsize=9)
    meta(fig,'Cohort composition sets the evaluation context','1,670 selected images; repeated model readouts do not create new cases.',
         'Final: 64 negative / 478 positive images. Constant Yes: accuracy 88.19%, balanced accuracy 50%.\nNo listed finding does not imply clinical health; no patient mapping; final cohort reused in transport follow-up.')
    publish(fig,1,'cohort_composition','数据组成与指标选择','为什么主回答指标必须考虑类别不平衡？',
      '训练1000、开发128、最终542张；最终阳性占88.19%。全部回答Yes也有88.19% accuracy，但BA仅50%。',
      '计数单位为图像；没有患者映射。最终来源图在后续传递研究中复用；不能称独立新队列。',
      'Data and experimental design；汇报的实验设置页',['04_cohort_counts.csv'],
      'Image counts and within-split class composition of the accepted ChestX-Det subset. Labels denote presence of any listed finding. The constant-Yes reference illustrates why accuracy alone is misleading; it is an analytic reference, not an additional model run.')

def f02():
    d=read('05_scale_final_with_posthoc_intervals.csv').set_index('endpoint');fig,aa=canvas(left=.20,wspace=.60)
    for j,seed in enumerate((43,44)):
        ax=aa[0,j]
        for i,m in enumerate(FOUR):
            r=d.loc[ep(seed,m)];x=r.auroc;lo=r.auroc_ci_low;hi=r.auroc_ci_high
            ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o' if m!='linear_head' else 's',color=C[m],capsize=3,ms=7,lw=2)
            ax.text(.995,i+.24,f'{x:.4f}  [{lo:.3f}, {hi:.3f}]',ha='right',va='center',fontsize=8.2,color=C[m])
        ax.set_yticks(range(4),[N[m] for m in FOUR]);ax.invert_yaxis();ax.set_ylim(3.6,-.7)
        ax.axvline(.5,color='#A8B1BE',ls=':',lw=1.2);ax.set_xlim(.2,1);ax.set_xlabel('AUROC');grid(ax,'x');ax.set_title(f'Seed {seed}',loc='left')
    meta(fig,'LCE-MIL improves auxiliary-score discrimination','Fixed step1000 · 542 final-source images · both retained seeds',
      'Dots: accepted estimates. Bars: new post-hoc 95% stratified image-bootstrap intervals (20,000 draws).\nLinear head is one fitted classifier, shown in both panels as the same reference. No patient-level inference.')
    publish(fig,2,'final_auroc','LCE-MIL 的核心异常排序成果','图像级辅助监督有没有改善异常分数的区分能力？',
      '固定LCE最终AUROC为0.8005/0.8037，高于对应LM-only的0.6033/0.3619；CABG-LCE同样约0.80。',
      '这是辅助异常分数的排序，非生成回答准确率。线性头0.7850是重要参照；区间仅为给定模型下的图像重采样。',
      'Main results；摘要的主要数字；汇报核心成果页',['05_scale_final_with_posthoc_intervals.csv'],
      'Final auxiliary-score AUROC for all three matched training arms at step1000 and the frozen-feature linear classifier. Point estimates are reproduced from accepted records; bars are newly computed post-hoc 95% class-stratified image-bootstrap intervals. The same single classifier is repeated as a reference in both panels.')

def f03():
    d=read('06_auroc_paired_posthoc.csv');fig,aa=canvas(left=.25,wspace=.72,size=(11.2,6.1))
    for j,seed in enumerate((43,44)):
        ax=aa[0,j];g=d[d.seed==seed].reset_index(drop=True)
        for i,r in g.iterrows():
            col=C['fixed_lce'] if r.comparison.startswith('Fixed') else C['cabg_lce']
            ax.plot([r.family8_ci_low,r.family8_ci_high],[i,i],color=col,alpha=.28,lw=7,solid_capstyle='round')
            ax.plot([r.ci95_low,r.ci95_high],[i,i],color=col,lw=2.5);ax.scatter(r.delta_auroc,i,c=col,s=45,zorder=3)
            ax.text(.645,i+.23,f'{r.delta_auroc:+.3f}',ha='right',fontsize=9,color=col)
        ax.set_yticks(range(len(g)),g.comparison.str.replace('Fixed LCE','Fixed LCE',regex=False));ax.set_ylim(3.55,-.6)
        ax.axvline(0,color='#586675',lw=1);ax.set_xlim(-.11,.66);grid(ax,'x');ax.set_xlabel('Paired AUROC difference');ax.set_title(f'Seed {seed}',loc='left')
    meta(fig,'The ranking gain depends on the comparator','All eight post-hoc contrasts retained; pairing uses the same images across models.',
      'Thin dark bars: 95% intervals. Wide pale bars: 99.375% Bonferroni-level sensitivity intervals for a family of8.\nApproximate bootstrap coverage; no adjustment for patient clustering, historical selection or seed-population uncertainty.')
    publish(fig,3,'auroc_paired_intervals','排序增量与多重比较敏感性','LCE-MIL 相对不同参照的优势有多大、不确定性如何？',
      '固定LCE−LM的AUROC差为+0.1972/+0.4418；95%及八项比较的保守敏感性区间均高于0。固定LCE−线性头、CABG−固定LCE的区间均跨0。',
      '新加的事后分析，不能改写原BA主终点。99.375%区间是近似的多重比较敏感性分析，不处理所有选择偏差。',
      'Main results；最有力的贡献证据页',['06_auroc_paired_posthoc.csv'],
      'Paired AUROC differences for the eight specified post-hoc comparisons. Dark intervals are 95% percentile intervals; pale outer intervals use the Bonferroni level of99.375% for eight contrasts. Stratified resampling preserves negative/positive counts and image pairing. All contrasts are shown, including inconclusive comparisons with the classifier and between weighting schemes.')

def f04():
    d=read('08_roc_pr_coordinates.csv');point=read('05_scale_final_with_posthoc_intervals.csv').set_index('endpoint')
    fig,aa=canvas(2,2,size=(10.5,8.2),left=.10,bottom=.14,top=.82,hspace=.52,wspace=.30)
    for j,seed in enumerate((43,44)):
        for i,curve in enumerate(('ROC','PR')):
            ax=aa[i,j]
            for m in FOUR:
                endpoint=ep(seed,m);g=d[(d.endpoint==endpoint)&(d.curve==curve)]
                metric=point.loc[endpoint,'auroc' if i==0 else 'average_precision']
                ax.plot(g.x,g.y,color=C[m],lw=2.4 if m=='fixed_lce' else 1.5,ls='--' if m=='linear_head' else '-',label=f'{N[m]} ({metric:.3f})')
            if i==0: ax.plot([0,1],[0,1],ls=':',color='#A7B1BE',lw=1)
            else:ax.axhline(478/542,ls=':',color='#A7B1BE',lw=1)
            ax.set(xlim=(0,1),ylim=(0,1.025),xlabel='False-positive rate' if i==0 else 'Recall',ylabel='True-positive rate' if i==0 else 'Precision')
            ax.set_title(f'{curve} · Seed {seed}',loc='left');grid(ax);ax.legend(loc='lower right' if i==0 else 'lower left',fontsize=7.8)
    meta(fig,'Ranking performance across operating points','Curves use saved scores only; no threshold was selected or deployed.',
      'Legend values: AUROC (top), non-interpolated AP (bottom). PR reference prevalence = 478/542 = 0.882.\nA high AP partly reflects class composition. The head is a classifier; neural curves use auxiliary anomaly scores.')
    publish(fig,4,'roc_pr_curves','ROC 与 Precision–Recall 曲线','排序差异是否能从整条曲线中看见？',
      '两个种子的LCE曲线均优于相应未加辅助监督的异常分数；固定LCE、CABG与线性头之间更接近。',
      'PR受阳性比例影响；曲线不表示已经选出可部署阈值，也不是患者独立泛化证明。',
      'Ranking analysis；详细报告可用，短汇报选ROC部分',['08_roc_pr_coordinates.csv','05_scale_final_with_posthoc_intervals.csv'],
      'ROC and precision–recall curves for the final-source cohort. All coordinates are calculated from the saved continuous scores. Dotted references indicate chance ROC and observed positive prevalence for PR. AP is the non-interpolated average precision, not trapezoidal PR area.')

def f05():
    d=read('07_scale_development_all_nodes.csv');fig,aa=canvas(2,2,size=(10.5,8),bottom=.22,top=.82,hspace=.55)
    for j,seed in enumerate((43,44)):
        for i,metric in enumerate(('auroc','balanced_accuracy')):
            ax=aa[i,j]
            for m in ARMS:
                g=d[(d.seed==seed)&(d.method==m)].sort_values('step')
                ax.plot(g.step,g[metric],marker='o',ms=5,color=C[m],label=N[m])
            ax.set_xscale('log');ax.set_xticks([24,100,300,1000],['24','100','300','1000']);ax.set_ylim(.25,.9)
            ax.set_xlabel('Training update (log-spaced axis)');ax.set_ylabel('Auxiliary-score AUROC' if i==0 else 'Generated BA')
            ax.set_title(f'Seed {seed} · '+('ranking' if i==0 else 'answers'),loc='left');grid(ax)
    fig.legend(*aa[1,0].get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.115),ncol=3,fontsize=9)
    meta(fig,'Ranking and answer trajectories measure different gains','All registered development nodes shown · the same128 images are reused.',
      'Fixed LCE exceeds matched LM-only AUROC at all8 displayed seed/node combinations; these are repeated readouts.\nThe final endpoint was fixed at update1000. No best-checkpoint selection or new validation cases are implied.')
    publish(fig,5,'development_trajectories','预定开发节点的学习过程','排序收益是否只发生在一个挑选的checkpoint？',
      '固定LCE在两个种子×四个预定节点的全部8个读出中AUROC均高于LM-only；生成BA没有同样一致的提升。',
      '同一128图反复读出，不是8次独立试验；折线连接只是视觉引导，最终端点始终是1000步。',
      'Learning dynamics；核心成果的过程支撑',['07_scale_development_all_nodes.csv'],
      'Development AUROC and generated BA at every prespecified readout (24,100,300,1000 updates). The x-axis is logarithmically spaced to display all four nodes. Each seed reuses the same development cohort, and no endpoint was selected from these curves.')

def f06():
    d=read('17_final_score_distributions.csv');fig,aa=canvas(2,3,size=(11.4,7.9),left=.085,bottom=.22,top=.81,wspace=.28,hspace=.55)
    for i,seed in enumerate((43,44)):
        for j,m in enumerate(ARMS):
            ax=aa[i,j];g=d[d.endpoint==ep(seed,m)]
            for target,col in [(0,'negative'),(1,'positive')]:
                vals=np.sort(g[g.target==target].score.to_numpy());ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=C[col],label=f'{col.capitalize()} (n={len(vals)})')
            ax.set(xlabel='Raw auxiliary score',ylabel='Within-class ECDF',ylim=(0,1.02));ax.set_title(f'{N[m]} · Seed {seed}',loc='left',fontsize=11);grid(ax)
    fig.legend(*aa[1,0].get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.115),ncol=2,fontsize=9)
    meta(fig,'LCE changes the class separation of auxiliary scores','Within-class empirical distributions; each class has equal visual weight.',
      'Score axes are in raw model-specific units and differ by panel; scores are not calibrated probabilities.\nHigher scores indicate listed findings. The LM-only seed44 reversal is retained rather than corrected after seeing labels.')
    publish(fig,6,'score_distributions','异常分数的类别分布','AUROC差异在分数分布上体现为什么？',
      'LCE训练后的阳性/阴性分数呈现更清楚的排序分离；保留LM-only seed44反向排序的原始方向。',
      '不同模型分数尺度不可直接比较；这不是概率校准分析，也没有看标签后翻转分数或调阈值。',
      'Supplementary ranking analysis；答辩解释AUROC',['17_final_score_distributions.csv'],
      'Within-class empirical cumulative distributions of saved auxiliary scores. Each ECDF normalizes its own class, avoiding dominance by the478 positive images. Horizontal scales are model-specific; distributions should be interpreted within panels rather than as calibrated probabilities.','补充')

def f07():
    d=read('16_final_confusion_counts.csv');fig,aa=canvas(2,4,size=(11.6,7.5),left=.085,bottom=.17,top=.80,wspace=.40,hspace=.65)
    for i,seed in enumerate((43,44)):
        for j,m in enumerate(FOUR):
            g=d[d.endpoint==ep(seed,m)];ax=aa[i,j]
            count=g.pivot(index='target',columns='prediction',values='count').to_numpy();norm=g.pivot(index='target',columns='prediction',values='class_fraction').to_numpy()
            ax.imshow(norm,vmin=0,vmax=1,cmap=CMAP)
            for yy in (0,1):
                for xx in (0,1):ax.text(xx,yy,f'{count[yy,xx]}\n{norm[yy,xx]:.1%}',ha='center',va='center',fontsize=10,color='white' if norm[yy,xx]>.55 else '#17283E')
            ax.set_xticks([0,1],['No','Yes']);ax.set_yticks([0,1],['No','Yes']);ax.set_xlabel('Predicted answer' if m!='linear_head' else 'Classifier output');ax.set_ylabel('Target label')
            ax.set_title(f'{N[m]}\n'+(f'Seed {seed}' if m!='linear_head' else 'Same single fit'),fontsize=10)
            ax.spines[['left','bottom']].set_visible(False)
    meta(fig,'Generated answers: inspect both classes','Final-source cohort · counts and row-normalized percentages',
      'Each panel contains64 negative and478 positive images. Head panels repeat one classifier reference, not two fits.\nNo listed finding is not clinical health; all saved answers in these endpoints are valid.')
    publish(fig,7,'confusion_matrices','两类回答正确率与混淆矩阵','平均指标背后，阴性与阳性分别发生了什么？',
      '固定LCE seed43阴性正确43/64，相对LM-only的38/64改善；但不同种子和类别的变化不一致。',
      '总accuracy受阳性占比影响。单个类别的改善不能替代整体BA和跨种子结果；线性头是分类输出。',
      'Generated-answer results；答辩备用页',['16_final_confusion_counts.csv'],
      'Confusion matrices at the fixed final endpoint. Cells show counts and within-target-class percentages; colors share the0–1 scale. The same frozen-feature classifier is displayed in both rows for reference. Neural predictions are parsed generated answers.','补充')

def f08():
    d=read('18_ba_final_multiplicity_sensitivity.csv');fig,aa=canvas(size=(11.4,7.2),left=.25,wspace=.66,bottom=.18,top=.82)
    comps=[('fixed_lce','lm_only','Fixed LCE − LM-only'),('cabg_lce','lm_only','CABG − LM-only'),('cabg_lce','fixed_lce','CABG − Fixed LCE'),
           ('fixed_lce','linear_head','Fixed LCE − head'),('cabg_lce','linear_head','CABG − head'),('lm_only','linear_head','LM-only − head')]
    for j,seed in enumerate((43,44)):
        ax=aa[0,j]
        for i,(a,b,label) in enumerate(comps):
            r=d[(d.left==ep(seed,a))&(d.right==ep(seed,b))].iloc[0];x=100*r.delta_balanced_accuracy;lo=100*r.ba_ci_low;hi=100*r.ba_ci_high
            ax.plot([r.family12_ci_low*100,r.family12_ci_high*100],[i,i],color=C[a],alpha=.25,lw=7,solid_capstyle='round')
            ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=C[a],capsize=3,ms=6)
            ax.text(18.5,i+.25,f'{x:+.2f} pp',ha='right',fontsize=8.3,color=C[a])
        ax.set_yticks(range(len(comps)),[x[2] for x in comps]);ax.set_ylim(5.6,-.65);ax.axvline(0,color='#586675',lw=1);ax.set_xlim(-11,19)
        grid(ax,'x');ax.set_xlabel('Generated BA difference (pp)');ax.set_title(f'Seed {seed}',loc='left')
    meta(fig,'Answer gains are conditional on the model and comparator','The original primary endpoint remains generated balanced accuracy.',
      'Dark: original descriptive95% intervals. Pale: new99.583% sensitivity intervals for12 final comparisons.\nHead comparisons use each method\'s existing output rule. pp = percentage points; no patient-level inference.')
    publish(fig,8,'generated_ba_pairs','原始主指标：生成BA的配对比较','LCE的排序收益是否稳定转化为回答收益？',
      '固定LCE相对线性头的BA为+7.19/+5.53个百分点，原未校正95%区间均为正；但12项最终比较敏感性区间均跨0。相对LM则为+3.49/−2.76个百分点。',
      '线性头对比是各自既定输出规则下的局部成果；生成BA与分类BA读出不同。保留原区间并公开较保守分析，不能扩大为一般优越性。',
      'Generated-answer results；局部成果与原主指标并列',['02_scale_paired_original.csv','18_ba_final_multiplicity_sensitivity.csv'],
      'All six final-answer comparisons per seed. Dark bars reproduce the original unadjusted95% paired image-bootstrap intervals. Pale bars are post-hoc Bonferroni-level99.583% sensitivity intervals for the12 final comparisons, using the existing20,000 resamples. Original endpoints and intervals remain unchanged. Head comparisons use different pre-existing output rules; these are not selection-adjusted patient-level superiority tests.')

def f09():
    d=read('11_training_steps_with_median50.csv');summ=read('10_training_control_and_resources.csv').set_index('trajectory')
    fig,aa=canvas(size=(10.7,6.4),left=.095,wspace=.32,bottom=.25)
    allrat=d[d.trajectory.str.contains('fixed_lce|cabg_lce')].auxiliary_to_lm_shared_ratio
    ylimits=(10**np.floor(np.log10(allrat.min())),allrat.max()*1.7)
    for j,seed in enumerate((43,44)):
        ax=aa[0,j]
        for m in ['fixed_lce','cabg_lce']:
            g=d[d.trajectory==f's{seed}-{m}'];y=g.auxiliary_to_lm_shared_ratio.to_numpy()
            assert (y>0).all()
            ax.plot(g.step,y,color=C[m],alpha=.45,lw=.7)
            ax.plot(g.step,g.auxiliary_to_lm_shared_ratio_median50,color=C[m],lw=2,label=N[m]+' (median50)')
        ax.axhline(.2,color='#414A58',ls='--',lw=1.3,label='Budget cap0.2');ax.set_yscale('log');grid(ax)
        ax.set(xlabel='Training update',ylabel='Shared auxiliary / LM gradient norm',xlim=(1,1000),ylim=ylimits);ax.set_title(f'Seed {seed}',loc='left')
        f=summ.loc[f's{seed}-fixed_lce'];c=summ.loc[f's{seed}-cabg_lce']
        ax.text(.04,.95,f'Max: fixed {f.ratio_max:.2f}; CABG {c.ratio_max:.3f}\nCABG cap active: {c.cap_active_fraction:.1%}',transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    fig.legend(*aa[0,0].get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.13),fontsize=9,ncol=3)
    meta(fig,'CABG implements the specified auxiliary-gradient budget','Faint lines: all1000 steps. Solid lines: fixed50-step trailing medians.',
      'Ratios use the accepted shared-gradient definition before the common full-gradient clip; they are not Adam updates.\nBoth weighting schemes trained successfully. Budget enforcement alone does not establish a stability or accuracy benefit.')
    publish(fig,9,'gradient_budget','CABG 的梯度预算控制成果','控制器是否完成了它被设计来完成的约束？',
      '固定权重组最大共享辅助/LM范数比约4.99/22.14；CABG最大约0.2，cap在28.9%/27.3%更新生效。',
      '范数比不是Adam参数位移；共同全局裁剪也在作用。固定组没有真实崩溃，不能宣称已证明防止崩溃。',
      'Optimization/control results；工程成果页',['11_training_steps_with_median50.csv','10_training_control_and_resources.csv'],
      'Shared auxiliary-to-LM gradient norm ratios for fixed and CABG weighting. Faint traces retain all logged updates, with fixed50-step trailing medians overlaid. The cap is implemented within the frozen numerical tolerance. These are gradient norms, not optimizer displacements or evidence that the alternative would fail.')

def f10():
    d=read('11_training_steps_with_median50.csv');fig,aa=canvas(2,2,size=(10.7,8),bottom=.22,top=.82,hspace=.54)
    for j,seed in enumerate((43,44)):
        for i,metric in enumerate(['lm_loss','class_balanced_lce_loss']):
            ax=aa[i,j]
            for m in ARMS:
                g=d[d.trajectory==f's{seed}-{m}'];ax.plot(g.step,g[metric],color=C[m],lw=.5,alpha=.13)
                ax.plot(g.step,g[metric+'_median50'],color=C[m],lw=1.8,label=N[m])
            ax.set_yscale('log');ax.set(xlabel='Training update',ylabel='LM loss' if i==0 else 'Logged auxiliary loss',xlim=(1,1000));grid(ax)
            ax.set_title(f'Seed {seed} · '+('language objective' if i==0 else 'auxiliary objective'),loc='left')
    fig.legend(*aa[1,0].get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.115),ncol=3,fontsize=9)
    meta(fig,'Training dynamics distinguish the learning objectives','All6000 update records retained · logarithmic loss axes',
      'Faint: raw losses. Solid:50-step trailing medians. LM-only logs the auxiliary loss without optimizing that objective.\nTraining losses do not establish validation superiority, convergence, or the upper limit of the method.')
    publish(fig,10,'training_losses','训练目标与损失动态','辅助监督是否改变了训练中的优化行为？',
      '六条轨迹的全部6000步均纳入；可以观察直接优化LCE与只记录LCE时的差异，并对照语言目标。',
      '平滑只是固定50步尾随中位数；训练loss降低不等于泛化提升，不据曲线选择新checkpoint。',
      'Optimization analysis；完整报告或附录',['11_training_steps_with_median50.csv'],
      'Logged language and auxiliary losses throughout all six matched training trajectories. Raw observations are faint and fixed50-step trailing medians are emphasized. The LM-only arm does not optimize the auxiliary objective; its auxiliary loss is logged for comparison.','补充')

def f11():
    d=read('09_readout_comparison_posthoc.csv');fig,aa=canvas(size=(10.5,5.7),left=.16,wspace=.54,top=.79,bottom=.23)
    for j,seed in enumerate((43,44)):
        ax=aa[0,j];g=d[(d.seed==seed)&(d.split=='final')].set_index('method').loc[ARMS]
        z=g[['auxiliary_auroc','first_token_margin_auroc','generated_BA']].to_numpy()
        ax.imshow(z,vmin=.3,vmax=.9,cmap=CMAP,aspect='auto')
        for y in range(3):
            for x in range(3):ax.text(x,y,f'{z[y,x]:.3f}',ha='center',va='center',fontsize=13,color='white' if z[y,x]>.66 else '#17283E')
        ax.set_xticks(range(3),['Auxiliary\nAUROC','First-token margin\nAUROC','Generated\nBA'],fontsize=9)
        ax.set_yticks(range(3),[N[m] for m in ARMS]);ax.set_title(f'Seed {seed}',loc='left');ax.spines[['left','bottom']].set_visible(False)
    meta(fig,'An auxiliary score is only one model readout','Separate the learned anomaly branch, language ranking and discrete answers.',
      'Post-hoc use of stored first-token Yes−No logits; no model forward or calibration. Compare methods within a column.\nThe LM-only auxiliary branch lacks LCE supervision; its low auxiliary AUROC is not the entire model\'s visual capability.')
    publish(fig,11,'readout_comparison','辅助分数、语言logits与生成答案的区别','为什么不能把异常支路的AUROC当作整个模型能力？',
      'LM-only的首词Yes−No margin AUROC为0.8115/0.8421，明显高于其辅助分数AUROC；LCE主要改善了直接受监督的异常分数。',
      '新事后描述：首词margin排序不是实际生成BA或概率校准。不能据此宣称模型整体能力从0.36提升到0.80。',
      'Discussion / construct validity；答辩的重要解释',['09_readout_comparison_posthoc.csv'],
      'Three distinct readouts at the fixed final endpoint: auxiliary-score AUROC, post-hoc AUROC of the stored first-token Yes−No logit margin, and actual generated-answer BA. No logits were newly produced and no threshold was optimized. Low auxiliary AUROC in LM-only should not be interpreted as the complete model lacking discriminative visual information.')

def f12():
    d=read('12_intervention_effects.csv');fig,aa=canvas(2,2,size=(11.1,9),left=.23,bottom=.13,top=.83,wspace=.65,hspace=.55)
    cond=['center_rms','softsign_rms','softsign_shuffle_rms','ano_swap','visual_swap','prior_odds']
    labels=['Centered map','Softsign map','Shuffled softsign','Ano-token swap','Direct-token swap','Prior correction']
    for j,seed in enumerate((43,44)):
        g=d[d.seed==seed].set_index('condition').loc[cond]
        ax=aa[0,j];ax.barh(range(6),g.answers_changed,color=C['fixed_lce'],height=.58)
        for i,v in enumerate(g.answers_changed):ax.text(v+1,i,f'{v}/128',va='center',fontsize=8.8)
        ax.set_xlim(0,76);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlabel('Changed generated answers');grid(ax,'x');ax.set_title(f'Seed {seed} · response',loc='left')
        ax=aa[1,j];ax.barh(range(6),g.corrected,color=C['fixed_lce'],height=.58,label='Corrected');ax.barh(range(6),-g.harmed,color='#B77166',height=.58,label='Harmed')
        for i,(_,r) in enumerate(g.iterrows()):
            ax.text(r.corrected+1,i,str(int(r.corrected)),va='center',fontsize=8.5);ax.text(-r.harmed-1,i,str(int(r.harmed)),ha='right',va='center',fontsize=8.5)
        ax.axvline(0,color='#526172',lw=1);ax.set_xlim(-53,23);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlabel('Harmed ← images → Corrected');grid(ax,'x');ax.set_title(f'Seed {seed} · correctness',loc='left')
    meta(fig,'Local interventions reveal unequal path sensitivity','Completed A1/A2 diagnostics · the same128 development images per seed',
      'Map edits preserve RMS as specified; token swaps can be off-manifold. Prior correction is the separate registered A2.\nA sensitive path is not necessarily defective. Zero answer changes do not establish a zero population effect.')
    publish(fig,12,'intervention_effects','机制干预：响应与纠错分开看','哪些干预真正改变答案，改变后又是改善还是破坏？',
      '直达token替换改变61/62个答案，Ano替换只改变3/4个；先验修正每seed纠正2例也破坏2例，净收益0。',
      '只针对同一128张开发图和特定checkpoint。替换可能分布外，不能据此证明Ano无用或直达路径故障。',
      'Mechanism results；展示研究深度',['12_intervention_effects.csv'],
      'Generated-answer changes and paired corrections/harms under the completed local interventions. Each condition reuses128 development images per seed. Prior correction belongs to the separately registered A2 follow-up. Counts are not additional patients or exclusive causal mediation estimates.')

def f13():
    d=read('03_transport_paired_original.csv');fig,aa=canvas(2,2,size=(11.5,8.2),left=.25,bottom=.16,top=.82,wspace=.68,hspace=.60)
    comp=[('aligned','fixed_lce','Aligned − Fixed LCE'),('aligned','shuffled','Aligned − Shuffled'),('shuffled','fixed_lce','Shuffled − Fixed LCE')]
    for j,seed in enumerate((43,44)):
        for i,split in enumerate(['development','final']):
            ax=aa[i,j]
            for k,(a,b,label) in enumerate(comp):
                r=d[(d.left==ep(seed,a,split))&(d.right==ep(seed,b,split))].iloc[0];x=100*r.delta_balanced_accuracy;lo=100*r.ba_ci_low;hi=100*r.ba_ci_high
                ax.errorbar(x,k,xerr=[[x-lo],[hi-x]],fmt='o',color=C[a],capsize=3,ms=6)
                ax.text(12.8,k+.23,f'{x:+.2f}',ha='right',fontsize=8.5,color=C[a])
            ax.set_yticks(range(3),[x[2] for x in comp]);ax.set_ylim(2.6,-.55);ax.set_xlim(-15,13.5);ax.axvline(0,color='#526172',lw=1);grid(ax,'x')
            ax.set_xlabel('Generated BA difference (pp)');ax.set_title(f'Seed {seed} · '+('development' if i==0 else 'reused final'),loc='left',fontsize=11)
    meta(fig,'Spatial alignment has no consistent additional answer gain','Matched retraining · fixed step1000 · original paired intervals',
      'Development n=128; reused final n=542. Seed43 changes direction between the two readouts.\nThese unadjusted descriptive intervals do not prove equivalence or rule out other evidence-routing designs.')
    publish(fig,13,'transport_comparisons','空间对齐传递的匹配训练结果','把异常权重传入视觉特征聚合是否带来一致额外收益？',
      '对齐−固定LCE最终BA为+0.628/−1.095个百分点；对齐−打乱在seed43开发/最终方向相反，seed44未支持优势。',
      '最终图为重复使用；固定打乱可被训练补偿，且不控制所有特征范数/优化差异。不能推广为所有传递机制无效。',
      'Mechanism extension / ablation；后半部分结果',['03_transport_paired_original.csv'],
      'Matched aligned and shuffled transport comparisons at step1000, displayed separately for development and the reused final cohort. Bars retain the original descriptive95% paired image-bootstrap intervals. The comparisons test this particular integration, including its training effects, rather than all possible anomaly-to-language interfaces.')

def f14():
    count=read('14_transport_inference_changes.csv');tr=read('13_transport_trace_subset_posthoc.csv')
    fig,aa=canvas(2,2,size=(10.8,8.9),left=.12,bottom=.15,top=.83,hspace=.65,wspace=.36)
    fields=['pooled_relative_change','mixed_relative_change','fed_ano_relative_change']
    for j,seed in enumerate((43,44)):
        ax=aa[0,j];r=count.set_index('seed').loc[seed];vals=[r.ano_changed,r.logit_gap_changed,r.answers_changed]
        ax.bar(range(3),vals,color=[C['aligned'],C['cabg_lce'],C['lm_only']],width=.58)
        for k,v in enumerate(vals):ax.text(k,v+3,f'{int(v)}/128',ha='center',fontsize=10)
        ax.set_xticks(range(3),['Ano\nrepresentation','Yes−No\nlogit gap','Generated\nanswer']);ax.set_ylim(0,148);ax.set_ylabel('Changed cases');grid(ax);ax.set_title(f'Seed {seed} · all128 cases',loc='left')
        ax=aa[1,j];g=tr[tr.seed==seed];z=g[fields].to_numpy()*100
        for row in z:ax.plot(range(3),row,color=C['aligned'],alpha=.20,marker='o',ms=3,lw=1)
        med=np.median(z,axis=0);ax.plot(range(3),med,color=C['aligned'],lw=2.8,marker='o',ms=6,label='Subset median')
        for k,v in enumerate(med):ax.annotate(f'{v:.2f}%',(k,v),xytext=(5,7),textcoords='offset points',fontsize=9)
        ax.set_xticks(range(3),['Pooled\nfeature','Equal\nmixture','Fed Ano\ntokens']);ax.set_ylim(0,tr[fields].max().max()*115);ax.set_xlim(-.15,2.3);ax.set_ylabel('Layer-relative change (%)');grid(ax);ax.set_title(f'Seed {seed} · first8 cases only',loc='left')
    meta(fig,'A changed representation need not change a discrete answer','Inference shuffle of aligned step1000 models · existing development records',
      'Bottom panels are post-hoc descriptions of a non-random first8-case subset; thin lines are individual cases.\nRelative norms use different denominators across layers. Their shrinkage does not identify a universal causal bottleneck.')
    publish(fig,14,'transport_trace_changes','从表示扰动到答案变化的层次','证据表征改变以后，变化是否传递到了最终答案？',
      '每seed的128个Ano表示均改变；首词logit差分别44/70例改变，但答案均0例改变。前8图中，送入LLM的Ano相对扰动中位数约2.80%/2.87%。',
      '前8图不是随机样本，各层相对范数分母不同；不能推断全部128例的扰动量级或唯一因果瓶颈。',
      'Mechanism analysis；精选汇报页或补充材料',['14_transport_inference_changes.csv','13_transport_trace_subset_posthoc.csv'],
      'Top: numbers of changed Ano representations, first-token logit margins and generated answers under inference-time spatial shuffle. Bottom: existing full-trace descriptions for the first eight manifest cases per seed, with individual trajectories and medians. The trace subset is non-random and layer-relative norms use different denominators.')

def f15():
    d=read('17_final_score_distributions.csv');fig,aa=canvas(size=(10.6,6),left=.10,wspace=.31,bottom=.21)
    for j,seed in enumerate((43,44)):
        ax=aa[0,j];arr=[d[d.endpoint==ep(seed,m)].effective_support.to_numpy() for m in ARMS]
        boxes=ax.boxplot(arr,patch_artist=True,widths=.52,showfliers=False,medianprops={'color':'#16283E','linewidth':1.6})
        for patch,m in zip(boxes['boxes'],ARMS):patch.set_facecolor(C[m]);patch.set_alpha(.55)
        for k,v in enumerate(arr):ax.text(k+1,.64,f'median {np.median(v):.1f}',transform=ax.get_xaxis_transform(),ha='center',fontsize=9,color=C[ARMS[k]],bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        ax.axhline(1024,color='#A4AFBC',ls=':',lw=1);ax.set_yscale('log');ax.set_ylim(1,1600);ax.set_xticks([1,2,3],['LM-only','Fixed LCE','CABG-LCE']);ax.set_ylabel('Effective support (out of1024 positions)');grid(ax);ax.set_title(f'Seed {seed}',loc='left')
    meta(fig,'Auxiliary learning also changes spatial concentration','Final-source images · medians, interquartile ranges and1.5×IQR whiskers',
      'Outlier symbols are omitted for readability; all observations remain in the figure data. Lower support means\nmore concentrated weights, not more accurate lesion localization. No box/polygon accuracy is established.')
    publish(fig,15,'spatial_support','空间权重分布的结构变化','LCE是否改变了异常权重的空间分布？',
      '固定LCE最终effective-support中位数约70.9/67.9，LM-only约389.2/1013.9；空间分布发生明显变化。',
      '更集中不是更准确；没有定位真值评价。本图用于描述学到的空间结构，不能作为病灶定位效果图。',
      'Supplementary spatial analysis；避免放大为核心有效性主张',['17_final_score_distributions.csv'],
      'Distribution of effective spatial support in the final-source cohort. Boxes show medians and interquartile ranges, with1.5×IQR whiskers; outlier symbols are suppressed but no observations are removed from the data. Concentration is not a localization-accuracy measure.','补充')

def f16():
    d=read('10_training_control_and_resources.csv');order=[f's{s}-{m}' for s in (43,44) for m in ARMS];d=d.set_index('trajectory').loc[order]
    fig,aa=canvas(size=(10.8,6.4),left=.25,wspace=.65,bottom=.17,top=.81)
    labs=[f'Seed {int(r.seed)} / '+N[r.method] for _,r in d.iterrows()]
    for j,(metric,title,xlabel) in enumerate([('model_process_minutes','A  Model-process time','Minutes per1000-update trajectory'),('peak_MiB','B  Sampled peak memory','GPU memory (GiB)')]):
        ax=aa[0,j];v=d[metric].to_numpy()/(1024 if j else 1)
        ax.barh(range(6),v,color=[C[m] for m in d.method],height=.6)
        for i,x in enumerate(v):ax.text(x+.4 if j==0 else x+.08,i,f'{x:.2f}',va='center',fontsize=9)
        ax.set_yticks(range(6),labs);ax.invert_yaxis();ax.set_xlim(0,70 if j==0 else 22);ax.set_xlabel(xlabel);ax.set_title(title,loc='left');grid(ax,'x')
    meta(fig,'The controlled comparison was executable on the existing L4','Scale study: six matched1000-update trajectories;29,561,345 trainable parameters.',
      'All trajectories used1000 unique training images and3000 image exposures each.126 recoverable checkpoints retained.\nProcess time includes loading and overhead; sampled memory is not an allocator maximum. No billing claim or speedup test.')
    publish(fig,16,'training_resources','计算资源与可复现工程成果','这些比较在什么计算预算下实际完成？',
      '六条1000步正式训练均完成，每条约58–62分钟模型进程时间，约19.4GiB采样峰值显存；保存126个可恢复checkpoint。',
      '这只计规模研究正式训练，非整个项目总成本。进程时长不等于GPU忙碌或账单时长，未检验速度优越性。',
      'Implementation / reproducibility；工程贡献与答辩备用页',['10_training_control_and_resources.csv'],
      'Observed model-process duration and sampled peak GPU memory for each matched scale-study trajectory. All runs used the same1000-update budget and1000 unique training images. The measurements establish execution feasibility, not statistically tested efficiency superiority or monetary cost.','补充')

def contacts():
    out=ROOT/'verification/visual';out.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(CAT),4):
        sheet=Image.new('RGB',(2000,1440),'#e9eef3');draw=ImageDraw.Draw(sheet)
        for k,item in enumerate(CAT[start:start+4]):
            im=Image.open(FIG/(item['stem']+'.png')).convert('RGB');im.thumbnail((970,650))
            x=(k%2)*1000+15;y=(k//2)*720+40
            sheet.paste(im,(x+(970-im.width)//2,y+(650-im.height)//2));draw.text((x,y-25),item['id']+' '+item['stem'],fill='#17283E')
        sheet.save(out/f'contact-{start//4+1:02d}.jpg',quality=92)

def main():
    global BOOK
    FIG.mkdir(exist_ok=True);FD.mkdir(exist_ok=True)
    with PdfPages(ROOT/'figures-all.pdf',metadata={'Title':'LCE-MIL: completed-study data analysis','Author':'Research Project','Subject':'Post-hoc numerical analysis of accepted records; no new model execution'}) as pdf:
        BOOK=pdf
        for fn in (f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15,f16):
            fn();print('Rendered',CAT[-1]['id'],flush=True)
    (FIG/'catalog.json').write_text(json.dumps(CAT,indent=2,ensure_ascii=False)+'\n')
    lines=['# Figure data manifest','', 'All figures use real accepted records or explicit arithmetic references. No mock/model-generated data. New intervals/readout analyses are post-hoc.','', '| Figure | Data files | Script | Outputs |','|---|---|---|---|']
    for item in CAT:lines.append('| '+item['id']+' | '+'; '.join('`'+p+'`' for p in item['data'])+' | `scripts/plot_figures.py` | '+'; '.join('`'+p+'`' for p in item['outputs'])+' |')
    (FIG/'data-manifest.md').write_text('\n'.join(lines)+'\n')
    contacts();print('Complete:',len(CAT),'figures; PNG450dpi + SVG + PDF; four contact sheets.')

if __name__=='__main__':main()
