"""Build a portable bilingual analysis guide; no manuscript/slide drafting."""
from pathlib import Path
from html import escape
from PIL import Image
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def read(name):return pd.read_csv(ROOT/'tables'/name)
def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def html_table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def a(path,label):return f'<a href="{escape(path,quote=True)}">{escape(label)}</a>'

SCHEMAS=[
('01_endpoint_master.csv','55个不同端点标识，含训练头；主结果另行筛选固定final','stage, endpoint, seed, method, split, step, n, TN/FP/FN/TP, AUROC, AP, BA, class recall'),
('02_scale_paired_original.csv','规模研究42项原始配对；保留原95% BA区间','left/right、配对正确/错误计数、各指标差、原BA区间'),
('03_transport_paired_original.csv','传递研究52项原始配对','同上；development/final/inference-shuffle按端点区分'),
('04_cohort_counts.csv','训练/开发/最终的两类图像数量','split, class, n, split_total'),
('05_scale_final_with_posthoc_intervals.csv','六个神经端点+一个线性头的最终点估计和新增区间','AUROC/AP/BA及20,000次bootstrap的95%区间'),
('06_auroc_paired_posthoc.csv','八项AUROC配对差及多重比较敏感性','delta_auroc、95%区间、99.375% family8区间'),
('07_scale_development_all_nodes.csv','全部24个神经开发读出及单个线性头开发读出','step24/100/300/1000，两个seed；不是独立样本量'),
('08_roc_pr_coordinates.csv','七个最终端点的ROC与PR坐标','curve=ROC时x=FPR/y=TPR；PR时x=Recall/y=Precision'),
('09_readout_comparison_posthoc.csv','固定1000步、两split、三方法、两seed的12项描述','auxiliary_auroc, first_token_margin_auroc, generated_BA'),
('10_training_control_and_resources.csv','六轨迹优化控制与资源汇总','ratio median/q95/max, cap fraction, loss, minutes, peak MiB'),
('11_training_steps_with_median50.csv','全部6000条更新及固定50步尾随中位数','原始loss/ratio/coefficient/clipping和median50'),
('12_intervention_effects.csv','每seed六种A1/A2干预相对native','answers_changed, corrected, harmed, delta_BA, 原始CI'),
('13_transport_trace_subset_posthoc.csv','每seed首8个manifest图的既有trace描述','pool/mix/fed Ano的相对范数变化；非随机子集'),
('14_transport_inference_changes.csv','每seed全部128图的改变计数','Ano改变、首词logit差改变、答案改变'),
('15_transport_endpoints.csv','22个新传递研究端点','aligned/shuffled、固定节点、推理打乱、dev/final'),
('16_final_confusion_counts.csv','七个最终端点的混淆计数','target, prediction, count, class_fraction'),
('17_final_score_distributions.csv','最终七端点的小型逐图记录','sample_id、标签/解析答案/连续分数/logit差/support；不是原图'),
('18_ba_final_multiplicity_sensitivity.csv','事后修订：全部12项最终BA比较的敏感性','原95%CI、新95%CI、新99.583% family12CI；复用同一bootstrap'),
]

TIPS={
'F01':'先看绝对图像数，再看类别比例；用恒答Yes参照解释为何不能只报accuracy。',
'F02':'同一panel内比较方法。圆点/方块是点估计，横线是新增图像级95%区间；两个panel保留种子差异。',
'F03':'零线表示没有AUROC增量。深色线为95%，浅粗线为八项比较的较保守敏感性区间；先读Fixed LCE−LM，再读与线性头的比较。',
'F04':'ROC越靠左上通常排序越好；PR须同时看0.882的阳性比例参照。图例括号分别是AUROC/AP，二者不能叫accuracy。',
'F05':'上排看异常排序，下排看生成BA。连接的四点均是预先指定的节点；不要从图中另挑最好checkpoint。',
'F06':'在每个panel内部看两类曲线的水平分离。同一数值轴位置不代表跨模型相同概率。',
'F07':'每行按真实类别归一化。先看负类64张，再看正类478张；重复出现的线性头是同一个拟合结果。',
'F08':'深色是历史95%区间，浅粗线是12项最终比较的事后敏感性区间。正值有利于左侧方法；要同时读比较对象、种子和区间。',
'F09':'虚线为0.2预算；淡线保留所有更新，实线为固定平滑。纵轴为log尺度，可以同时看到低比率与尖峰。',
'F10':'淡线保留原始损失，实线仅帮助阅读趋势；两个目标分开看，不能把不同loss的绝对值直接比较。',
'F11':'按列比较方法：异常分数AUROC、首词margin AUROC、生成BA测量不同东西，不把三个列值当同一个指标。',
'F12':'上排问“有没有改变”，下排问“改对还是改错”。直接路径很敏感，但许多改变是损害，这不是一个可直接部署的修复。',
'F13':'先比较同一seed的开发与最终行，再比较两个seed。保留方向反转，不将局部正区间当总体优势。',
'F14':'上排是全部128图的改变计数，下排只有首8图。细线是每个既有case，粗线是中位数；两排样本范围不同。',
'F15':'低support表示质量集中在较少位置。这里观察空间分布结构，不评价病灶定位正确性。',
'F16':'读取相同训练预算下的真实进程时长与采样显存；几分钟波动没有被当成加速收益。',
}

def main():
    cat=json.loads((ROOT/'figures/catalog.json').read_text());ci=read('06_auroc_paired_posthoc.csv');fin=read('05_scale_final_with_posthoc_intervals.csv').set_index('endpoint')
    ba=read('18_ba_final_multiplicity_sensitivity.csv');check_path=ROOT/'verification/independent-analysis-check.json'
    verified=json.loads(check_path.read_text())['status']=='PASS' if check_path.exists() else False
    status='数值与资产核验 PASS；视觉检查见独立记录' if verified else '图表已生成；最终核验进行中'
    mainrows=[]
    for seed in [43,44]:
        for m,label in [('lm_only','LM-only'),('fixed_lce','固定LCE-MIL'),('cabg_lce','CABG-LCE-MIL')]:
            r=fin.loc[f's{seed}-{m}-step1000-final']
            mainrows.append([seed,label,f'{r.auroc:.4f}',f'[{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]',f'{r.balanced_accuracy*100:.3f}%',f'{int(r.tn)}/64',f'{int(r.tp)}/478'])
    h=fin.loc['linear-head-final'];mainrows.append(['单次拟合','冻结线性分类头',f'{h.auroc:.4f}',f'[{h.auroc_ci_low:.3f}, {h.auroc_ci_high:.3f}]',f'{h.balanced_accuracy*100:.3f}%',f'{int(h.tn)}/64',f'{int(h.tp)}/478'])
    cheaders=['Seed','比较','ΔAUROC','配对95%区间','八项敏感性区间']
    crows=[[int(r.seed),r.comparison,f'{r.delta_auroc:+.4f}',f'[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]',f'[{r.family8_ci_low:+.4f}, {r.family8_ci_high:+.4f}]'] for _,r in ci.iterrows()]
    brows=[]
    for seed in [43,44]:
        r=ba[(ba.left==f's{seed}-fixed_lce-step1000-final')&(ba.right=='linear-head-final')].iloc[0]
        brows.append([seed,f'{r.delta_balanced_accuracy*100:+.2f}',f'[{r.ba_ci_low*100:+.2f}, {r.ba_ci_high*100:+.2f}]',f'[{r.family12_ci_low*100:+.2f}, {r.family12_ci_high*100:+.2f}]'])
    headers=['Seed','方法','分数AUROC','新增95%区间','生成/分类BA','阴性正确','阳性正确']
    sections=[
      ('summary','分析结论与使用顺序',[
       '本包面向UTS32933报告与汇报准备，以LCE-MIL的受控排序成果为主线，保留CABG、生成回答和机制研究的完整证据。它是数据分析与图表材料包，不是已经写完的课程报告或幻灯片。',
       '核心正面成果：固定LCE在两个保留种子上的最终异常分数AUROC为0.8005/0.8037，相对匹配LM-only增加0.1972/0.4418。新增配对95%区间及八项比较的较保守敏感性区间均为正。这里改善的是辅助异常分数，不能把它直接表述为整个模型诊断准确率的增量。',
       '局部回答成果：固定LCE相对冻结线性分类头的BA点估计高7.19/5.53个百分点，原始未校正95%区间均为正；本轮额外检查了全部12项最终BA比较，较保守的敏感性区间跨零。因此可作为有条件的局部成果展示，不能声称已确证一般优越性。',
       '工程成果：CABG把共享辅助/LM梯度范数比限制在约0.2，固定权重组峰值约4.99/22.14。它证明预算按设计执行；两组都训练完成，因此不能直接推导防止崩溃或提升准确率。',
       '解释性发现：LM-only的首词Yes−No margin AUROC仍为0.8115/0.8421。该模型的辅助异常支路缺少LCE监督，其低辅助AUROC不能代表整个视觉语言模型缺少区分信息。LCE的贡献应准确落在直接受监督的读出上。',
       '建议主线阅读：F01→F02→F03→F05→F09→F08→F11→F12/F13。F04、F06、F07、F10、F14、F15、F16供详细报告或答辩补充。所有不利方向、两个种子和线性头参照均保留。']),
      ('methods','统计口径与证据边界',[
       '本轮只读取26个已接受的小型CSV/JSON/审计快照（合计约4.74MB），没有读取图像、checkpoint或模型权重，没有新增前向、训练、阈值选择或GPU任务。模型、数据和原始实验文件均未改动。',
       '规模与传递导出共复核87组端点记录，其中32组是历史复用；合并后55个不同端点标识，包含训练头的记录。主结果固定使用step1000的六个神经端点和一个最终分类头；训练头的100%拟合结果没有当作泛化证据。',
       'BA=(阴性正确率+阳性正确率)/2。AUROC衡量连续分数的排序；AP是非插值average precision，受阳性比例影响。AUROC/AP不是生成回答准确率，也没有作为校准概率来解释。',
       '新增bootstrap固定64阴性和478阳性，每类内对图像有放回抽样20,000次，所有方法和种子共享对应的图像抽样权重，RNG seed=20260920。95%为百分位区间；八项AUROC对比另给99.375%敏感性区间。',
       'BA敏感性为明示的事后修订：在发现固定LCE−头的局部正结果后，对全部12项最终比较统一复用同一批bootstrap，报告99.583%区间；没有追加抽样直到结果有利。它覆盖最终12项，不覆盖全部42项历史开发/最终比较。',
       '这些区间是条件于已拟合模型的图像级不确定性。它们不处理未知患者相关性、预训练重叠、历史选择和全项目多重分析，也不提供种子总体推断。没有进行两个种子的t检验或把多次读出合成更大的患者样本量。',
       '原研究的生成BA主终点保持原样；本包对AUROC的强调是事后综合呈现，不能改写原协议。跨零不证明等效，零答案变化也不证明人口层面的零效应。',
       '开发128图反复使用，最终542图在传递研究中再次使用；没有患者映射，基座具体预训练重叠未知。标签仅指13项已列发现，阴性不代表临床健康。空间集中度也不是定位准确率。'])]
    md=['# LCE-MIL 课程数据分析与图表解读','',f'日期：2026-09-20 · {status}','',
        '[图文总览](analysis-index.html) · [16页图表PDF](figures-all.pdf) · [复现说明](README.md) · [数据来源](provenance/input-manifest.json)','']
    htmlparts=[]
    for sid,title,paras in sections:
        md += ['## '+title,'']+sum(([p,''] for p in paras),[])
        htmlparts.append('<section id="'+sid+'"><h2>'+title+'</h2>'+''.join('<p>'+escape(p)+'</p>' for p in paras)+'</section>')
    md += ['## 主要结果表','',md_table(headers,mainrows),'','AUROC区间为本轮新增事后分析；BA值为原始最终输出的点估计。线性头是分类器，其他方法为生成回答。','',
           '### 配对AUROC增量','',md_table(cheaders,crows),'',
           '### 固定LCE相对线性分类头的局部BA成果','',md_table(['Seed','ΔBA(pp)','原95%区间(pp)','12项敏感性区间(pp)'],brows),'',
           '这两个输出系统使用各自既定规则，未重新校准或选择阈值；不是对所有分类器的优越性结论。','']
    htmlparts.insert(1,'<section id="key-results"><h2>主要结果表</h2>'+html_table(headers,mainrows)+'<p class="note">AUROC区间为新增事后分析；BA是既定输出规则的原始结果。线性头为单次拟合的分类参照。</p><h3>配对 AUROC 增量</h3>'+html_table(cheaders,crows)+'<h3>固定 LCE 相对线性头的局部 BA 成果</h3>'+html_table(['Seed','ΔBA(pp)','原95%区间(pp)','12项敏感性区间(pp)'],brows)+'<p class="note">原始局部正结果与较保守分析并列；不同输出规则，不是一般分类器优越性结论。</p></section>')
    md+=['## 图表逐项说明','']
    for item in cat:
        title=item['id']+'｜'+item['title_cn'];stem=item['stem'];links=' · '.join(f'[{ext.upper()}](figures/{stem}.{ext})' for ext in ['png','svg','pdf'])
        md+=['### '+title,'',links,'',f'![{title}](figures/{stem}.png)','',
             '**对应问题：** '+item['question'],'','**如何阅读：** '+TIPS[item['id']],'','**主要结果：** '+item['finding'],'',
             '**可支持的范围与限制：** '+item['limits'],'','**建议放置：** '+item['placement'],'',
             '**English caption：** '+item['caption_en'],'','**数据表：** '+', '.join(f'[{x}](tables/{x})' for x in item['tables']),'']
        downloads=' '.join(a(f'figures/{stem}.{ext}',ext.upper()) for ext in ['png','svg','pdf'])
        with Image.open(ROOT/'figures'/f'{stem}.png') as im: img_width,img_height=im.size
        data_links=' · '.join(a('tables/'+x,x) for x in item['tables'])
        htmlparts.append(f'<section id="{item["id"]}" class="figure-section"><div class="eyebrow">{item["priority"]} · {item["id"]}</div><h2>{escape(item["title_cn"])}</h2><p class="question">{escape(item["question"])}</p><div class="downloads">{downloads}</div><a href="figures/{stem}.svg"><img width="{img_width}" height="{img_height}" loading="lazy" src="figures/{stem}.png" alt="{escape(title)}"></a><div class="interpretation"><div><h3>怎样读</h3><p>{escape(TIPS[item["id"]])}</p></div><div><h3>说明了什么</h3><p>{escape(item["finding"])}</p></div></div><div class="limit"><strong>解释边界</strong><p>{escape(item["limits"])}</p></div><p><strong>报告位置：</strong>{escape(item["placement"])}</p><details><summary>英文图注与数据来源</summary><p lang="en">{escape(item["caption_en"])}</p><p>{data_links}</p></details></section>')
    glossary=[['AUROC','随机阳性分数高于随机阴性的排序概率（平分计半）；不是accuracy。'],['AP','非插值average precision；阳性比例会影响其水平。'],['BA','两类正确率的平均；本研究原生成回答主指标。'],['pp','百分点；如0.0349的BA差为3.49个百分点。'],['paired','同一图像在两个方法下成对比较，保留配对结构。'],['post-hoc','在已知历史结果后组织的新分析，不冒充预注册。'],['effective support','空间权重分散/集中的描述，非定位准确率。'],['independent verifier','实现分离的核验，本项目仍由同一操作者执行。']]
    md+=['## 指标速查','',md_table(['术语','含义'],glossary),'','## 文件与复现','',
         '- `tables/`：18份汇总/逐图/配对数据表。','- `figures/`：16组PNG450dpi、SVG、PDF及图表专属数据。','- `data/source/`：26份小型原始结果快照；不含模型和图像。','- `scripts/`：收集、复算、绘图、导读、核验脚本。','- `plan/`：事后分析口径、修订记录、贡献追溯。','- `verification/`：数值核验、视觉检查与联系图。','',
         '## 方法文档','',
         '- [scikit-learn AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)','- [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)','- [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)','',
         '数值实现和本地版本见脚本、requirements.txt及environment.json；本轮采用自定义的配对分层bootstrap，而非直接调用SciPy默认抽样。']
    htmlparts.append('<section id="files"><h2>文件、术语与复现</h2>'+html_table(['术语','含义'],glossary)+'<p>'+a('README.md','复现说明')+' · '+a('analysis-guide.md','完整Markdown解读')+' · '+a('figures-all.pdf','16页图表PDF')+' · '+a('figures/data-manifest.md','图表数据清单')+' · '+a('provenance/input-manifest.json','来源及哈希')+' · '+a('plan/analysis-protocol.md','统计口径')+' · '+a('plan/analysis-amendment-01.md','BA敏感性修订')+'</p><h3>18份数据表</h3>'+''.join('<p>'+a('tables/'+name,name)+' — '+escape(desc)+'</p>' for name,desc,_ in SCHEMAS)+'<p>方法参考：'+a('https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html','AUROC')+' · '+a('https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html','Average precision')+' · '+a('https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html','Bootstrap')+'</p></section>')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef3f5;color:#233349;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}a{color:#087f78;text-decoration:none}a:hover{text-decoration:underline}aside{position:fixed;inset:0 auto 0 0;width:258px;background:#142b3f;color:#e4edf4;padding:30px 22px;overflow:auto}aside h1{font-size:22px;line-height:1.4;margin:0 0 12px}aside p{font-size:12px;color:#a9bfce}aside a{display:block;color:#cbdde7;font-size:13px;line-height:1.6;margin:12px 0}main{margin-left:258px;padding:44px 42px;max-width:1600px}.hero{background:#142b3f;color:white;padding:38px;border-radius:14px}.hero h1{font-size:32px;line-height:1.35;margin:8px 0 14px}.hero p{color:#c6d9e4}.eyebrow{font-size:12px;letter-spacing:.08em;color:#00867d;font-weight:700}.hero .eyebrow{color:#80d4c3}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:24px}.kpi{border:1px solid #3a5263;border-radius:8px;padding:18px}.kpi strong{font-size:26px;display:block;color:#8bdbca}.kpi span{font-size:13px;color:#c5d8e4}section{background:white;padding:32px 36px;margin:26px 0;border-radius:12px;box-shadow:0 3px 14px #10263608;scroll-margin-top:20px}h2{font-size:25px;line-height:1.5;margin:4px 0 22px}h3{font-size:17px;margin:18px 0 8px}p{margin:12px 0}.note{font-size:13px;color:#617184}.question{font-size:18px;color:#34566a}img{display:block;width:100%;height:auto;border:1px solid #e4ebef;border-radius:6px;margin:22px 0}.downloads{display:flex;gap:10px}.downloads a{border:1px solid #c4ddd9;border-radius:5px;padding:4px 14px;font-size:13px}.interpretation{display:grid;grid-template-columns:1fr 1fr;gap:30px}.limit{background:#f4f7f9;border-left:3px solid #9bafbd;padding:15px 20px;margin:14px 0}.limit p{margin:4px 0;font-size:14px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.7}th{background:#eaf2f3;text-align:left;padding:12px;white-space:nowrap}td{padding:10px 12px;border-bottom:1px solid #e3ebef}tbody tr:nth-child(even){background:#f8fbfc}details{border-top:1px solid #e4ebef;margin-top:18px;padding-top:12px;font-size:14px}summary{cursor:pointer;color:#087f78}code{font-size:13px}footer{font-size:13px;color:#617184;padding:20px}@media(max-width:1050px){aside{position:static;width:auto}aside nav{display:flex;gap:15px;overflow:auto}aside nav a{min-width:160px}main{margin:0;padding:20px}.kpis{grid-template-columns:1fr}.interpretation{grid-template-columns:1fr}section{padding:24px}.hero h1{font-size:27px}}@media print{aside,.downloads{display:none}main{margin:0;padding:0}body{background:white;font-size:11pt}.hero,section{box-shadow:none;break-inside:avoid;border:0}.figure-section{break-before:page}details{display:block}.hero{background:white;color:#142b3f}.hero p{color:#233349}.kpis{display:none}a{color:inherit}img{max-height:65vh;object-fit:contain}}'''
    nav=a('#summary','结论与使用顺序')+a('#key-results','主要结果与区间')+a('#methods','统计口径')+''.join(a('#'+x['id'],x['id']+' '+x['title_cn']) for x in cat)+a('#files','文件、术语与复现')
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LCE-MIL｜课程数据分析与图表解读</title><style>'+css+'</style></head><body><aside><h1>LCE-MIL<br>研究数据分析</h1><p>UTS32933 · 2026-09-20<br>已完成实验的事后整理</p><nav>'+nav+'</nav></aside><main><header class="hero"><div class="eyebrow">CONTROLLED EVIDENCE · TRACEABLE RESULTS</div><h1>把 LCE-MIL 的成果讲清楚</h1><p>16组专业图表、18份数据表与逐项中文解释。英文图表可用于学术报告，原始主指标和不确定性完整保留。</p><div class="kpis"><div class="kpi"><strong>0.8005 / 0.8037</strong><span>固定LCE最终异常分数AUROC</span></div><div class="kpi"><strong>+0.197 / +0.442</strong><span>相对LM-only的AUROC增量</span></div><div class="kpi"><strong>0.2</strong><span>CABG共享辅助梯度预算上限</span></div></div></header>'+''.join(htmlparts)+'<footer>'+escape(status)+'。本包无新模型执行；不是患者独立或临床有效性验证。</footer></main></body></html>'
    (ROOT/'analysis-index.html').write_text(page);(ROOT/'analysis-guide.md').write_text('\n'.join(md)+'\n')
    schema=['# Table schema','', 'All performance metrics and intervals are fractions unless marked pp. AUC differences are absolute AUC units. Counts are images/readouts, not patients. The linear-head fit is shared across seed panels.','',md_table(['File','Purpose / rows','Main fields'],SCHEMAS)]
    (ROOT/'tables/table-schema.md').write_text('\n'.join(schema)+'\n')
    trace=[['LCE排序收益','辅助图像级监督','af matched LM/fixed/CABG','F02–F06','两个seed中对匹配LM的异常分数改善；不等同整个模型能力','原点估计+新增事后区间'],
      ['局部回答成果','既定生成/分类读出','af final Fixed LCE−head','F07–F08','两个点估计较高，原未校正区间为正；多重比较敏感性跨零','原描述性结果+事后敏感性'],
      ['梯度预算控制','CABG shared-gradient cap','af6000 logged updates','F09–F10','控制器按规定执行；不推出防崩溃/优化器位移','原训练记录'],
      ['区分表征与答案','两类连续读出/生成','af stored scores and margins','F11','不同读出衡量不同能力；不将未监督辅助分数当全部能力','新增事后描述'],
      ['路径局部依赖','map/token/prior interventions','ag A1/A2','F12','特定checkpoint对局部干预的响应；非独占因果归因','已核验诊断'],
      ['传递适配的边界','aligned/shuffled routing','ah matched training/inference shuffle','F13–F14','该适配未见一致额外收益；首8trace不能推广','原结果+已有事后trace'],
      ['工程可复现性','matched runs and checkpoints','af execution records','F16','既有L4上完成固定预算训练；不宣称速度/成本最优','执行审计']]
    (ROOT/'plan/method-experiment-traceability.md').write_text('# Contribution-to-evidence map\n\n'+md_table(['Contribution','Module','Experiment','Figures','Allowed claim','Status'],trace)+'\n')
    readme='''# LCE-MIL course analysis package

Start with **analysis-index.html** (Chinese illustrated guide), or **analysis-guide.md**. English publication assets are in figures/; figures-all.pdf contains all 16 figures. This package analyzes accepted saved results; it is not a final course manuscript or a model backup.

## Reproduce without a GPU

Use a Python environment with requirements.txt. The original run used the existing local Miniforge Python and installed numerical/plotting libraries; no installation or model loading was required. Execute from any working directory:

```sh
python /path/to/course-analysis-20260920-v1/scripts/analyze.py
python /path/to/course-analysis-20260920-v1/scripts/ba_sensitivity.py
python /path/to/course-analysis-20260920-v1/scripts/plot_figures.py
python /path/to/course-analysis-20260920-v1/scripts/build_guide.py
python /path/to/course-analysis-20260920-v1/scripts/verify_analysis.py
python /path/to/course-analysis-20260920-v1/scripts/build_guide.py
```

The included 26 source snapshots are sufficient for these steps. **collect_sources.py** is only needed to build/check snapshots against the original workspace; do not run it in a detached copy without the source project. Scripts overwrite only their derived outputs inside this package. PDF metadata may differ between runs; numerical tables should reproduce within the recorded floating-point tolerance. Rebuilding the package can invalidate its original output inventory: retain the delivered archive as a fixed version or create a new copy for changes.

## Content

- tables/: 18 CSVs and table-schema.md. Original BA intervals and new post-hoc sensitivities are separate columns/files.
- figures/: 16 PNGs at 450 dpi, 16 SVGs, 16 individual PDFs, figure-specific data and catalog.json/data-manifest.md.
- data/source/: 26 small accepted snapshots (~4.74MB); no image, model or checkpoint bodies.
- data/derived/bootstrap_final.npz: 20,000×7×3 saved scalar bootstrap reductions; **not model tensors**.
- plan/:retrospective protocol, explicitly timed BA amendment and evidence traceability.
- provenance/:source hashes, environment and output inventory.
- verification/:separate-code numerical/asset checks and visual QA.

## Interpretation rules

AUROC concerns auxiliary-score ranking. The original registered primary endpoint remains generated BA. AP is prevalence dependent. There are only two retained model seeds; the repeated 128 development/542 final images do not become additional independent cases. Patient identity and exact pretraining overlap remain unknown. The head is one fixed classifier; its output is not generated text. No localization or clinical-validation claim is established.

New AUROC and BA sensitivity intervals are explicitly post-hoc. Their multiplicity families are respectively 8 AUROC comparisons and 12 final BA comparisons, not every analysis in the project's history. The interval coverage is approximate and does not remove patient/selection dependence. All main comparators and adverse results are retained.

No server connection, training, inference, threshold fitting, image/cache download, Git commit or push was performed for this package. Model re-execution is outside this package and would still require the authorized server environment.
'''
    (ROOT/'README.md').write_text(readme)
    print('Built HTML/Markdown guide,18-table schema, contribution traceability and README.')

if __name__=='__main__':main()
