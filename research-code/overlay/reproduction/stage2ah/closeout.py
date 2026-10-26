"""Local documentation from verified small artifacts only; never loads a model."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from reproduction.stage2ah.ops import LOCAL, WORKSPACE


def read(path):
    return json.loads(path.read_text())


def main():
    r = LOCAL
    m = r/'verified-execution-v1'
    results = m/'results-v1'
    verification = read(results/'independent-verification.json')
    training = read(m/'training-independent-verification-v1.json')
    closure = read(r/'closure-v1/final-live-status.json')
    mirror = read(r/'closure-v1/mirror-and-results-check.json')
    assert verification['status'] == training['status'] == mirror['status'] == 'PASS'
    assert not closure['budget']['open_gpu_jobs'] and not closure['gpu_compute_processes']
    assert all(x['exit_code'] == 0 for x in closure['jobs'])
    assert (training['steps_verified'], training['sparse_blocks_verified'], training['checkpoints_verified']) == (4000, 20, 84)
    assert (verification['verified_new_generated_records'], verification['endpoints'], verification['comparisons']) == (4472, 54, 52)
    rows = read(results/'endpoint-summary.json')
    table = {x['endpoint']: x for x in rows}
    pairs = read(results/'paired-comparisons.json')
    classes = {(x['endpoint'], int(x['target'])): x for x in csv.DictReader((results/'class-counts.csv').open())}
    trace = read(results/'trace-descriptives-posthoc.json')
    assert sum(x['invalid'] for x in rows if x['origin'] == 'stage2ah') == 0
    labels = dict(aligned='对齐传递', shuffled='训练时打乱', fixed_lce='固定 LCE', lm_only='LM-only', cabg_lce='CABG-LCE')
    final = ['| seed | 方法 | BA% | 阴性正确/64 | 阳性正确/478 | AUROC | AP |', '|---|---|---:|---:|---:|---:|---:|']
    dev = ['| seed | 对齐 BA% | 打乱 BA% | 固定 LCE BA% | LM-only BA% | CABG BA% |', '|---|---:|---:|---:|---:|---:|']
    contrasts = ['| 数据 | seed | 对比 | ΔBA 百分点 | 图像配对 95% 区间 |', '|---|---|---|---:|---|']
    for seed in [43, 44]:
        dev.append('|'+str(seed)+'|'+ '|'.join(f"{table[f's{seed}-{arm}-step1000-development']['balanced_accuracy']*100:.3f}" for arm in labels)+'|')
        for arm, label in labels.items():
            name = f's{seed}-{arm}-step1000-final'
            x = table[name]
            final.append(f"|{seed}|{label}|{100*x['balanced_accuracy']:.3f}|{classes[name, 0]['correct']}|{classes[name, 1]['correct']}|{x['auroc']:.6f}|{x['average_precision']:.6f}|")
        for split in ['development', 'final']:
            for other in ['fixed_lce', 'shuffled']:
                suffix = f'-step1000-{split}'
                x = pairs[f's{seed}-aligned{suffix}__minus__s{seed}-{other}{suffix}']
                low, high = x['paired_balanced_accuracy_95ci']
                contrasts.append(f"|{'开发' if split=='development' else '重复最终'}|{seed}|对齐 − {labels[other]}|{100*x['delta_balanced_accuracy']:+.3f}|[{100*low:+.3f}, {100*high:+.3f}]|")
    stamp = datetime.fromtimestamp(closure['epoch'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    window = closure['budget']['research_window_seconds']/3600
    model_hours = closure['model_process_seconds_including_stage2ag']/3600
    charged_hours = closure['budget']['gpu_process_seconds']/3600
    training_hours = sum(x['seconds'] for x in closure['jobs'] if 'formal' in x['name'] or 'prefix' in x['name'])/3600
    local_bytes = sum(p.stat().st_size for folder in [r, WORKSPACE/'Medic-AD/reproduction/stage2ah'] for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    assert local_bytes < closure['carryover']['local_storage_reservation_bytes']
    resources = dict(cutoff_utc=stamp, **{k: closure[k] for k in ['model_process_seconds_including_stage2ag', 'cpu_process_seconds_conservatively_charged', 'charged_storage_bytes', 'remote_stage2ah_bytes']},
                     charged_process_seconds=closure['budget']['gpu_process_seconds'], research_window_seconds=closure['budget']['research_window_seconds'],
                     local_stage2ah_bytes_at_document_build=local_bytes, local_reservation_bytes=1000000000, free_bytes=closure['budget']['free_bytes'], training_process_hours_including_prefixes=training_hours,
                     optimizer_updates=4000, peak_MiB=max(x['peak_MiB'] for x in closure['jobs']), jobs=closure['jobs'])
    (r/'closure-v1/resource-account.json').write_text(json.dumps(resources, indent=2)+'\n')
    with (r/'closure-v1/job-costs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(closure['jobs'][0])); writer.writeheader(); writer.writerows(closure['jobs'])
    audit = f'''# Stage2ah completion audit — 2026-09-19

**COMPLETE / execution PASS；稳定回答收益与空间对齐优势未获支持。** 本轮已完成实际机制开发、两条24步可恢复前缀、四条1000步匹配训练、全部实际生成及独立核验。没有以诊断负结果代替训练。stage2ag的诊断证据保留；其当时的停止训练决定已由本轮授权及执行接续。

研究对象仍为1000张训练、128张重复开发图及542张已使用最终图（64阴性/478阳性），seed43/44。没有新数据、种子、受保护internal-test访问或本地模型计算。最终读出于 epoch1789813350.3928967 在四条训练和开发评价完成后统一释放；未据最终结果调参或重测。

**机制与公平性。** 对四层 softsign(A−N) 取均值，以既有 τ=1/log1024 得到 MIL 权重；32×32权重按2×2求和映射到16×16视觉value网格，得到 z=qᵀV，再与原Q-former输出等权混合后送入原post-FFN。消融只以固定置换 (101p+37) mod256 打乱新增value聚合的对应。原查询路径、直达视觉路径、16个Ano及64个直达tokens仍在；没有增加参数。T21=29,561,345、训练顺序、共同初态、固定LCE系数、BF16/FP32优化器、1000步日程及解码口径与stage2af匹配，复用LM-only/固定LCE/CABG历史基线。完整设计、候选比较和反向审查见[冻结协议](plan/protocol.md)。

两条24步短程是预先冻结1000步轨迹的真实前缀，计入4000次更新，没有额外优化器更新。禁用新机制后完整logits与原实现逐位一致；保存重载后下一块LM/LCE梯度及RNG一致；新增LM梯度路径、冻结参数身份通过。正式训练对应12000次图像曝光；另有8次禁用/原始前向和12次恢复工程梯度图像执行，均无额外更新（不把激活检查点重计算另计为样本）。固定step1000是主端点，24/100/300只为预定开发轨迹读出。

**重复最终集结果（真实自由词表贪心生成，严格原Yes/No解析）。**

{chr(10).join(final)}

所有4472条新生成均有效。历史冻结线性头仅是分类参照：最终BA67.720%、35/64阴性及386/478阳性正确；不混入生成方法。异常分数AUROC/AP与生成BA是不同读出，AP还受阳性高占比影响。

**开发集固定step1000结果。**

{chr(10).join(dev)}

**预设主对比与机制对比。**

{chr(10).join(contrasts)}

区间来自按类别分层的10000次图像配对bootstrap，并经独立100000次多项式重采样复核；未作多重比较校正。对齐相对固定LCE的最终差值为+0.628/−1.095个百分点，两个区间均跨零。对齐相对训练时打乱在seed43开发为−7.031、最终为+3.439个百分点，seed44最终为−0.935；方向并不稳定。seed43的局部区间排除零不能替代跨种子、跨读出的总体论据，也不能把跨零解释为等效性证明。全部52项预设比较保留在[配对表](verified-execution-v1/results-v1/paired-comparisons.csv)。

**机制观测与事后量级描述。** 两seed的固定最终开发模型在推理时打乱后，均有128/128个Ano表示改变，直达tokens与辅助异常分数完全相同，生成答案均0/128改变。这是同一128张图的重复读出，不能算作256个独立病例。已有记录的首词Yes−No logit差分别有44/128、70/128改变，因此不能称该路径对语言模型完全无影响。

看到答案不变后，另作不加载模型的事后数值描述：同一组前8张manifest图中，pooling表示的相对变化中位数为37.10%/37.79%，混合后为14.36%/18.39%，实际送入语言模型的Ano为2.80%/2.87%。这些数值与post-FFN后的相对扰动收缩一致，但样本不是随机抽样、不同层的相对范数分母不同，不能推出普遍因果瓶颈或代表全部128张图。无新前向、读出条件或最终图逐图分析。见[事后范围记录](posthoc-trace-description-plan.json)与[原始描述表](verified-execution-v1/results-v1/trace-descriptives-posthoc.json)。

**核验范围。** [训练独立核验](verified-execution-v1/training-independent-verification-v1.json)PASS：4000个标量/控制器/顺序/日程记录、20个原始稀疏梯度块、84个检查点。它不是每一步稠密优化器重演。[结果独立核验](verified-execution-v1/results-v1/independent-verification.json)PASS：4472个新增生成记录、176个完整传递trace、含历史参照的54端点和52配对。checkpoint→评价参数身份、token解码/解析、输入身份、AUROC/AP/BA、空间权重、pooling与混合、实际Ano/direct tokens均核对。最大score误差1.43e−7、空间权重误差2.53e−7、pooling相对误差1.47e−4、最大绝对误差0.0078125，均在冻结容差内。独立实现由同一操作者执行，不是独立实验室复现。

本地9046个镜像文件共102,731,156字节，经[写入后hash复核](closure-v1/mirror-and-results-check.json)全部一致。综合逐图表有11466行，包含重复读出与历史参照，不是新增病例数。真实结果PNG/SVG已视觉检查，[开发曲线](figures/development-curves.png)和[最终配对图](figures/final-paired-comparisons.png)附[data manifest](figures/data-manifest.md)。权重、图像cache、检查点及完整原始梯度/特征数组留在服务器。

**资源与失败记录。** 账目截点{stamp}。继承stage2ag的1199.453秒与原首次GPU epoch1789749069.5550537；累计模型进程{model_hours:.4f}小时，另有CPU分析/核验{closure['cpu_process_seconds_conservatively_charged']:.3f}秒保守计入预算，总计{charged_hours:.4f}/16小时。四轨迹训练含前缀{training_hours:.4f}小时。研究窗口{window:.4f}/36小时，包含等待与本地整理时间，不是仅计GPU忙碌时间。4000/9000次更新；峰值{resources['peak_MiB']}MiB；远端新增{closure['remote_stage2ah_bytes']/1e9:.6f}GB，含旧阶段记账和1GB本地预留的存储保守计数{closure['charged_storage_bytes']/1e9:.6f}/110GB，空余{closure['budget']['free_bytes']/1e9:.3f}GB（下限50GB）。本地实际小于1GB预留。没有查阅本轮供应商账单，不以旧小时价格虚报费用。详见[资源账](closure-v1/resource-account.json)及[逐任务成本](closure-v1/job-costs.csv)。

两前缀、四训练、八评价及四CPU分析/核验任务全部正常退出，共18个登记任务；没有本轮模型/训练/评价失败重跑。评价前两项记录修订分别增强checkpoint身份核验和实际fed-token追踪，未改变冻结方法、输入、训练或生成规则，旧源码保留。收尾一次未压缩文本镜像在60秒超时，随后用gzip传输同一白名单成功，未重跑模型；[失败与修复记录](closure-v1/mirror-attempt-01.json)保留。

**源码与运行收尾。** 本地branch innovation/cabg-mil-v1.2-redesign-r0，HEAD cb5a33ca4d6d976d5f736f3f0a3217fdb5e47b2f，tracked clean；原stage2ae/af/ag/r未跟踪内容保留，新stage2ah未提交。HEAD不能单独代表运行身份：协议SHA256为6465a4e56406b72c535fb3b7d714a87c8dec9e1e7488d7331a889b986ab6fa90，每任务源码快照绑定实际实现；收尾168个共同运行源码hash均未改变，本地关键模型/核验源码与远端相同。仅本地观察、镜像和文档工具有后续维护。原ControlMaster仍有效；本轮模型及campaign进程已退出，GPU0MiB/0%，medic-ad仍为1窗口/attached、pane%0/PID131/bash，未重建session或向pane发送输入。[只读收尾快照](closure-v1/final-live-status.json)。

**研究判断与边界。** 本轮得到可复查的机制适配和有对照的负面/不一致证据，没有建立稳定能力提升或新方法优越性。归一化注意力聚合已有明确先例；冻结后补读的Anomaly-OV也含异常显著性池化与Q-former路径，必须正面承认接近性，不能把这里的组合改名包装为原创。见[设计解释](design-interpretation.md)与[最近方法审查](nearest-prior-review.md)，后者明确标注阅读发生在冻结之后。

当前证据不足以支撑一篇宣称有效新方法的独立论文。保留固定LCE作为可复用排序基线，停止继续扩大本轮对齐传递矩阵。若未来另获授权，优先区分“归一化空间权重丢失图像级置信度”与“下游读出/直达路径补偿”两种解释，并以新的独立来源评价检验能力；这是后续研究问题，不是本轮已证明的原因或已启动的实验。

128开发反复使用、542最终重复使用，方法选择受历史项目证据影响；分层图像区间不能消除选择风险、患者相关性或预训练重叠。没有患者映射；NIH来源家族参与基础模型预训练已知，具体样本重叠未知。仅两个固定初始化、单一设置、1000步不能证明整个机制类别无效或最优。共享VPT与原Ano/direct路径阻止独占因果归因；固定sham可被训练补偿，且不保持pooling特征范数或有效8×8权重分布。无定位真值、临床有效性或患者独立泛化主张。

本周期收尾后停止。实例和用户tmux保留；无commit、push、部署、网站或论文写作，无子代理、付费扩容、新数据/种子或下一轮自动启动。
'''
    (r/'completion-audit.md').write_text(audit)
    (r/'STATUS.md').write_text('''# Stage2ah status

COMPLETE / execution PASS. All four1000-update tracks,22 new generation endpoints,4472 new generated records and independent verification are complete. Stable generated-answer improvement and a consistent spatial-alignment advantage are NOT SUPPORTED. This is not proof of equivalence or universal mechanism failure.

See [completion audit](completion-audit.md), [artifact index](artifact-index.md), [training verification](verified-execution-v1/training-independent-verification-v1.json), [result verification](verified-execution-v1/results-v1/independent-verification.json), and [process/resource closeout](closure-v1/final-live-status.json).

All owned execution processes ended; GPU empty; existing ControlMaster/tmux/instance preserved. No next study was launched. The same128 development images and previously used542 final images remain reused evidence; no new patient independence or pretraining-overlap evidence was obtained.
''')
    (r/'artifact-index.md').write_text('''# Stage2ah artifact index — completed cycle

[Completion audit and scientific judgment](completion-audit.md) · [Current status](STATUS.md).

Design: [frozen protocol](plan/protocol.md), [machine protocol](protocol.json), [seal](protocol-seal.json), [design interpretation](design-interpretation.md), [nearest-prior review](nearest-prior-review.md), [baseline reuse](baseline-reuse-evidence-v1.json).

Execution: [campaign completion](verified-execution-v1/campaign-complete.json), [one final-evaluation release](verified-execution-v1/final-evaluation-release.json), [accepted endpoints](verified-execution-v1/accepted-endpoints.json), [initial seal](implementation-seal-v1.json), [evaluation seal](evaluation-implementation-seal-v1.json), [checkpoint-identity amendment](verification-amendment-01.json), [pre-evaluation fed-token instrumentation](evaluation-instrumentation-amendment-02.json). Per-job actual source/process/exit/GPU records are under verified-execution-v1/control; all4000 step records and recovery records are under formal. Historical sources and failed mirror attempt remain preserved.

Verified results: [training audit](verified-execution-v1/training-independent-verification-v1.json), [independent raw result audit](verified-execution-v1/results-v1/independent-verification.json), [endpoint table](verified-execution-v1/results-v1/endpoint-summary.csv), [class counts](verified-execution-v1/results-v1/class-counts.csv), [all52 paired comparisons](verified-execution-v1/results-v1/paired-comparisons.csv), [all comparison rows](verified-execution-v1/results-v1/per-image.csv). The producer analysis-status file is an immutable pre-verification snapshot; the later independent PASS closes its pending state.

Mechanism arithmetic: [post-hoc scope](posthoc-trace-description-plan.json), [development trace magnitudes](verified-execution-v1/results-v1/trace-descriptives-posthoc.json). No extra model execution or final-case analysis was performed for these descriptions.

Figures: [development PNG](figures/development-curves.png) / [SVG](figures/development-curves.svg); [final pairwise PNG](figures/final-paired-comparisons.png) / [SVG](figures/final-paired-comparisons.svg); [real-data manifest](figures/data-manifest.md).

Closure: [remote processes/GPU/tmux/source identity](closure-v1/final-live-status.json), [resource account](closure-v1/resource-account.json), [job costs](closure-v1/job-costs.csv), [9046-file disk hash check](closure-v1/mirror-and-results-check.json), [text-transfer timeout and repair](closure-v1/mirror-attempt-01.json). [Carryover](carryover.json) and [prospective feasibility forecast](short-trial-forecast.json) preserve the unrestarted budget.

Code: [transport](../../../Medic-AD/reproduction/stage2ah/transport.py), [training](../../../Medic-AD/reproduction/stage2ah/train.py), [evaluation](../../../Medic-AD/reproduction/stage2ah/evaluate.py), [independent results verifier](../../../Medic-AD/reproduction/stage2ah/verify_results.py), [local plotting](../../../Medic-AD/reproduction/stage2ah/plot_results.py).

Remote primary root: /home/Medic-AD/outputs/cabg-mil-v1.2/evidence-transport-20260919-v1. Model weights, image caches,84 checkpoints,20 full raw-gradient blocks and full raw feature arrays stay there. Local mirrors contain selected small text/numerical evidence only. Historical stage2af/stage2ag primary artifacts are unchanged.
''')
    current = f'''## 当前阶段 — stage2ah 机制开发与匹配训练已完成（2026-09-19）

**COMPLETE / execution PASS；稳定生成收益和空间对齐优势未获支持。** 完成LCE空间权重value聚合主机制及固定空间打乱消融×seed43/44，四条1000步轨迹、4000次更新、84个checkpoint。两条24步短程是正式轨迹前缀，已计入4000步。新增4472条实际生成、22端点；含公平历史基线的54端点/52配对和训练记录均通过独立核验。

542张**重复使用的最终图像**上，生成BA：对齐75.539%/72.156%，打乱72.101%/73.091%，固定LCE74.912%/73.251%（seed43/44）。对齐−固定为+0.628/−1.095个百分点，图像配对95%区间均跨0。对齐−打乱在seed43开发和最终方向相反，seed44也未支持一致优势；不能据局部正区间宣称总体有效。

固定最终开发模型的推理打乱使两个seed各128个Ano表示全部改变，但生成答案均0/128改变，直达tokens与辅助分数相同；部分语言logits仍改变，不能称该路径完全无影响。事后前8图trace描述与post-FFN后的相对扰动收缩一致，只限该非随机子集。已有加权池化先例和Anomaly-OV接近方法被明确记录；没有确认新颖性或可独立成篇的方法优势。

模型计算仅在原L4；含stage2ag累计模型进程{model_hours:.4f}小时，保守计入CPU核验后{charged_hours:.4f}/16小时，窗口账目{window:.4f}/36小时；存储保守计数{closure['charged_storage_bytes']/1e9:.3f}/110GB，空余{closure['budget']['free_bytes']/1e9:.3f}GB。9046个本地小型镜像文件hash通过，PNG/SVG已检查。18个登记任务全部退出，GPU空闲，原tmux与实例保留。一次文本传输超时已压缩重传修复，无模型重跑。

患者映射及具体预训练重叠仍未知，开发/最终均已重复使用；不声称患者独立泛化、临床有效或机制普遍无效。本轮开发训练补全了stage2ag当时未执行的环节；该阶段停止训练决定只属历史。研究周期结束，不自动追加实验或写论文；没有commit/push、子代理或本地模型替代。

[完成审计](../evidence-transport-20260919-v1/completion-audit.md)、[产物索引](../evidence-transport-20260919-v1/artifact-index.md)、[训练核验](../evidence-transport-20260919-v1/verified-execution-v1/training-independent-verification-v1.json)、[结果核验](../evidence-transport-20260919-v1/verified-execution-v1/results-v1/independent-verification.json)、[最终配对图](../evidence-transport-20260919-v1/figures/final-paired-comparisons.png)。历史记录如下。

'''
    for name in ['progress.md', 'project-overview.md']:
        path = WORKSPACE/'document/cabg_mil_v1_2/plan'/name
        text = path.read_text(); start = text.index('## 当前阶段 — stage2ah'); end = text.index('## 已完成诊断阶段', start)
        text = text[:start]+current+text[end:]
        text = text.replace('## 最近完成的训练阶段 — stage2af', '## 历史训练阶段 — stage2af')
        path.write_text(text)
    path = WORKSPACE/'PROJECT_INDEX.md'; text = path.read_text()
    start = text.index('- **当前stage2ah'); end = text.index('- **已完成stage2ag', start)
    entry = f'''- **当前stage2ah COMPLETE / execution PASS（2026-09-19）。** 四条1000步匹配训练、4000更新、84checkpoint、4472条新生成、22新端点全部完成；独立核验PASS54端点/52配对。对齐机制最终BA75.539%/72.156%，固定LCE74.912%/73.251%；差值+0.628/−1.095个百分点，两个配对区间均跨0，未支持稳定收益。两个seed的固定开发模型推理打乱各改变128个Ano表示、0个生成答案。最终542图为重复使用证据；患者独立性和新颖性未建立。
  累计模型进程{model_hours:.4f}小时，保守预算{charged_hours:.4f}/16小时；窗口{window:.4f}/36小时，存储保守{closure['charged_storage_bytes']/1e9:.3f}/110GB。9046个小型镜像hash通过，图表已核对。本轮进程结束、GPU空闲、原tmux/实例保留，无下一轮启动。本轮补全stage2ag当时未执行的开发训练，其停止决定只属历史。
  [完成审计](document/cabg_mil_v1_2/evidence-transport-20260919-v1/completion-audit.md)、[产物索引](document/cabg_mil_v1_2/evidence-transport-20260919-v1/artifact-index.md)、[结果表](document/cabg_mil_v1_2/evidence-transport-20260919-v1/verified-execution-v1/results-v1/endpoint-summary.csv)、[配对图](document/cabg_mil_v1_2/evidence-transport-20260919-v1/figures/final-paired-comparisons.png)。

'''
    text = text[:start]+entry+text[end:]
    text = text.replace('**最近完成训练stage2af：', '**历史训练stage2af：').replace('stage2ae/stage2af/stage2ag/stage2r', 'stage2ae/stage2af/stage2ag/stage2ah/stage2r')
    path.write_text(text)
    print(json.dumps(dict(status='DOCUMENTED',model_hours=model_hours,charged_hours=charged_hours,window_hours=window,local_bytes_at_build=local_bytes)))


if __name__ == '__main__':
    main()
