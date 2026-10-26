"""Local causal probes: change a registered map or one visual-token branch."""
import inspect
import textwrap
import types
import torch

CONDITIONS=('native','center_rms','softsign_rms','softsign_shuffle_rms','ano_swap','visual_swap')

def replacement(a,n,e,condition):
    if condition not in (*CONDITIONS,'clone'):raise ValueError(condition)
    if condition in ('native','ano_swap','visual_swap'):return e
    if condition=='clone':return e.clone()
    gap=a.float()-n.float()
    x=torch.tanh(gap/4) if condition=='center_rms' else gap/(1+gap.abs())
    original_rms=e.float().square().mean(dim=-1,keepdim=True).sqrt()
    x=x*(original_rms/x.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(1e-30))
    if condition=='softsign_shuffle_rms':
        assert x.shape[-1]==1024
        permutation=(torch.arange(1024,device=x.device)*405+137)%1024
        x=x[...,permutation]
    return x.to(e.dtype)


class Intervention:
    def __init__(self,model):
        self.model=model;self.module=model.visual.anomaly_qformer
        self.condition='native';self.capture=False;self.trace={};self.donor=None
        self.old=self.module.forward;self.handles=[]
    def record(self,key,value):
        if self.capture:self.trace[key]=value.detach().float().cpu().numpy().copy()
    def evidence(self,module,a,n,e):
        x=replacement(a,n,e,self.condition)
        for k,v in [('a',a),('n',n),('original_e',e),('replacement_e',x)]:self.record(k,v)
        return x
    def install(self):
        module=inspect.getmodule(type(self.module))
        src=textwrap.dedent(inspect.getsource(type(self.module).forward))
        needle='pre_gate_evidence = A_log - N_log'
        assert src.count(needle)==1
        src=src.replace(needle,needle+'\n    pre_gate_evidence = _stage2ag_evidence(self, abnormal_map, normal_map, pre_gate_evidence)')
        namespace=dict(vars(module));namespace['_stage2ag_evidence']=self.evidence
        exec(compile(src,'<stage2ag-registered-evidence-hook>','exec'),namespace)
        self.module.forward=types.MethodType(namespace['forward'],self.module)
        def query(mod,inputs,output):
            self.record('query',inputs[0]);self.record('attention',output[1]);self.record('pre_ffn',output[0])
        def post(mod,inputs,output):self.record('post_ffn',output)
        def visual(mod,inputs,output):
            result=list(output);combined=result[2];base=result[0]
            assert combined.ndim==2 and combined.shape[0]==base.shape[0]+16
            direct,ano=combined[:-16],combined[-16:]
            if self.condition in ('ano_swap','visual_swap'):
                assert self.donor is not None and self.donor.shape==combined.shape
                donor=self.donor.to(combined)
                if self.condition=='ano_swap':ano=donor[-16:]
                else:direct=donor[:-16]
                result[2]=torch.cat([direct,ano])
            self.last_combined=result[2].detach().cpu()
            if self.capture:
                self.record('fed_ano',ano)
                if self.condition in ('native','clone'):self.record('direct',direct)
                self.fed_direct=direct.detach().cpu()
            return tuple(result)
        self.handles=[self.module.q_former.register_forward_hook(query),
                      self.module.post_ffn.register_forward_hook(post),
                      self.model.visual.register_forward_hook(visual)]
        return self
    def close(self):
        self.module.forward=self.old
        for h in self.handles:h.remove()
