"""Read actual imported implementation/dependency and base-model identities."""
import argparse,hashlib,importlib,inspect,json,os,subprocess,sys,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    modules=['torch','transformers','transformers.models.qwen2_5_vl.modeling_qwen2_5_vl','transformers.modeling_flash_attention_utils','flash_attn','flash_attn.flash_attn_interface','numpy','PIL','scipy','qwen_vl_utils.vision_process']
    packages={};implementations={}
    for name in modules:
        m=importlib.import_module(name);f=Path(inspect.getfile(m));packages[name]={'version':getattr(m,'__version__',None),'path':str(f),'sha256':sha(f)};implementations[str(f)]=sha(f)
    repo=root/'code/runtime';bundled=repo/'transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py';actual=packages['transformers.models.qwen2_5_vl.modeling_qwen2_5_vl']
    # Installed patched implementation is authoritative; capture any difference
    # explicitly, never assume PYTHONPATH overlays a regular installed package.
    base=Path('/home/checkpoints/Lingshu-7B');ix=json.loads((base/'model.safetensors.index.json').read_text());files=sorted(set(ix['weight_map'].values()))
    files += ['model.safetensors.index.json','config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json','generation_config.json']
    identity={str(base/f):{'bytes':(base/f).stat().st_size,'sha256':sha(base/f)} for f in files}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    result={'python':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'packages':packages,'implementation_files':implementations,
            'installed_qwen_matches_archived_repository':actual['sha256']==sha(bundled),'archived_qwen_sha256':sha(bundled),
            'gpu':gpu,'base_model':identity,'checked_epoch':time.time(),'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    with (root/'environment/identity-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'CAPTURED','installed_qwen_matches_archive':result['installed_qwen_matches_archived_repository'],'base_shards':4,'gpu':gpu}),flush=True)
if __name__=='__main__':main()
