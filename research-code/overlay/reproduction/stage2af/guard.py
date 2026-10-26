"""File-open boundary for model processes: only the current frozen input cache."""
import os
import sys
from pathlib import Path

class InputGuard:
    def __init__(self,root,rows):
        self.root=Path(root).resolve();self.allowed={str(Path(r['prepared_path']).resolve()) for r in rows}
        self.opened=set();self.violations=[]
    def install(self):
        def hook(event,args):
            if event!='open' or not args or isinstance(args[0],int):return
            try:p=os.fsdecode(args[0])
            except TypeError:return
            # Resolve without recursive stat/open audit activity.
            p=os.path.realpath(os.path.abspath(p))
            old=p.startswith('/home/data/medic-ad/')
            stage_image=p.startswith(str(self.root/'data/train')+'/') or p.startswith(str(self.root/'data/test')+'/')
            cached=p.startswith(str(self.root/'data/prepared-v1')+'/') and p.endswith('.pt')
            if old or stage_image or (cached and p not in self.allowed):
                self.violations.append(p);raise PermissionError('Image/cache outside frozen stage allowlist')
            if cached:self.opened.add(p)
        sys.addaudithook(hook);return self
    def record(self):
        return {'allowed_cache_files':len(self.allowed),'opened_unique_cache_files':len(self.opened),
                'violations':self.violations,'historical_image_files_opened':0,'raw_images_opened_by_model':0}
