"""Resolve module import aliases used by shared model components."""
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import types

class _Aliases(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def __init__(self,config):
        self.modules=config['modules'];self.exports=config['exports']
        self.parents=set()
        for name in [*self.modules,*self.exports]:
            parts=name.split('.')
            self.parents.update('.'.join(parts[:i]) for i in range(1,len(parts)))
        self.names=set(self.modules)|set(self.exports)|self.parents
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in self.names:return None
        # A real flat module always takes precedence over its old package name.
        root=Path(__file__).resolve().parents[2]
        if (root.joinpath(*fullname.split('.')).with_suffix('.py')).exists():return None
        if (root.joinpath(*fullname.split('.'))/'__init__.py').exists():return None
        return importlib.util.spec_from_loader(fullname,self,is_package=fullname in self.parents)
    def create_module(self,spec):return None
    def exec_module(self,module):
        name=module.__name__
        def get(attribute):
            if attribute.startswith('__'):raise AttributeError(attribute)
            if attribute in self.exports.get(name,{}):
                target,symbol=self.exports[name][attribute]
                value=getattr(importlib.import_module(target),symbol)
            elif name+'.'+attribute in self.names:
                value=importlib.import_module(name+'.'+attribute)
            elif name in self.modules:value=getattr(importlib.import_module(self.modules[name]),attribute)
            else:raise AttributeError(attribute)
            setattr(module,attribute,value);return value
        module.__getattr__=get

def install_import_aliases():
    if any(isinstance(f,_Aliases) for f in sys.meta_path):return
    config=Path(__file__).resolve().parents[2]/'configs/imports.json'
    sys.meta_path.append(_Aliases(json.loads(config.read_text())))
