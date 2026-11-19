"""Internal per-process entry; start/stop through launch.py."""
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'inference'))
from bootstrap import activate
activate()
component=sys.argv.pop(1)
if component in ('text','generator','viewer'):
    script={'text':ROOT/'text/server.py','generator':ROOT/'inference/server.py','viewer':ROOT/'viewer/run_viewer.py'}[component]
    if component=='text':sys.path.insert(0,str(ROOT/'text'))
    sys.argv[0]=str(script)
    runpy.run_path(str(script),run_name='__main__')
elif component=='web':
    import uvicorn
    sys.path.insert(0,str(ROOT/'web/backend'))
    uvicorn.run('app:app',host='0.0.0.0',port=int(os.environ['MM2_WEB_PORT']))
elif component=='physics':
    from unitree_sdk2py.core import channel
    folder=Path(os.environ['MM2_RUNTIME_DIR'])/'dds';folder.mkdir(parents=True,exist_ok=True)
    for name in ('ChannelConfigAutoDetermine','ChannelConfigHasInterface'):
        setattr(channel,name,getattr(channel,name).replace('/tmp/cdds.LOG',str(folder/f'cdds_{os.getuid()}.LOG')))
    script=Path(os.environ['MM2_SONIC_ROOT'])/'gear_sonic/scripts/run_sim_headless_web.py'
    sys.argv[0]=str(script)
    runpy.run_path(str(script),run_name='__main__')
else:raise ValueError(component)
