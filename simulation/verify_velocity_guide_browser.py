"""Exercise keyboard overlays and a bounded live playback in the dedicated browser."""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request

from websockets.sync.client import connect

p = argparse.ArgumentParser()
p.add_argument('--cdp-port', type=int, default=9236)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--viewer-port', type=int, default=2333)
p.add_argument('--traces', type=Path)
p.add_argument('--repo-guides', action='store_true')
p.add_argument('--heading', action='store_true')
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
pages = json.load(opener.open(f'http://127.0.0.1:{args.cdp_port}/json/list'))
page = next(p for p in pages if f':{args.viewer_port}' in p.get('url', ''))
with connect(page['webSocketDebuggerUrl'], proxy=None, max_size=32*1024*1024) as ws:
    seq = 0
    def call(method, **params):
        global seq
        seq += 1
        ws.send(json.dumps(dict(id=seq, method=method, params=params)))
        while True:
            r = json.loads(ws.recv(timeout=20))
            if r.get('id') == seq:
                assert 'error' not in r and 'exceptionDetails' not in r.get('result', {}), r
                return r['result']
    def js(s):
        return call('Runtime.evaluate', expression=s, returnByValue=True)['result'].get('value')
    def wait(predicate, timeout=15):
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(.1)
        raise AssertionError(js('document.body.innerText'))
    def key(name, number):
        call('Input.dispatchKeyEvent', type='keyDown', key=name, code=name, windowsVirtualKeyCode=number)
        call('Input.dispatchKeyEvent', type='keyUp', key=name, code=name, windowsVirtualKeyCode=number)

    call('Page.bringToFront')
    call('Emulation.setDeviceMetricsOverride', width=1416, height=992, deviceScaleFactor=1, mobile=False)
    if args.traces:
        def latest_trace():
            files = sorted(args.traces.glob('*/playback.jsonl'), key=lambda p:p.stat().st_mtime, reverse=True)
            if not files:
                return {}
            return json.loads(files[0].read_text().splitlines()[-1])
        wait(lambda: latest_trace().get('windows', 0) >= 1)
    wait(lambda: js("document.body.innerText.includes('Active Prompt:')"))
    js("[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Got it'))?.click()")
    wait(lambda: not js("[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Got it'))"))
    js("[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.includes('Generate'))?.click()")
    wait(lambda: js("!![...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.includes('Generate')&&b.getAttribute('aria-selected')==='true')"))
    js("(()=>{const l=[...document.querySelectorAll('label')].find(l=>l.textContent==='Use Target Velocity');const e=document.getElementById(l.htmlFor);if(!e.checked)e.click();document.activeElement?.blur()})()")
    wait(lambda: js("[...document.querySelectorAll('div')].filter(e=>e.childElementCount===0&&['↑','←','↓','→'].includes(e.textContent)).length===4"))
    velocity_fields = """(()=>{
      const label=[...document.querySelectorAll('label')].find(l=>l.textContent==='Target Root Velocity (xz)');
      for(let node=label?.parentElement;node;node=node.parentElement){
        const inputs=[...node.querySelectorAll('input:not([type=hidden])')];
        if(inputs.length===2)return inputs.map(e=>Number(e.value));
      }
      return null;
    })()"""
    before_velocity = js(velocity_fields)
    assert before_velocity is not None
    if args.heading:
        js("(()=>{const l=[...document.querySelectorAll('label')].find(l=>l.textContent==='Use Target Heading');if(!l)throw Error('heading control missing');const e=document.getElementById(l.htmlFor);if(!e.checked)e.click()})()")
    key('ArrowUp', 38)
    time.sleep(.2)
    key('ArrowLeft', 37)
    # Keyboard handling now uses continuous input and no transient turn toast.
    # Check its actual control values instead of requiring the old UI message.
    wait(lambda: js(velocity_fields) != before_velocity)
    if args.repo_guides:
        def scene_and_timeline():
            return js("""(()=>{
              let viewer=null,timeline=null;
              for(const canvas of document.querySelectorAll('canvas')){
                let f=canvas[Object.keys(canvas).find(k=>k.startsWith('__reactFiber'))];
                for(;f;f=f.return){
                  const p=f.memoizedProps;
                  if(p?.value?.mutable&&p.value.useSceneTree)viewer=p.value;
                  if(p?.timelineState?.tracks)timeline=p.timelineState;
                }
              }
              if(!viewer||!timeline)return null;
              window.guideTestViewer=viewer;
              const nodes=Object.entries(viewer.useSceneTree.getState()).filter(([k,n])=>n&&k.includes('2D Root')).map(([k,n])=>({name:k,visible:n.effectiveVisibility,position:n.position??n.message?.position}));
              const track=timeline.tracks.find(t=>t.name==='2D Root');
              const keys=timeline.keyframes.filter(k=>k.track_id===track?.uuid);
              return {nodes,keys,current_frame:timeline.current_frame,end_frame:timeline.end_frame};
            })()""")
        wait(lambda: (scene_and_timeline() or {}).get('keys', []))
        first_scene = scene_and_timeline()
        assert len(first_scene['keys']) == 5, first_scene
        assert sorted(k['frame'] for k in first_scene['keys']) == [10,20,30,40,50], first_scene
        assert len([n for n in first_scene['nodes'] if n['name'].endswith('/sphere')]) == 5, first_scene
        if args.heading:
            assert len([n for n in first_scene['nodes'] if n['name'].endswith('/arrow_head')]) == 5, first_scene
        # Adjust only this isolated test page for an inspectable screenshot.
        js("""(()=>{
          const m=window.guideTestViewer.mutable.current;
          const name=Object.keys(m.nodeRefFromName).find(k=>k.includes('g1_mesh/pelvis'));
          const obj=m.nodeRefFromName[name];if(!obj)return;
          const p=obj.getWorldPosition(m.camera.position.clone());
          m.camera.up.set(0,1,0);m.cameraControl.updateCameraUp();
          m.cameraControl.setLookAt(p.x+2.5,p.y+1.2,p.z+2.5,p.x,p.y-.3,p.z,false);
        })()""")
        time.sleep(.5)
        (args.output/'REPOSITORY_GUIDE_SCENE.json').write_text(json.dumps(first_scene, indent=2))
    (args.output/'keyboard_guides.png').write_bytes(base64.b64decode(call('Page.captureScreenshot', format='png')['data']))
    # A full run exercises streaming append and draws the path while moving.
    key(' ', 32)
    time.sleep(12)
    key(' ', 32)
    time.sleep(.5)
    result = dict(passed=True, body=js('document.body.innerText'),
                  scope='real control owner: enable Target Velocity, increase speed, turn, live play/pause; inspect server render metrics separately')
    assert 'unsafe replacement seam' not in result['body'], result
    assert 'Another viewer owns SONIC' not in result['body'], result
    if args.traces:
        result['trace'] = latest_trace()
        assert result['trace']['source_frame'] > 60 and not result['trace']['failed'], result
    if args.repo_guides:
        result['guide_scene'] = scene_and_timeline()
        keys = sorted(k['frame'] for k in result['guide_scene']['keys'])
        assert len(keys) == 5 and all(b-a==10 for a,b in zip(keys,keys[1:])), result
        assert result['guide_scene']['end_frame'] >= keys[-1], result
    (args.output/'VELOCITY_BROWSER_ACCEPTANCE.json').write_text(json.dumps(result, indent=2))
    (args.output/'after_playback.png').write_bytes(base64.b64decode(call('Page.captureScreenshot', format='png')['data']))
    print('KEYBOARD_GUIDE_PLAYBACK_PASS', flush=True)
