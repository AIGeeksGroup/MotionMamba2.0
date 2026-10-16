"""Render saved raw qpos, without regenerating or physics stabilization."""
import html,os,subprocess
from pathlib import Path
import numpy as np
from .io import read,write,sha

def render(config,output):
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    from PIL import Image,ImageDraw
    from .engine import config as load_config
    cfg=load_config(config);root=Path(cfg['budget_root']);dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path('/data3/hy/GR00T-WholeBodyControl/gear_sonic_deploy/g1/scene_29dof.xml')
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=270,width=400)
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.distance=3.;cam.elevation=-18;cam.azimuth=125
    cases=[]
    for group in ([] if (dest/'CASES.json').exists() else ['W','S','T','X']):
        key=f'{group}_3_n0';cols=[]
        for name,folder in [('P0','P0/development'),('B8','B/eval8'),('B128 STOPPED','B/eval128')]:
            path=root/folder/(key+'.json');row=read(path)
            cols.append(dict(name=name,artifact=row['artifact'],events=row['events'],receipts=row['receipts'],windows=row['windows'],source=str(path)))
        cases.append((key,cols))
    for group in ([] if (dest/'CASES.json').exists() else ['W','X']):
        path=root/'C/pool0_supplement'/f'{group}_0_teacher_chain.json'
        if path.exists():
            row=read(path);cases.append((f'rejected_official_{group}',[dict(name='Official same-state teacher: failed complete event',artifact=row['artifact'],source=str(path),events=row['events'])]))
    if (dest/'CASES.json').exists():
        cases=[(entry['id'],entry['cols']) for entry in read(dest/'CASES.json')]
        for _,columns in cases:
            for column in columns:
                if column.get('row_source'):
                    source=read(column['row_source']);column.update(receipts=source['receipts'],windows=source['windows'])
    manifest=[]
    try:
      for key,cols in cases:
        arrays=[np.load(c['artifact'])['qpos'][c.get('start_frame',0):c.get('end_frame')] for c in cols];frames=max(map(len,arrays))
        for col,q in zip(cols,arrays):col['actual_available_frames']=len(q)
        width=400*len(cols);height=350;video=dest/(key+'.mp4');sheetfile=dest/(key+'.jpg')
        if not video.exists():
            tmp=dest/(key+'.partial.mp4');indices=np.linspace(0,frames-1,9,dtype=int).tolist();sheet=Image.new('RGB',(width*3,height*3),'white')
            pipe=subprocess.Popen(['ffmpeg','-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','22','-pix_fmt','yuv420p',str(tmp)],stdin=subprocess.PIPE)
            try:
              for i in range(frames):
                frame=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(frame)
                for j,(col,q) in enumerate(zip(cols,arrays)):
                    ox=j*400
                    if i>=len(q):
                        draw.text((ox+5,4),col['name'],fill='black')
                        draw.text((ox+5,40),'NO CONTINUATION: saved rollout ended',fill='red')
                        draw.text((ox+5,60),'See recorded failure; no padded motion',fill='red')
                        continue
                    data.qpos[:]=q[i];mujoco.mj_forward(model,data);cam.lookat[:]=[q[i,0],q[i,1],.7]
                    renderer.update_scene(data,camera=cam);frame.paste(Image.fromarray(renderer.render()),(ox,80))
                    draw.text((ox+5,4),col['name'],fill='black');draw.text((ox+5,20),f'{key}: frame {i}, {i/25:.2f}s | RAW, no physics',fill='black')
                    if 'receipts' in col:
                        r=col['receipts'][i//52];w=col['windows'][i//52]
                        draw.text((ox+5,37),f'{r["action"]} / {r["kind"]} / {r["phase"]}',fill='black')
                        slip=w['metrics']['whole']['contact_slip_percent']
                        slip_text='N/A' if slip is None else f'{slip:.1f}%'
                        draw.text((ox+5,54),f'skate {slip_text} | joint excess {w["metrics"]["joint_limit_max_deg"]:.1f}deg',fill='black')
                pipe.stdin.write(np.asarray(frame).tobytes())
                if i in indices:
                    k=indices.index(i);sheet.paste(frame,((k%3)*width,(k//3)*height))
            finally:pipe.stdin.close()
            assert pipe.wait()==0;tmp.replace(video);sheet.save(sheetfile,quality=90)
        manifest.append(dict(id=key,frames=frames,fps=25,raw_qpos=True,physics=False,cols=cols,video=str(video),video_sha256=sha(video),contact_sheet=str(sheetfile)))
        write(dest/'MANIFEST.json',manifest);print('RENDERED',key,frames,flush=True)
    finally:renderer.close()
    body='<!doctype html><meta charset="utf-8"><title>Unified trainer raw comparisons</title><h1>完整原始动作对照</h1><p>25 FPS，同视角，无物理稳定化。以各列标签、MANIFEST来源为准。包含失败案例；不是生产预览。</p>'
    for r in manifest:body+=f'<h2>{html.escape(r["id"])}</h2><video controls preload="metadata" width="100%" src="{Path(r["video"]).name}"></video><p>{r["frames"]} frames; <a href="{Path(r["contact_sheet"]).name}">接触图</a></p>'
    (dest/'index.html').write_text(body)
