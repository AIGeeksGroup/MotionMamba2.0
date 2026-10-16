from pathlib import Path
from ..io import read,write

def compare(parent,current):
    base=read(Path(parent)/'RESULT.json');cur=read(Path(current)/'RESULT.json');b={r['id']:r for r in base['rows']};losses=[];severe=[];deltas=[];invalid=[]
    for r in cur['rows']:
        q=b[r['id']];assert q['seed']==r['seed'] and q['source']['source_family_id']==r['source']['source_family_id']
        if not r['valid']:invalid.append(r['id'])
        for ei,(pe,ce) in enumerate(zip(q['events'],r['events'])):
            for k,v in pe['checks'].items():
                if v and not ce['checks'].get(k,False):losses.append(dict(id=r['id'],event=ei,check=k,kind=ce['kind']))
        for wi,(pw,cw) in enumerate(zip(q['windows'],r['windows'])):
            pm,cm=pw['metrics'],cw['metrics'];ps,cs=pm['whole']['contact_slip_percent'],cm['whole']['contact_slip_percent']
            pj,cj=pm['joint_limit_max_deg'],cm['joint_limit_max_deg'];why=[]
            if cj>30 and cj-pj>5:why.append('new_severe_joint')
            if ps is not None and cs is not None and cs-ps>20 and cm['whole']['contact_speed_m_s']>1.5*pm['whole']['contact_speed_m_s']:why.append('contact_collapse')
            if cm['seam_root_m']>.5 and cm['seam_root_m']>pm['seam_root_m']+.1:why.append('root_seam')
            if cm['seam_joint_rms_deg']>45 and cm['seam_joint_rms_deg']>pm['seam_joint_rms_deg']+10:why.append('joint_seam')
            if why:severe.append(dict(id=r['id'],source=r['source']['source_family_id'],noise=r['noise_index'],window=wi,event=r['receipts'][wi]['event'],why=why))
            deltas.append(dict(id=r['id'],window=wi,joint_delta_deg=cj-pj,slip_delta_pp=cs-ps if cs is not None and ps is not None else None,path_delta_m=cw['path_m']-pw['path_m'],heading_delta_deg=cw['heading_deg']-pw['heading_deg'],parent_joint=pj,candidate_joint=cj))
    # Confirmation uses SAME source/task/phase and two registered noises.
    confirmed=[]
    for x in severe:
        other=next((y for y in severe if y['source']==x['source'] and y['event']==x['event'] and y['noise']!=x['noise'] and set(y['why'])&set(x['why'])),None)
        if other:confirmed.append(x)
    phase_confirmed=[]
    for x in losses:
        if x['check'] not in ['progress_and_legs_by_deadline','sustained_after_deadline','holds_after_buffer','stopped_by_end_window1']:continue
        otherid=x['id'][:-1]+('1' if x['id'].endswith('0') else '0')
        if any(y['id']==otherid and y['event']==x['event'] and y['check']==x['check'] for y in losses):phase_confirmed.append(x)
    # Basic old capabilities are protected independently of interaction total.
    pb={r['id']:r for r in base['foundation']['rows']};basic=[]
    for r in cur['foundation']['rows']:
        p=pb[r['id']]
        if p['function_pass'] and not r['function_pass']:basic.append(dict(id=r['id'],issue='semantic_loss'))
        pj=p['metrics']['joint_limit_max_deg'];cj=r['metrics']['joint_limit_max_deg']
        if cj>30 and cj-pj>5:basic.append(dict(id=r['id'],issue='severe_joint'))
    basic_confirmed=[r for r in basic if any(x['id']==r['id'].replace('_0_cold','_1_cold').replace('_0_gt','_1_gt') and x!=r and x['issue']==r['issue'] for x in basic)]
    result=dict(stop=bool(invalid or confirmed or phase_confirmed or basic_confirmed),invalid=invalid,severe_instances=severe,confirmed_severe=confirmed,old_passed_checks_lost=losses,confirmed_old_stages_lost=phase_confirmed,basic_regressions=basic,basic_confirmed=basic_confirmed,deltas=deltas,all_pass=sum(r['all_pass'] for r in cur['rows']),parent_all_pass=sum(r['all_pass'] for r in base['rows']),unconfirmed_blocks_qualification=bool(severe),scope='same source/noise/visible protocol paired retention; no claim final qualification')
    write(Path(current)/'COMPARISON.json',result);return result

