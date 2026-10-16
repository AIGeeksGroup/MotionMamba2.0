"""Plot saved paired retention metrics, without new generation."""
from pathlib import Path
import numpy as np
from .io import read,write,sha

def render(config,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True);spec=read(dest/'SOURCES.json')
    fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True,constrained_layout=True);used=[]
    for ni in [0,1]:
        for label,folder in spec.items():
            path=Path(folder)/f'S_3_n{ni}.json';row=read(path);ws=row['windows'];x=np.arange(len(ws))
            values=[np.array([w['metrics']['whole']['contact_slip_percent'] for w in ws]),np.array([w['metrics']['whole']['contact_fraction']*100 for w in ws]),np.array([w['contact_diagnostics']['geometric_low_foot_speed_m_s'] if w['contact_diagnostics']['geometric_low_foot_speed_m_s'] is not None else np.nan for w in ws])]
            for ax,v in zip(axes[:,ni],values):ax.plot(x,v,marker='o',markersize=3,label=label)
            used.append(dict(label=label,noise=ni,source=str(path),sha256=sha(path),values=[v.tolist() for v in values]))
        axes[0,ni].set_title(f'S3 stop / resume / release | noise {ni}')
        for ax in axes[:,ni]:
            ax.axvspan(5.75,6.25,color='red',alpha=.12)
            for pos in [2.5,6.5,9.5,13.5]:ax.axvline(pos,color='grey',ls=':',lw=.7)
            ax.grid(alpha=.2);ax.set_xlim(-.2,17.2)
        axes[-1,ni].set_xlabel('Generated window (52 frames, zero-based)')
    for ax,label in zip(axes[:,0],['Slip under predicted contact (%)','Predicted contact fraction (%)','Geometric low-foot speed (m/s)']):ax.set_ylabel(label)
    axes[0,0].legend(loc='best');fig.suptitle('Paired retention: contact proxy and geometric diagnostic differ\nRed: confirmed stop-gate window; no ground-truth contact / physics labels',fontsize=12)
    fig.savefig(dest/'retention_contact.png',dpi=160);fig.savefig(dest/'retention_contact.pdf');plt.close(fig)
    write(dest/'DATA.json',dict(rows=used,scope='Same saved motions; predicted contact and geometric low-feet masks are different diagnostics, not physical ground truth'))
