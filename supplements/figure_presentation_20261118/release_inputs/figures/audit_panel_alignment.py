"""Study-authored portable geometry gate for the two supplied plotters.

This compact checker is not a redistributed third-party QA implementation. It
checks final axes geometry; an optional external QA helper can be selected by
the plotters when a broader journal-preflight tool is available.
"""
from pathlib import Path
import json

def require_matplotlib_panel_alignment(fig,axes=None,panel_ids=None,row_groups=None,column_groups=None,
        exemptions=None,json_out=None,overlay_svg=None,tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        require_panel_labels=False,strict=True,**kwargs):
    fig.canvas.draw();axes=list(fig.axes if axes is None else axes)
    ids=list(panel_ids or [chr(97+i) for i in range(len(axes))]);width,height=fig.get_size_inches()*72
    rect={name:[ax.get_position().x0*width,ax.get_position().y0*height,ax.get_position().x1*width,ax.get_position().y1*height] for name,ax in zip(ids,axes)}
    failures=[];checks=[]
    def close(group,coordinates,reason):
        for index in coordinates:
            values=[rect[n][index] for n in group]
            delta=max(values)-min(values);checks.append(dict(group=group,check=reason,coordinate=index,range_pt=delta))
            if delta>tolerance_pt:failures.append(dict(group=group,check=reason,range_pt=delta))
    def inferred(indices):
        groups=[];used=set()
        for name in ids:
            if name in used:continue
            group=[n for n in ids if all(abs(rect[n][i]-rect[name][i])<=tolerance_pt for i in indices)]
            if len(group)>1:groups.append(group);used.update(group)
        return groups
    rows=row_groups if row_groups is not None else inferred([1,3])
    columns=column_groups if column_groups is not None else inferred([0,2])
    for group in rows:
        close(group,[1,3],'shared row edges')
        widths=[rect[n][2]-rect[n][0] for n in group]
        if max(widths)-min(widths)>tolerance_pt:failures.append(dict(group=group,check='row widths',range_pt=max(widths)-min(widths)))
    for group in columns:close(group,[0,2],'shared column edges')
    for i,name in enumerate(ids):
        a=rect[name]
        for other in ids[i+1:]:
            b=rect[other]
            if min(a[2],b[2])-max(a[0],b[0])>tolerance_pt and min(a[3],b[3])-max(a[1],b[1])>tolerance_pt:
                failures.append(dict(group=[name,other],check='overlapping plot areas'))
    renderer=fig.canvas.get_renderer();labels={}
    for name,ax in zip(ids,axes):
        candidates=[t for t in ax.texts if len(t.get_text())==1 and t.get_text().islower()]
        if candidates:
            box=candidates[0].get_window_extent(renderer);labels[name]=[box.x0/fig.dpi*72,box.y1/fig.dpi*72]
        elif require_panel_labels:failures.append(dict(group=[name],check='missing panel label'))
    for groups,coordinate in [(rows,1),(columns,0)]:
        for group in groups:
            values=[labels[n][coordinate] for n in group if n in labels]
            if len(values)>1 and max(values)-min(values)>tolerance_pt:failures.append(dict(group=group,check='panel label anchors'))
    result=dict(verdict='FIX BEFORE DELIVERY' if failures else 'PASS',checker='Study-authored axes/label geometry gate',
        scope='Final plot-area row/column edges, widths, overlap and available panel-label anchors; not PDF stroke collision analysis',
        width_pt=float(width),height_pt=float(height),tolerance_pt=tolerance_pt,rectangles_pt=rect,row_groups=rows,column_groups=columns,
        declared_exemptions=exemptions or [],checks=checks,failures=failures)
    if json_out:Path(json_out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if overlay_svg:
        elements=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">']
        for n,(x0,y0,x1,y1) in rect.items():elements.append(f'<rect x="{x0}" y="{height-y1}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#0072B2"/><text x="{x0}" y="{height-y1+9}" font-size="8">{n}</text>')
        Path(overlay_svg).write_text('\n'.join(elements+['</svg>']),encoding='utf-8')
    if strict and failures:raise ValueError('Plot-area alignment failed: '+json.dumps(failures))
    return result
