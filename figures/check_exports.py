"""Check one-page figure text bounds and optional frozen-reference geometry with Poppler.

This checks text extraction and page bounds, not PDF stroke collisions. It does
not replace visual inspection. Only parser input is cleaned of XML control codes.
"""
from pathlib import Path
import argparse, hashlib, json, re, subprocess
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
NAMES=['fig13_reference_filter_risk','fig21_allocation_new10_acc','fig22_allocation_extension']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(pdf):
    q=subprocess.run(['pdftotext','-bbox-layout',str(pdf),'-'],capture_output=True,check=True)
    text=q.stdout.decode('utf-8');controls=re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',text)
    tree=ET.fromstring(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]','\ufffd',text));pages=[]
    for page in tree.iter():
        if page.tag.split('}')[-1]!='page':continue
        width,height=float(page.attrib['width']),float(page.attrib['height']);words=[];outside=[]
        for word in page.iter():
            if word.tag.split('}')[-1]!='word':continue
            value=''.join(word.itertext());bounds=[float(word.attrib[k]) for k in ['xMin','yMin','xMax','yMax']]
            words.append(dict(text=value,bounds=bounds))
            if bounds[0]<-.5 or bounds[1]<-.5 or bounds[2]>width+.5 or bounds[3]>height+.5:outside.append(value)
        pages.append(dict(width_pt=width,height_pt=height,words=words,outside_page_words=outside))
    return dict(pages=pages,parser_control_codes_replaced=len(controls))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=HERE/'generated')
    p.add_argument('--reference-directory',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    reports=[]
    for name in NAMES:
        pdf=a.directory/(name+'.pdf');content=extract(pdf);assert len(content['pages'])==1
        page=content['pages'][0];assert not page['outside_page_words'],name
        result=dict(file=name+'.pdf',sha256=sha(pdf),pages=1,width_mm=page['width_pt']*25.4/72,height_mm=page['height_pt']*25.4/72,
            words=len(page['words']),outside_page_words=0,parser_control_codes_replaced=content['parser_control_codes_replaced'])
        if a.reference_directory:
            old=a.reference_directory/(name+'.pdf');reference=extract(old)
            result.update(reference_sha256=sha(old),reference_word_text_and_bbox_identical=content['pages']==reference['pages'])
        reports.append(result)
    report=dict(status='PASS_TEXT_BOUNDS',figures=reports,backend='Matplotlib exports; Poppler geometry inspection',
        collision_audit='PyMuPDF unavailable; no new PyMuPDF text-stroke collision pass is claimed',
        interpretation='Unchanged text coordinates plus numerical/render checks support faithful redraw; manually inspect all panels')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
