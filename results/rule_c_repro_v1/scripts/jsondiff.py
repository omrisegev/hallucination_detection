import json,sys
A=json.load(open(sys.argv[1])); B=json.load(open(sys.argv[2])); tol=float(sys.argv[3]) if len(sys.argv)>3 else 0
n=[0,0]
def walk(a,b,p=''):
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)): walk(a.get(k),b.get(k),p+'/'+str(k))
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)): walk(x,y,p+f'[{i}]')
    else:
        n[0]+=1
        if isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and abs(a-b)<=tol: return
        if a!=b: n[1]+=1; print('DIFF',p,'new=',a,'bundle=',b)
walk(A,B); print(f'leaves={n[0]} diffs={n[1]}')
