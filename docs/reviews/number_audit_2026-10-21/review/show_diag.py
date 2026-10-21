import json,sys
d=json.load(open(sys.argv[1]))
def show(x,depth=0,maxd=4):
    if isinstance(x,dict):
        for k,v in x.items():
            if isinstance(v,(dict,list)) and depth<maxd: print('  '*depth+str(k)+':'); show(v,depth+1,maxd)
            else: print('  '*depth+str(k)+': '+str(v)[:160])
    elif isinstance(x,list):
        print('  '*depth+'[list len %d] '%len(x)+str(x)[:160])
show(d)
