import sys,json,math,random,numpy as np
sys.path.insert(0,'/home/cobalt_y/poolsight')
import guide as G
G.CFG.update({'felt':[150,170,1130,660],'r':14.5,'bounces':3,'real':True})
random.seed(17); cases=[]; want={(nb,pw):0 for nb in (1,2,3) for pw in (0.5,0.75,1.0)}
tries=0
while min(want.values())<7 and tries<900:
    tries+=1
    pw=random.choice((0.5,0.75,1.0)); G.CFG['power']=pw
    P=np.array([random.uniform(240,1040),random.uniform(240,590)])
    angs=np.radians(np.arange(0,360,0.25)); hits=[]
    for a in angs:
        d=np.array([math.cos(a),math.sin(a)])
        cue=P-d*29-d*150                      # straight-on shot from 150 behind the ghost spot
        if not (170<cue[0]<1110 and 190<cue[1]<640): hits.append(None); continue
        pts,pot,stop=G.rebound_path(P,d,3,G.obj_speed(P,d,cue))
        nb=len(pts)-2
        hits.append((G.nearest_pocket(pot),nb) if pot is not None and nb>=1 else None)
    runs=[];i=0
    while i<len(hits):
        if hits[i]:
            j=i
            while j+1<len(hits) and hits[j+1]==hits[i]: j+=1
            if j-i>=2 and want[(hits[i][1],pw)]<7: runs.append((i,j,hits[i]))
            i=j+1
        else: i+=1
    if not runs: continue
    i,j,(tgt,nb)=random.choice(runs)
    a=angs[(i+j)//2]; d=np.array([math.cos(a),math.sin(a)]); g=P-d*29; cue=g-d*150
    want[(nb,pw)]+=1
    cases.append([round(P[0],2),round(P[1],2),round(g[0],3),round(g[1],3),round(cue[0],2),round(cue[1],2),int(tgt),int(nb),pw,round(math.degrees(angs[j]-angs[i])+0.25,2)])
json.dump(cases,open('dbg/banks2.json','w')); print(len(cases),want); print(json.dumps(cases))
