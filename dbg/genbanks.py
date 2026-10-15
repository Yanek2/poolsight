import sys,json,math,random,numpy as np
sys.path.insert(0,'/home/cobalt_y/poolsight')
import guide as G
G.CFG.update({'felt':[150,170,1130,660],'r':14.5,'bounces':2,'real':True})
random.seed(5); cases=[]; tries=0
while len(cases)<40 and tries<400:
    tries+=1
    P=np.array([random.uniform(240,1040),random.uniform(240,590)])
    angs=np.radians(np.arange(0,360,0.25)); hits=[]
    for a in angs:
        d=np.array([math.cos(a),math.sin(a)]); pts,pot=G.rebound_path(P,d,2)
        nb=len(pts)-2
        hits.append((G.nearest_pocket(pot),nb) if pot is not None and nb>=1 else None)
    runs=[];i=0
    while i<len(hits):
        if hits[i]:
            j=i
            while j+1<len(hits) and hits[j+1]==hits[i]: j+=1
            if j-i>=2: runs.append((i,j,hits[i]))
            i=j+1
        else: i+=1
    if not runs: continue
    i,j,(tgt,nb)=random.choice(runs)
    a=angs[(i+j)//2]; d=np.array([math.cos(a),math.sin(a)])
    cut=math.radians(random.uniform(0,15))*random.choice((-1,1))
    back=-np.array([d[0]*math.cos(cut)-d[1]*math.sin(cut), d[0]*math.sin(cut)+d[1]*math.cos(cut)])
    g=P-d*29; cue=g+back*random.uniform(120,200)
    if not (180<cue[0]<1100 and 200<cue[1]<630): continue
    cases.append([round(P[0],2),round(P[1],2),round(g[0],3),round(g[1],3),round(cue[0],2),round(cue[1],2),int(tgt),int(nb),round(math.degrees(angs[j]-angs[i])+0.25,2)])
json.dump(cases,open('dbg/banks.json','w')); print(len(cases)); print(json.dumps(cases))
