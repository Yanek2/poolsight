import mss,time,numpy as np,cv2,cProfile,pstats,sys
sys.path.insert(0,'.')
import poolsight as P
s=mss.MSS(); mon=s.monitors[0]
def frame():
    if P.needs_full_frame(): base,reg=(0,0),mon
    else:
        x0,y0,x1,y1=P.table_region(mon['width'],mon['height']); base=(x0,y0); reg={'left':x0,'top':y0,'width':x1-x0,'height':y1-y0}
    shot=np.ascontiguousarray(np.asarray(s.grab(reg))[...,:3]); P.STATE['own']=P.own_mask(shot); return P.analyse(shot,base)
for _ in range(3): frame()
ts=[]
for _ in range(10):
    t=time.time(); frame(); ts.append((time.time()-t)*1000)
print('live frame ms',np.round(ts))
cProfile.run('for _ in range(5): frame()','dbg/prof')
pstats.Stats('dbg/prof').sort_stats('cumtime').print_stats(25)
