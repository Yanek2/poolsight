import cv2,json,numpy as np,sys
d=json.load(open('dbg/shot_result.json')); tr=np.load('dbg/shot_tracks.npy',allow_pickle=True)
im=cv2.imread('dbg/shot_before.png')
cols=[(0,0,255),(0,255,0),(255,0,255),(0,165,255),(255,255,0),(255,0,0)]
k=0
for t in tr:
    t=np.array(t,float)
    if np.linalg.norm(t[-1]-t[0])<10: continue
    c=cols[k%len(cols)]; k+=1
    for a,b in zip(t,t[1:]): cv2.line(im,tuple(a.astype(int)),tuple(b.astype(int)),c,2)
    for p in t[::3]: cv2.circle(im,tuple(p.astype(int)),2,c,-1)
x0,y0,x1,y1=[int(v) for v in sys.argv[1:5]] if len(sys.argv)>4 else (300,300,1640,1000)
cv2.imwrite('dbg/viz.png',im[y0:y1,x0:x1])
