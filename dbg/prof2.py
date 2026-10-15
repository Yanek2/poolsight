import cv2,time,cProfile,pstats,sys
sys.path.insert(0,'.')
import poolsight as P
img=cv2.imread('dbg/ui3.png')
for _ in range(3): P.analyse(img)
t=time.time()
for _ in range(10): P.analyse(img)
print('ms/frame steady', (time.time()-t)*100)
cProfile.run('for _ in range(5): P.analyse(img)','dbg/prof')
pstats.Stats('dbg/prof').sort_stats('tottime').print_stats(14)
