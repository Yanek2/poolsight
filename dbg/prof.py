import cv2,cProfile,pstats,time,sys
sys.path.insert(0,'.')
import poolsight as P
img=cv2.imread('dbg/n0.png'); P.analyse(img); P.analyse(img)
t=time.time(); [P.analyse(img) for _ in range(5)]; print('ms',(time.time()-t)/5*1000)
cProfile.run('P.analyse(img)','dbg/prof')
pstats.Stats('dbg/prof').sort_stats('cumtime').print_stats(20)
