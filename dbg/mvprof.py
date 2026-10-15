import cv2,glob,time,cProfile,pstats,sys
sys.path.insert(0,'.')
import poolsight as P
fr=[cv2.imread(f) for f in sorted(glob.glob('dbg/mv*.png'))]
P.analyse(fr[0])
def run():
    for f in fr: P.analyse(f)
t=time.time(); run(); print('ms/frame while moving',(time.time()-t)/len(fr)*1000)
cProfile.run('run()','dbg/prof'); pstats.Stats('dbg/prof').sort_stats('tottime').print_stats(12)
