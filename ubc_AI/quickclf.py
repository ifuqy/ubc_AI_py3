import pickle, glob, ubc_AI
from ubc_AI.data import pfdreader
AI_PATH = '/'.join(ubc_AI.__file__.split('/')[:-1])
classifier = pickle.load(open(AI_PATH+'/trained_AI/pics_py3_14061.pkl','rb'))
#pfdfile = glob.glob('*.pfd') + glob.glob('*.ar') + glob.glob('*.ar2') + glob.glob('*.spd')
pfdfile = glob.glob('./data/*.pfd')
AI_scores = classifier.report_score([pfdreader(f) for f in pfdfile])

text = '\n'.join(['%s %s' % (pfdfile[i], AI_scores[i]) for i in range(len(pfdfile))])
fout = open('clfresult.txt', 'w')
fout.write(text)
fout.close()
