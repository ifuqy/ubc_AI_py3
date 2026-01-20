import pickle
import glob
from ubc_AI.data import pfdreader
import ubc_AI

AI_PATH = '/'.join(ubc_AI.__file__.split('/')[:-1])

with open(f'{AI_PATH}/trained_AI/pics_py3_14061.pkl', 'rb') as f:
    classifier = pickle.load(f)

pfd_files = glob.glob('./data/*.pfd')
AI_scores = classifier.report_score([pfdreader(f) for f in pfd_files])

output_lines = [f'{pfd_files[i]} {AI_scores[i]}' for i in range(len(pfd_files))]
output_text = '\n'.join(output_lines)

with open('clfresult.txt', 'w') as fout:
    fout.write(output_text)