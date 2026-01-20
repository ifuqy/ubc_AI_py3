# ubc_AI_py3
**Author:** Qiuyang Fu (NAOC)  
**E-mail:** fuqy@bao.ac.cn 

This repository provides a **Python 3 implementation** of pulsar candidate classification code:  
https://github.com/zhuww/ubc_AI

---

## Requirements

This project has been tested with the following dependencies:

- Python 3.10.12
- numpy==1.22.4
- matplotlib==3.5.1
- scipy==1.11.3
- scikit-learn==1.1.3
- theano==1.0.5

System-level dependencies (Linux):

- libblas-dev
- python-dev
- imagemagick

---

### Tests:
To run the tests, use the following command:
python quickclf.py

---

### Explanation:

Running 'quickclf.py' will score the '.pfd' files in the 'data' directory of the current folder using 'pics_py3_14061.pkl' and output the results to 'clfresult.txt'.

Consider adding ubc_AI to your $PYTHONPATH.

---

### Note:
Be careful: When there are more than one cpu available, the default behavior of the code is to use multi-threading. The code will use up to 20 threads or max(cpu)-1. If you want to turn this behavior off, you can change the default max_threads parameter in file: threadit.py.