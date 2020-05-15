# CyCADA Enhancement with 3-Level Cycle-Consistency
A Enhancement of [CyCADA](https://arxiv.org/pdf/1711.03213.pdf).
    
# Train image adaptation only (digits)

## 실행
```
$ git clone --recursive https://github.com/DevSusu/cycada_release.git cycada
$ cd cycada

$ python3 -m venv ENV
$ source ENV/bin/activate
(ENV) $ pip install -r requirements.txt
(ENV) $ cd cyclegan
(ENV) $ pip install -r requirements.txt

(ENV) $ cd ..
(ENV) $ python run_everything.py
```
