#!/bin/bash

#for preprocess in Add_BlackRects
#do
#    for sampler in normal over under
#    do
#        for gamma in 0 
#        do
#            for beta in 0
#            do
#                for lr in 1e-5 
#                do
#                python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N epoch=150 lr=$lr preprocess=$preprocess beta=$beta gamma=$gamma sampler=$sampler 
#                done
#            done
#        done
#    done
#done
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0 beta=-1.0 lr=0.0005 epoch=84
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0 beta=0.0 lr=0.0001 epoch=44
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0 beta=0.9 lr=5e-05 epoch=73
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0 beta=0.99 lr=0.0001 epoch=81
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0 beta=0.9990000000000001 lr=0.0001 epoch=44
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.1 beta=-1.0 lr=1e-05 epoch=113
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.1 beta=0.0 lr=5e-05 epoch=111
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.1 beta=0.9 lr=1e-05 epoch=16
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.1 beta=0.99 lr=1e-05 epoch=80
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.1 beta=0.9990000000000001 lr=1e-05 epoch=108
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.5 beta=-1.0 lr=5e-05 epoch=44
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.5 beta=0.0 lr=0.0001 epoch=44
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.5 beta=0.9 lr=1e-05 epoch=4
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.5 beta=0.99 lr=1e-05 epoch=44
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=0.5 beta=0.9990000000000001 lr=5e-05 epoch=65
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=1 beta=-1.0 lr=0.0005 epoch=84
#python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=1 beta=0.0 lr=0.0005 epoch=112
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=1 beta=0.9 lr=0.0001 epoch=48
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=1 beta=0.99 lr=5e-05 epoch=80
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=1 beta=0.9990000000000001 lr=0.0001 epoch=85
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=2 beta=-1.0 lr=0.0005 epoch=65
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=2 beta=0.0 lr=0.0005 epoch=72
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=2 beta=0.9 lr=5e-05 epoch=44
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=2 beta=0.99 lr=0.0005 epoch=5
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=normal gamma=2 beta=0.9990000000000001 lr=0.0005 epoch=5
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0 beta=-1.0 lr=0.0005 epoch=60
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0 beta=0.0 lr=0.0005 epoch=6
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0 beta=0.9 lr=5e-05 epoch=112
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0 beta=0.99 lr=0.0001 epoch=2
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0 beta=0.9990000000000001 lr=0.0001 epoch=79
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.1 beta=-1.0 lr=1e-05 epoch=112
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.1 beta=0.0 lr=5e-05 epoch=84
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.1 beta=0.9 lr=1e-05 epoch=6
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.1 beta=0.99 lr=1e-05 epoch=72
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.1 beta=0.9990000000000001 lr=1e-05 epoch=104
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.5 beta=-1.0 lr=0.0001 epoch=8
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.5 beta=0.0 lr=5e-05 epoch=24
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.5 beta=0.9 lr=1e-05 epoch=54
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.5 beta=0.99 lr=1e-05 epoch=16
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=0.5 beta=0.9990000000000001 lr=1e-05 epoch=65
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=1 beta=-1.0 lr=0.0005 epoch=116
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=1 beta=0.0 lr=0.0005 epoch=24
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=1 beta=0.9 lr=0.0001 epoch=44
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=1 beta=0.99 lr=0.0005 epoch=44
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=1 beta=0.9990000000000001 lr=0.0005 epoch=76
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=2 beta=-1.0 lr=0.0005 epoch=91
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=2 beta=0.0 lr=0.0005 epoch=40
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=2 beta=0.9 lr=0.0001 epoch=4
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=2 beta=0.99 lr=0.0005 epoch=117
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=over gamma=2 beta=0.9990000000000001 lr=0.0005 epoch=74
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0 beta=-1.0 lr=0.0005 epoch=72
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0 beta=0.0 lr=0.0005 epoch=90
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0 beta=0.9 lr=0.0001 epoch=30
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0 beta=0.99 lr=0.0001 epoch=60
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0 beta=0.9990000000000001 lr=5e-05 epoch=85
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.1 beta=-1.0 lr=0.0001 epoch=56
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.1 beta=0.0 lr=5e-05 epoch=106
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.1 beta=0.9 lr=1e-05 epoch=31
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.1 beta=0.99 lr=1e-05 epoch=100
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.1 beta=0.9990000000000001 lr=0.0001 epoch=119
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.5 beta=-1.0 lr=0.0005 epoch=76
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.5 beta=0.0 lr=0.0005 epoch=74
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.5 beta=0.9 lr=1e-05 epoch=116
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.5 beta=0.99 lr=0.0001 epoch=77
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=0.5 beta=-1.0 lr=0.0005 epoch=76
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=1 beta=-1.0 lr=0.0005 epoch=37
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=1 beta=0.0 lr=0.0005 epoch=3
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=1 beta=0.9 lr=5e-05 epoch=96
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=1 beta=0.99 lr=5e-05 epoch=8
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=1 beta=0.9990000000000001 lr=0.0005 epoch=80
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=2 beta=-1.0 lr=0.0001 epoch=114
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=2 beta=0.0 lr=0.0005 epoch=116
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=2 beta=0.9 lr=0.0001 epoch=85
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=2 beta=0.99 lr=0.0005 epoch=101
python3 ./src/run.py cv=0 evaluate=1 mode=horizontal type=N preprocess=Add_BlackRects sampler=under gamma=2 beta=0.9990000000000001 lr=5e-05 epoch=114