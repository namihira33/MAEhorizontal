#!/bin/bash

for preprocess in Add_BlackRects
do
    for lr in 1e-5 5e-5 1e-4 5e-4
    do
        for beta in 0 -1 0.9 0.99 0.999
        do
            for gamma in 2
            do
                for sampler in normal over under
                do
                python3 ./src/run.py cv=1 evaluate=0 mode=horizontal type=N epoch=150 lr=$lr preprocess=$preprocess beta=$beta gamma=$gamma sampler=$sampler 
                done
            done
        done
    done
done