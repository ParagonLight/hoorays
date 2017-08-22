#!/bin/sh
function=mkDataForMatlab.py
python $function fm 20
python $function fm 40
python $function fm 60
python $function fm 80
python $function delicious 20
python $function delicious 40
python $function delicious 60
python $function delicious 80
