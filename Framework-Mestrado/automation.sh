#!/bin/bash
rm output/classifier.opf && rm output/saida.dat.out
/home/messias/Dropbox/GitHub/Clustering/LibOPF-master/tools/./txt2opf saida.txt saida.dat
rm saida.txt
/home/messias/Dropbox/GitHub/Clustering/LibOPF-master/bin/opf_cluster saida.dat 100 1 0.2
mv classifier.opf /home/messias/Dropbox/GitHub/Framework-Mestrado/output/ && mv saida.dat.out /home/messias/Dropbox/GitHub/Framework-Mestrado/output/
rm saida.dat