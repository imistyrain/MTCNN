# widerface
#please open chrome https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?usp=sharing

#landmark
if [ ! -f labdmark] then;
    wget http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip
    unzip train.zip
    mv train landmark
fi