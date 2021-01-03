if [ ! -d build ] ; then
    mkdir build
fi
cd build
cmake ..
make -j4
./cpp/mtcnn
#./ncnn/ntnn