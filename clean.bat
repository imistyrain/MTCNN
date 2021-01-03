@echo off
del bin\*.pdb
del windows\*.pdb
del x64\*.pdb
if exist windows\Release rd windows\Release /s /q
if exist MTCNN-light\x64 rd MTCNN-light\x64 /s /q
if exist ncnn\x64 rd ncnn\x64 /s /q
if exist Fast-MTCNN\x64 rd Fast-MTCNN\x64 /s /q
pause