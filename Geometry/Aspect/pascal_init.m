% initialize the PASCAL development kit

tmp = pwd;
cd('/home/yuxiang/Projects/Pose_Dataset/PASCAL/VOCdevkit');
addpath([cd '/VOCcode']);
VOCinit;
cd(tmp);
