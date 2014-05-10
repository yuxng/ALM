% initialize the PASCAL development kit

tmp = pwd;
opt = globals;
cd(opt.path_pascal);
addpath([cd '/VOCcode']);
VOCinit;
cd(tmp);
