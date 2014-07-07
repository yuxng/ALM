% Set up global variables used throughout the code

% dataset to use
if exist('setVOCyear', 'var') == 1
  VOCyear = setVOCyear;
  clear('setVOCyear');
else
  VOCyear = '2012';
end

% directory for caching models, intermediate data, and results
cachedir = 'data/';

if exist(cachedir, 'dir') == 0
  unix(['mkdir -p ' cachedir]);
  if exist([cachedir 'learnlog/'], 'dir') == 0
    unix(['mkdir -p ' cachedir 'learnlog/']);
  end
end

% directory for LARGE temporary files created during training
tmpdir = 'data/';

if exist(tmpdir, 'dir') == 0
  unix(['mkdir -p ' tmpdir]);
end

% should the tmpdir be cleaned after training a model?
cleantmpdir = true;

% PASCAL3D+ directories
PASCAL3Dopts.pascal3d = '/home/yuxiang/Projects/Pose_Dataset/PASCAL3D+_release1.1';
PASCAL3Dopts.path_img_pascal = [PASCAL3Dopts.pascal3d '/Images/%s_pascal'];
PASCAL3Dopts.path_ann_pascal = [PASCAL3Dopts.pascal3d '/Annotations/%s_pascal'];
PASCAL3Dopts.path_img_imagenet = [PASCAL3Dopts.pascal3d '/Images/%s_imagenet'];
PASCAL3Dopts.path_ann_imagenet = [PASCAL3Dopts.pascal3d '/Annotations/%s_imagenet'];
PASCAL3Dopts.path_cad = [PASCAL3Dopts.pascal3d '/CAD/%s.mat'];

% directory with PASCAL VOC development kit and dataset
VOCdevkit = [PASCAL3Dopts.pascal3d '/PASCAL/VOCdevkit'];