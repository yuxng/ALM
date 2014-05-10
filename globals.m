function opt = globals()

opt.pascal3d = '/home/yuxiang/Projects/Pose_Dataset/PASCAL3D+_release1.1';

opt.path_img_pascal = [opt.pascal3d '/Images/%s_pascal'];
opt.path_ann_pascal = [opt.pascal3d '/Annotations/%s_pascal'];
opt.path_img_imagenet = [opt.pascal3d '/Images/%s_imagenet'];
opt.path_ann_imagenet = [opt.pascal3d '/Annotations/%s_imagenet'];
opt.path_pascal = [opt.pascal3d '/PASCAL/VOCdevkit'];
opt.path_cad = [opt.pascal3d '/CAD/%s.mat'];
opt.path_alm = [opt.pascal3d '/ALM'];