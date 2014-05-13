function opt = globals()

paths = {'/home/yuxiang/Projects/Pose_Dataset/PASCAL3D+_release1.1', ...
    '/home/ma/yxiang/Projects/PASCAL3D+_release1.1'};

for i = 1:numel(paths)
    if exist(paths{i}, 'dir') ~= 0
        pascal3d = paths{i};
        break;
    end
end
 
opt.pascal3d = pascal3d;
opt.path_img_pascal = [opt.pascal3d '/Images/%s_pascal'];
opt.path_ann_pascal = [opt.pascal3d '/Annotations/%s_pascal'];
opt.path_img_imagenet = [opt.pascal3d '/Images/%s_imagenet'];
opt.path_ann_imagenet = [opt.pascal3d '/Annotations/%s_imagenet'];
opt.path_pascal = [opt.pascal3d '/PASCAL/VOCdevkit'];
opt.path_cad = [opt.pascal3d '/CAD/%s.mat'];
opt.path_alm = [opt.pascal3d '/ALM'];