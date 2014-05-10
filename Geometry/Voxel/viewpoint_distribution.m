% check the viewpoint distribution
function [azimuth, elevation, distance, width, height] = viewpoint_distribution(cls, is_imagenet)

opt = globals;

% load cad model
filename = sprintf(opt.path_cad, cls);
object = load(filename);
cad = object.(cls);

if is_imagenet == 1
    path_ann = sprintf(opt.path_ann_imagenet, cls);
else
    path_ann = sprintf(opt.path_ann_pascal, cls);
end
files = dir(fullfile(path_ann, '*.mat'));

N = numel(files);
azimuth = [];
elevation = [];
distance = [];
width = [];
height = [];
count = 0;
for i = 1:N
    % load annotation
    filename_ann = fullfile(path_ann, files(i).name);

    if exist(filename_ann) == 0
        errordlg('No annotation available for the image');
    else
        object = load(filename_ann);
        record = object.record;
        for j = 1:numel(record.objects)
            if strcmp(record.objects(j).class, cls) == 1
                a = record.objects(j).viewpoint.azimuth;
                e = record.objects(j).viewpoint.elevation;
                d = record.objects(j).viewpoint.distance;
                if d ~= 0
                    count = count + 1;
                    azimuth(count) = a;
                    elevation(count) = e;
                    distance(count) = d;
                    cad_index = record.objects(j).cad_index;
                    x = project_3d(0.8 * cad(cad_index).vertices, record.objects(j));
                    width(count) = max(x(:,1)) - min(x(:,1));
                    height(count) = max(x(:,2)) - min(x(:,2));
                end
            end
        end
    end
end