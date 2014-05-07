% prepare testing data for a aspect layout model of a category
% cls: object category
function model_test(cls, subtype, is_imagenet)

if nargin < 2
    subtype = [];
    cls_data = cls;
else
    cls_data = subtype;
end

if nargin < 3
    is_imagenet = 0;
end

% read positive test images
fprintf('Read postive samples\n');
maxnum = 64;
pos = read_test_positive(cls, subtype, is_imagenet, maxnum);
fprintf('%d positive samples\n', numel(pos));

neg = [];

% write training samples to file
fprintf('Writing data\n');
filename = sprintf('data/%s.tst', cls_data);
write_data(filename, pos, neg);


% read positive training images
function pos = read_test_positive(cls, subtype, is_imagenet, maxnum)

opt = globals();
if is_imagenet == 0
    path_image = sprintf(opt.path_img_pascal, cls);
    path_anno = sprintf(opt.path_ann_pascal, cls);
    ext = 'jpg';

    filename = sprintf('ids_%s.mat', cls);
    object = load(filename);
    ids = object.ids_val;
else
    path_image = sprintf(opt.path_img_imagenet, cls);
    path_anno = sprintf(opt.path_ann_imagenet, cls);
    ext = 'JPEG';

    files = dir([path_anno '/*.mat']);
    N = numel(files);
    ids = cell(N, 1);
    for i = 1:N
        ids{i} = files(i).name(1:end-4);
    end
end
N = numel(ids);

count = 0;
for i = 1:N
    file_ann = sprintf('%s/%s.mat', path_anno, ids{i});
    image = load(file_ann);
    record = image.record;
    
    flag = 0;
    for j = 1:numel(record.objects)
        object = record.objects(j);
        % filtering objects
        if strcmp(cls, object.class) == 1 && (isempty(subtype) == 1 || ...
                (isempty(subtype) == 0 && strcmp(object.subtype, subtype) == 1))
            flag = 1;
            break;
        end
    end
    if flag == 0
        continue;
    end
    
    file_img = sprintf('%s/%s.%s', path_image, ids{i}, ext);
    I = imread(file_img);
    
    count = count + 1;
    pos(count).image = I;
    pos(count).object_label = -1;
    pos(count).part_label = [];
    pos(count).cad_label = [];
    pos(count).view_label = [];
    pos(count).bbox = [];       
    pos(count).occlusion = [];    
        
    if count >= maxnum
        return;
    end
end