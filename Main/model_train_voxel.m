% prepare training data for a aspect layout model of a category
% cls: object category
function model_train_voxel(cls, cls_data, subtype, is_imagenet)

if nargin < 2
    subtype = [];
    cls_data = cls;
end

if nargin < 3
    subtype = [];
end

if nargin < 4
    is_imagenet = 0;
end

root_path = pwd;
cd('..');
addpath(pwd);
cd(root_path);

is_flip = 1;
% small number for debugging
maxnum = inf;
data_dir = 'data';

% load cad model, currently only one cad model for all the categories
cad_num = 1;
cad = cell(cad_num,1);
object = load(sprintf('../Geometry/Voxel/%s.mat', cls_data));
cad{1} = object.(cls_data);

% write cad model to file
filename = sprintf('%s/%s.cad', data_dir, cls_data);
write_cad(filename, cad);

% wrap positive training images
fprintf('Read wrapped positive samples\n');
is_wrap = 1;
pos = read_positive(cls, subtype, is_imagenet, is_wrap, cad, maxnum);
fprintf('%d wrapped positive samples\n', numel(pos));

if is_flip
    % flip positive samples
    pos_flip = flip_positive(pos, cad);
    fprintf('%d flipped positive samples\n', numel(pos_flip));
    pos = [pos pos_flip];
end
neg = [];

% write training samples to file
fprintf('Writing wrapped positives\n');
if is_flip
    filename = sprintf('%s/%s_wrap_flip.dat', data_dir, cls_data);
else
    filename = sprintf('%s/%s_wrap.dat', data_dir, cls_data);
end
write_data(filename, pos, neg);


% read unwrapped positive training images
fprintf('Read unwrapped positive samples\n');
is_wrap = 0;
pos = read_positive(cls, subtype, is_imagenet, is_wrap, cad, maxnum);
fprintf('%d unwrapped positive samples\n', numel(pos));

if is_flip
    % flip positive samples
    pos_flip = flip_positive(pos, cad);
    fprintf('%d flipped positive samples\n', numel(pos_flip));
    pos = [pos pos_flip];
end
neg = [];

% write training samples to file
fprintf('Writing unwrapped positives\n');
if is_flip
    filename = sprintf('%s/%s_unwrap_flip.dat', data_dir, cls_data);    
else
    filename = sprintf('%s/%s_unwrap.dat', data_dir, cls_data);
end
write_data(filename, pos, neg);


% % sample negative training images for VOC pascal
% fprintf('Randomize negative PASCAL samples\n');
% maxnum = 96;
% neg = rand_negative(cls, maxnum);
% fprintf('%d negative samples\n', numel(neg));
%  
% % write training samples to file
% fprintf('Writing negative samples\n');
% filename = sprintf('%s/%s_neg.dat', data_dir, cls_data);
% pos = [];
% write_data(filename, pos, neg);


% read positive training images
function pos = read_positive(cls, subtype, is_imagenet, is_wrap, cads, maxnum)

opt = globals();
if is_imagenet == 0
    path_image = sprintf(opt.path_img_pascal, cls);
    path_anno = sprintf(opt.path_ann_pascal, cls);
    ext = 'jpg';
    
    pascal_init;
    [ids, label] = textread(sprintf(VOCopts.imgsetpath, [cls '_train']), '%s%d');
    ids(label == -1) = [];
    N = numel(ids);
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

count = 0;
scale = 1.5;
for i = 1:N
    file_ann = sprintf('%s/%s.mat', path_anno, ids{i});
    image = load(file_ann);
    record = image.record;
    
    file_img = sprintf('%s/%s.%s', path_image, ids{i}, ext);
    I = imread(file_img);
    if size(I, 3) == 1
        I = repmat(I, [1 1 3]);
    end
    
    for j = 1:numel(record.objects)
        object = record.objects(j);
        % filtering objects
        if strcmp(cls, object.class) == 0 || object.viewpoint.distance == 0 || ...
                (isfield(object, 'difficult') == 1 && object.difficult == 1) || ...
                (isempty(subtype) == 0 && strcmp(object.subtype, subtype) == 0) || ...  
                is_occld_trunc(object) == 1
%                 is_trunc(object) == 1 || ...
%                 object.viewpoint.distance > cads{1}.distance(end) || ...
%                 object.viewpoint.distance < cads{1}.distance(1)           
            continue;
        end
        
        cad = cads{1};
        % part2d = project_cad(cad, object.viewpoint.azimuth, object.viewpoint.elevation, object.viewpoint.distance);
        view_label = find_closest_view(cad, object);
        part2d = cad.parts2d(view_label);
        
        % part label
        part_label = zeros(numel(cad.pnames), 2);
        
        % determine the bounding box
        if is_occld_trunc(object)
            index = find_interval(object.viewpoint.azimuth, cad.view_num);
            root_index = find(cad.roots == 1);
            ind = root_index(index);
            center = part2d.centers(ind,:) - [cad.viewport/2 cad.viewport/2] + [object.viewpoint.px object.viewpoint.py];
            part = part2d.(cad.pnames{ind});
            part = part + repmat(center, size(part,1), 1);
            x1 = min(part(:,1));
            x2 = max(part(:,1));
            y1 = min(part(:,2));
            y2 = max(part(:,2));
        else
            bbox = object.bbox;
            x1 = bbox(1);
            x2 = bbox(3);
            y1 = bbox(2);
            y2 = bbox(4);
        end
        
        for k = 1:numel(cad.pnames)
            if isempty(part2d.(cad.pnames{k})) == 1
                continue;
            end
            if cad.roots(k) == 1
                part_label(k,:) = [(x1+x2)/2 (y1+y2)/2];
            else
                index = mod(k-1, cad.subpart_num+1);
                index_x = floor((index-1)/cad.subpart_size(1));
                index_y = mod(index-1, cad.subpart_size(1));
                width = x2 - x1;
                height = y2 - y1;
                x1_part = x1 + (width / cad.subpart_size(1)) * index_y;
                x2_part = x1_part + width / cad.subpart_size(1);
                y1_part = y1 + (height / cad.subpart_size(2)) * index_x;
                y2_part = y1_part + height / cad.subpart_size(2);
                part_label(k,:) = [(x1_part+x2_part)/2 (y1_part+y2_part)/2];        
            end               
        end
 
        count = count + 1;
        % object label
        pos(count).object_label = 1;
        % cad label
        pos(count).cad_label = 1;
        % view label
        pos(count).view_label = view_label;
        
        % generate bounding box
        if is_wrap
            bbox = generate_bbox(cad, part2d, part_label);
        else
            bbox = [object.bbox(1) object.bbox(2) object.bbox(3)-object.bbox(1) object.bbox(4)-object.bbox(2)];
        end
        
        % wrap positive
        if is_wrap == 0 || is_occld_trunc(object) == 1
            im = I;
            sx = 1;
            sy = 1;
        else
            padx = (object.bbox(3)-object.bbox(1)) / 10;
            pady = (object.bbox(4)-object.bbox(2)) / 10;
            sx = bbox(3) / (object.bbox(3)-object.bbox(1) + padx);
            numcols = round(size(I,2) * sx);
            sy = bbox(4) / (object.bbox(4)-object.bbox(2) + pady);
            numrows = round(size(I,1) * sy);
            im = imresize(I, [numrows numcols], 'bilinear');

            bbox(1) = (bbox(1) + bbox(3)/2) * sx - bbox(3)/2;
            bbox(2) = (bbox(2) + bbox(4)/2) * sy - bbox(4)/2;
        end
        
        % enlarge the bbox by scale
        width = scale * bbox(3);
        height = scale * bbox(4);
        x = bbox(1) + bbox(3)/2;
        y = bbox(2) + bbox(4)/2;
        % b1 is enlarged bounding box
        b1 = [x - width/2, y - height/2, x + width/2, y + height/2];
        % b2 is the bounding box of the image
        b2 = [1, 1, size(im,2), size(im,1)];
        % b3 is union of b1 and b2
        b3 = [min(b1(1), b2(1)), min(b1(2), b2(2)), max(b1(3), b2(3)), max(b1(4), b2(4))];
        origin = [b3(1) b3(2)];
        origin_bbox = [bbox(1)+bbox(3)/2-width/2, bbox(2)+bbox(4)/2-height/2];
        pos(count).bbox = [bbox(1)-origin_bbox(1)+1 bbox(2)-origin_bbox(2)+1 bbox(3) bbox(4)];
        
        % image
        w = round(b3(3) - b3(1) + 1);
        h = round(b3(4) - b3(2) + 1);
        image = zeros(h, w, 3, 'uint8');
        b4 = [b2(1)-origin(1)+1 b2(2)-origin(2)+1 b2(3) b2(4)];
        b4 = round([b4(1) b4(2) b4(1)+b4(3)-1 b4(2)+b4(4)-1]);
        image(b4(2):b4(4), b4(1):b4(3), :) = im;
        bbox_crop = [bbox(1)-origin(1)+1 bbox(2)-origin(2)+1 bbox(3) bbox(4)];
        pos(count).image = crop_bbox(image, bbox_crop, scale);        
        
        % part_label
        pos(count).part_label = part_label;
        for k = 1:numel(cad.pnames)
            part = part2d.(cad.pnames{k});
            if isempty(part) == 0
                pos(count).part_label(k,:) = part_label(k,:) .* [sx sy] - origin_bbox;
            else
                pos(count).part_label(k,:) = [0 0];
            end
        end

        % dummy occlusion flag
        pos(count).occlusion = zeros(numel(cad.pnames),1);
        
        if count >= maxnum
            return;
        end
    end
end

% flip positive samples
function pos_flip = flip_positive(pos, cads)

N = numel(pos);
pos_flip = pos;

for i = 1:N
    % object label
    pos_flip(i).object_label = pos(i).object_label;
    
    % cad label
    cad_label = pos(i).cad_label;
    pos_flip(i).cad_label = cad_label;
   
    % view label
    cad = cads{cad_label};
    view_label = pos(i).view_label;
    
    azimuth = cad.parts2d(view_label).azimuth;
    % flip viewpoint
    azimuth = 360 - azimuth;
    if azimuth >= 360
        azimuth = 360 - azimuth;
    end   
    elevation = cad.parts2d(view_label).elevation;
    distance = cad.parts2d(view_label).distance;

    a = cad.azimuth;
    e = cad.elevation;
    d = cad.distance;

    [~, ind] = min(abs(azimuth - a));
    aind = ind - 1;
    [~, ind] = min(abs(elevation - e));
    eind = ind - 1;
    [~, ind] = min(abs(distance - d));
    dind = ind - 1;
    pos_flip(i).view_label = aind*numel(e)*numel(d) + eind*numel(d) + dind + 1;
    
    % width
    width = size(pos(i).image, 2);
    
    % part label
    part_num = numel(cad.pnames);
    pos_flip(i).part_label = zeros(part_num, 2);
    for j = 1:part_num
        if cad.roots(j) == 1 && (pos(i).part_label(j,1) ~= 0 || pos(i).part_label(j,2) ~= 0)
            % root location
            index = find_interval(azimuth, cad.view_num);
            root_index = find(cad.roots == 1);
            ind = root_index(index);            
            pos_flip(i).part_label(ind,1) = width - pos(i).part_label(j,1) + 1;
            pos_flip(i).part_label(ind,2) = pos(i).part_label(j,2);
            % part location
            ind_root = ind;
            xnum = cad.subpart_size(1);
            ynum = cad.subpart_size(2);
            for yindex = 1:ynum
                for xindex = 1:xnum
                    ind = (yindex - 1)*xnum + xindex;
                    ind_old = (yindex - 1)*xnum + xnum - xindex + 1;
                    pos_flip(i).part_label(ind_root+ind,1) = width - pos(i).part_label(j+ind_old,1) + 1;
                    pos_flip(i).part_label(ind_root+ind,2) = pos(i).part_label(j+ind_old,2);
                end
            end
            break;
        end
    end
    
    % bounding box
    x2 = pos(i).bbox(1) + pos(i).bbox(3);
    bbox = pos(i).bbox;
    pos_flip(i).bbox = [width-x2+1 bbox(2) bbox(3) bbox(4)];
    
    % image
    pos_flip(i).image = pos(i).image(:,end:-1:1,:);
end

% crop the bounding box
function B = crop_bbox(I, bbox, scale)

% enlarge the bbox by scale
width = scale * bbox(3);
height = scale * bbox(4);
x = bbox(1) + bbox(3)/2;
y = bbox(2) + bbox(4)/2;
bbox = [x - width/2, y - height/2, width, height];

x1 = max(1, round(bbox(1)));
x2 = min(size(I,2), round(bbox(1)+bbox(3)));
y1 = max(1, round(bbox(2)));
y2 = min(size(I,1), round(bbox(2)+bbox(4)));
B = I(y1:y2, x1:x2, :);


% randomly select negative training images from pascal data set
function neg = rand_negative(cls, maxnum)

pascal_init;
ids = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');
count = 0;
for i = 1:length(ids)
%     fprintf('%s: parsing negatives: %d/%d\n', cls, i, length(ids));    
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    if isempty(clsinds)
        if count >= maxnum
            break;
        end        
        filename = [VOCopts.datadir rec.imgname];
        I = imread(filename);
        count = count + 1;
        neg(count).image = I;
        neg(count).object_label = -1;
        neg(count).part_label = [];
        neg(count).cad_label = [];
        neg(count).view_label = [];
        neg(count).bbox = [];       
        neg(count).occlusion = [];
    end
end

% project the CAD model to generate aspect part locations
function bbox = generate_bbox(cad, part2d, part_label)

pnames = cad.pnames;

x1 = inf;
x2 = -inf;
y1 = inf;
y2 = -inf;
for i = 1:numel(pnames)
    part = part2d.(pnames{i});
    center = part_label(i,:);
    if isempty(part) == 0 && cad.roots(i) == 1
        part = part + repmat(center, size(part,1), 1);
        x1 = min([x1; part(:,1)]);
        x2 = max([x2; part(:,1)]);
        y1 = min([y1; part(:,2)]);
        y2 = max([y2; part(:,2)]);
    end
end

% build the bounding box
bbox = [x1 y1 x2-x1 y2-y1];
