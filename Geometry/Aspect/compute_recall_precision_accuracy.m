% compute recall and viewpoint accuracy
function [recall, precision, accuracy, ap, aa] = compute_recall_precision_accuracy(cls, vnum, azimuth_interval)

if nargin < 3
    azimuth_interval = [0 (360/(vnum*2)):(360/vnum):360-(360/(vnum*2))];
end

% viewpoint annotation path
opt = globals;
path_ann_view = sprintf(opt.path_ann_pascal, cls);

% read ids of validation images
pascal_init;
ids = textread(sprintf(VOCopts.imgsetpath, 'val'), '%s');
M = numel(ids);


% load cad model
cls_data = [cls ''];
cad_file = sprintf('%s.mat', cls_data);
cad = load(cad_file);
cad = cad.(cls_data);

% read detections
pre_file = 'data/val_latent_0.pre';
fid = fopen(pre_file, 'r');
examples = cell(M,1);
for i = 1:M
    num = fscanf(fid, '%d', 1);
    example = [];
    for j = 1:num
        example(j).object_label = fscanf(fid, '%d', 1);
        example(j).cad_label = fscanf(fid, '%d', 1) + 1;
        example(j).view_label = fscanf(fid, '%d', 1) + 1;
        example(j).energy = fscanf(fid, '%f', 1);
        part_num = numel(cad.pnames);
        example(j).part_label = fscanf(fid, '%f', part_num*2);
        example(j).part_label = reshape(example(j).part_label, part_num, 2);
        example(j).bbox = fscanf(fid, '%f', 4)';
        example(j).class = cls;
    end
    examples{i} = example;
end
fprintf('Read detections done\n');


energy = [];
correct = [];
correct_view = [];
overlap = [];
count = zeros(M,1);
num = zeros(M,1);
num_pr = 0;
for i = 1:M
    fprintf('%s view %d: %d/%d\n', cls, vnum, i, M);   
    
    % read ground truth bounding box
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    diff = [rec.objects(clsinds).difficult];
    clsinds(diff == 1) = [];
    n = numel(clsinds);
    bbox = zeros(n, 4);
    for j = 1:n
        bbox(j,:) = rec.objects(clsinds(j)).bbox;
    end
    count(i) = size(bbox, 1);
    det = zeros(count(i), 1);
    
    % read ground truth viewpoint
    if isempty(clsinds) == 0
        filename = fullfile(path_ann_view, sprintf('/%s.mat', ids{i}));
        object = load(filename);
        record = object.record;
        view_gt = zeros(n, 1);
        for j = 1:n
            if record.objects(clsinds(j)).viewpoint.distance == 0
                azimuth = record.objects(clsinds(j)).viewpoint.azimuth_coarse;
            else
                azimuth = record.objects(clsinds(j)).viewpoint.azimuth;
            end
            view_gt(j) = find_interval_internal(azimuth, azimuth_interval);
        end
    else
        view_gt = [];
    end
    
    % get predicted bounding box
    example = examples{i};
    num(i) = numel(example);
    % for each predicted bounding box
    for j = 1:num(i)
        num_pr = num_pr + 1;
        energy(num_pr) = example(j).energy;
        bbox_pr = example(j).bbox;
        
        bbox_pr(1) = max(1, bbox_pr(1));
        bbox_pr(2) = max(1, bbox_pr(2));
        bbox_pr(3) = min(bbox_pr(3), rec.imgsize(1));
        bbox_pr(4) = min(bbox_pr(4), rec.imgsize(2));         
        
        cad_label = example(j).cad_label;
        view_label = example(j).view_label;
        azimuth = cad(cad_label).parts2d(view_label).azimuth;
        view_pr = find_interval_internal(azimuth, azimuth_interval);
        
        % compute box overlap
        if isempty(bbox) == 0
            o = box_overlap(bbox, bbox_pr);
            [maxo, index] = max(o);
            if maxo >= 0.5 && det(index) == 0
                overlap{num_pr} = index;
                correct(num_pr) = 1;
                det(index) = 1;
                % check viewpoint
                if view_pr == view_gt(index)
                    correct_view(num_pr) = 1;
                else
                    correct_view(num_pr) = 0;
                end
            else
                overlap{num_pr} = [];
                correct(num_pr) = 0;
                correct_view(num_pr) = 0;
            end
        else
            overlap{num_pr} = [];
            correct(num_pr) = 0;
            correct_view(num_pr) = 0;
        end
    end
end
overlap = overlap';


[threshold, index] = sort(energy, 'descend');
correct = correct(index);
correct_view = correct_view(index);
n = numel(threshold);
recall = zeros(n,1);
precision = zeros(n,1);
accuracy = zeros(n,1);
num_correct = 0;
num_correct_view = 0;
for i = 1:n
    % compute precision
    num_positive = i;
    num_correct = num_correct + correct(i);
    if num_positive ~= 0
        precision(i) = num_correct / num_positive;
    else
        precision(i) = 0;
    end
    
    % compute accuracy
    num_correct_view = num_correct_view + correct_view(i);
    if num_correct ~= 0
        accuracy(i) = num_correct_view / num_positive;
    else
        accuracy(i) = 0;
    end
    
    % compute recall
    recall(i) = num_correct / sum(count);
end


ap = VOCap(recall, precision);
fprintf('AP = %.4f\n', ap);

aa = VOCap(recall, accuracy);
fprintf('AVP = %.4f\n', aa);

% draw recall-precision and accuracy curve
figure;
hold on;
plot(recall, precision, 'r', 'LineWidth',3);
plot(recall, accuracy, 'g', 'LineWidth',3);
xlabel('Recall');
ylabel('Precision/Accuracy');
tit = sprintf('Average Precision = %.1f / Average Accuracy = %.1f', 100*ap, 100*aa);
title(tit);
hold off;


function ind = find_interval_internal(azimuth, a)

for i = 1:numel(a)
    if azimuth < a(i)
        break;
    end
end
ind = i - 1;
if azimuth > a(end)
    ind = 1;
end
