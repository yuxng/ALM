% Draw aspect parts on 2d images
function draw_aspect_parts
 
cls = 'car';
% subtype = {'hatchback', 'minivan', 'sedan', 'SUV', 'truck', 'wagon'};
subtype = [];

% load cad model
if isempty(subtype) == 0
    n = numel(subtype);
    cads = cell(n, 1);
    for i = 1:n
        object = load([subtype{i} '.mat']);
        cads{i} = object.cad;
    end
else
    cads = cell(1, 1);
    object = load('car.mat');
    cads{1} = object.cad;
end

% path of annotation files
anno_path = '../Annotations/car_pascal/';
img_path = '../Images/car_pascal/';
files = dir([anno_path '*.mat']);

figure(1);
% count = 0;
for i = 1:numel(files)

    % load annotation files
    object = load([anno_path files(i).name]);
    record = object.record;
    
    filename = [img_path record.filename];
    I = imread(filename);

    for j = 1:numel(record.objects)
        object = record.objects(j);
        if strcmp(cls, object.class) == 0
            continue;
        end
        if isempty(subtype) == 0
            index = strcmp(object.subtype, subtype);
            index_type = find(index == 1);
            if isempty(index_type) == 1
                continue;
            end
        else
            index_type = 1;
        end
        if object.viewpoint.distance == 0
            continue;
        end
        
%         count = count + 1;
%         if count ~= 1 && mod(count-1, 16) == 0
%             pause;
%         end
%         ind = mod(count-1,16)+1;
%         subplot(4, 4, ind);        
        
        disp(files(i).name);
        cad = cads{index_type};
        % part2d = project_cad(cad, object.viewpoint.azimuth, object.viewpoint.elevation, object.viewpoint.distance);
        view_label = find_closest_view(cad, object);
        part2d = cad.parts2d(view_label);
        
        imshow(I);
        hold on;
        
        % load aspect layout model
        cad_index = object.cad_index;
        filename = sprintf('../../Pose_Dataset/ALM/car_%02d.mat', cad_index);
        alm = load(filename);
        alm = alm.cad;
        rescale = 0.8;
        part_label = zeros(numel(alm.pnames), 2);
        for k = 1:numel(alm.parts),
            part_label(k,:) = project_3d(rescale * alm.parts(k).center, object);
        end
        index = find_interval(object.viewpoint.azimuth, alm.view_num) + numel(alm.parts);
        if is_occld_trunc(object)
            part_label(index,:) = part2d.centers(index,:) - [cad.viewport/2 cad.viewport/2] + [object.viewpoint.px object.viewpoint.py];
        else
            bbox = object.bbox;
            part_label(index,:) = [(bbox(1) + bbox(3))/2 (bbox(2) + bbox(4))/2];
        end
        
        pnames = cad.pnames;
        color = hsv(numel(pnames));        
        for k = numel(pnames):-1:1
            part = part2d.(pnames{k});
            center = part_label(k,:);
            if isempty(part) == 0
                part = part + repmat(center, size(part,1), 1);
                if k > numel(cad.parts)
                    patch(part(:,1), part(:,2), 'g', 'FaceColor', 'none', 'EdgeColor', 'g', 'LineWidth', 2);
                else
                    patch(part(:,1), part(:,2), color(k,:), 'FaceColor', color(k,:), 'EdgeColor', color(k,:), 'FaceAlpha', 0.5);
                end
                plot(center(1), center(2), 'o');
            end
        end
        str = sprintf('a = %.2f, e = %.2f, d = %.2f, error = %.2f, theta = %.2f', ...
            part2d.azimuth, part2d.elevation, ...
            part2d.distance, object.viewpoint.error, object.viewpoint.theta);
        title(str);
        
        hold off;
        pause;        

%         for k = 1:numel(alm.parts),
%             color = hsv(numel(alm.parts));
%             x = project_3d(alm.parts(k).vertices, object);
%             patch(x(:,1), x(:,2), color(k,:), 'FaceAlpha', 0.5);
%         end
    end
end