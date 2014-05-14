function display_result

cls = 'car';
cls_data = 'val';
dat_file = sprintf('data/%s.tst', cls);
pre_file = sprintf('data/%s.pre', cls);

% load cad model
cad_file = sprintf('../Geometry/Voxel/%s.mat', cls);
cad = load(cad_file);
cad = cad.(cls);
pnames = cad.pnames;
part_num = numel(pnames);

% open prediction file
fdat = fopen(dat_file, 'r');
fpre = fopen(pre_file, 'r');

N = fscanf(fdat, '%d', 1);
figure;
for i = 1:N
    if i ~= 1 && mod(i-1, 16) == 0
        pause;
    end
    ind = mod(i-1, 16)+1;
    
    % read ground truth
    example = read_sample(fdat, cad, 1);
    I = uint8(example.image);
    subplot(4, 4, ind);
    imshow(I);
    axis off;
    
    % read detections
    num = fscanf(fpre, '%d', 1);
    if num == 0
        fprintf('no detection for test image %d\n', i);
        continue;
    else
        examples = cell(num, 1);
        for j = 1:num
            examples{j} = read_sample(fpre, cad, 0);
        end
    end    
    
    hold on;

    til = sprintf('%d: ', i);
    for k = 1:min(4, num)
        % get predicted bounding box
        bbox_pr = examples{k}.bbox;
        if example.object_label == 1
            overlap = box_overlap(bbox_pr', example.bbox');
        else
            overlap = 0;
        end
        
        view_label = examples{k}.view_label + 1;
        part2d = cad.parts2d(view_label);

        til = sprintf('%sa=%.2f, e=%.2f, d=%.2f, score=%.2f, overlap=%.2f\n', ...
            til, part2d.azimuth, part2d.elevation, part2d.distance, examples{k}.energy, overlap);
            
        part_label = examples{k}.part_label;
        for a = 1:part_num
            if isempty(part2d.homographies{a}) == 0 && part_label(a,1) ~= 0 && part_label(a,2) ~= 0
                plot(part_label(a,1), part_label(a,2), 'ro');
                % render parts
                part = part2d.(pnames{a}) + repmat(part_label(a,:), 5, 1);
                patch('Faces', [1 2 3 4 5], 'Vertices', part, 'FaceColor', 'r', 'EdgeColor', 'r', 'FaceAlpha', 0.1, 'LineWidth', 2);           
            end
        end
        
        % draw bounding box
%         bbox_pr(1) = max(1, bbox_pr(1));
%         bbox_pr(2) = max(1, bbox_pr(2));
%         bbox_pr(3) = min(size(I,2), bbox_pr(3));
%         bbox_pr(4) = min(size(I,1), bbox_pr(4));
        bbox_draw = [bbox_pr(1), bbox_pr(2), bbox_pr(3)-bbox_pr(1), bbox_pr(4)-bbox_pr(2)];
        rectangle('Position', bbox_draw, 'EdgeColor', 'g', 'LineWidth', 2);
        text(bbox_draw(1), bbox_draw(2), num2str(k), 'BackgroundColor', 'r');
        line([bbox_pr(1) bbox_pr(1)], [bbox_pr(2) bbox_pr(4)], 'Color', 'g', 'LineWidth', 2);
    end
    
    title(til);
    subplot(4, 4, ind);
    hold off;
end

fclose(fdat);
fclose(fpre);