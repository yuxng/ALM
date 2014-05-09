% add root parts to the CAD model
function cad_new = add_root_parts(cad)

distance = cad.distance_front_root;
vnum = cad.view_num;

cad_new = cad;
pnames = cad.pnames;
part_num = numel(pnames);

width = zeros(vnum,1);
height = zeros(vnum,1);
center = zeros(vnum,2);

for k = 1:vnum
    count = 0;
    w = [];
    h = [];
    c = zeros(1,2);
    for i = 1:numel(cad.parts2d)
        part2d = cad.parts2d(i);
        index = find_interval(part2d.azimuth, vnum);
        if part2d.distance ~= distance || index ~= k
            continue;
        end
        x1 = inf;
        x2 = -inf;
        y1 = inf;
        y2 = -inf;
        for j = 1:part_num
            if isempty(part2d.(pnames{j})) == 0
                part = part2d.(pnames{j}) + repmat(part2d.centers(j,:), 5, 1);
                x1 = min([x1; part(:,1)]);
                x2 = max([x2; part(:,1)]);
                y1 = min([y1; part(:,2)]);
                y2 = max([y2; part(:,2)]);
            end
        end
        count = count + 1;
        w(count) = x2 - x1;
        h(count) = y2 - y1;
        c = c + [(x1+x2)/2 (y1+y2)/2];
    end
        
    aspects = h./w;
    aspect = mean(aspects);

    areas = h.*w;
    area = 0.8*mean(areas);

    % pick dimensions
    width(k) = sqrt(area/aspect);
    height(k) = width(k)*aspect;
    center(k,:) = c ./ count;
end

% add frontal root parts
w = width;
h = height;
count = 0;
for i = 1:vnum
    if isnan(w(i)) == 1
        continue;
    end
    count = count + 1;
    x1 = center(i,1) - w(i)/2;
    x2 = center(i,1) + w(i)/2;
    y1 = center(i,2) - h(i)/2;
    y2 = center(i,2) + h(i)/2;
    part = [x1 y1;x1 y2; x2 y2; x2 y1; x1 y1];
    c = center(i,:);
    % assign the front part
    cad_new.parts2d_front(part_num+count).vertices = part - repmat(c, size(part,1), 1);
    cad_new.parts2d_front(part_num+count).center = c;
    width = round(max(part(:,1))-min(part(:,1)));
    if mod(width, 6) >= 3
        width = width + 6 - mod(width, 6);
    else
        width = width - mod(width, 6);
    end
    cad_new.parts2d_front(part_num+count).width = width;
    height = round(max(part(:,2))-min(part(:,2)));
    if mod(height, 6) >= 3
        height = height + 6 - mod(height, 6);
    else
        height = height - mod(height, 6);
    end
    cad_new.parts2d_front(part_num+count).height = height;
    cad_new.parts2d_front(part_num+count).distance = distance;
    cad_new.parts2d_front(part_num+count).viewport = cad.parts2d_front(1).viewport;
    cad_new.parts2d_front(part_num+count).pname = sprintf('view%d', i);
    cad_new.pnames{part_num+count} = sprintf('view%d', i);
    cad_new.symmetry(part_num+count) = -1;
end