% render part from its frontal view
function parts2d_front = render_part_front(cad)

view_num = cad.view_num;
subpart_num = cad.subpart_num;
subpart_size = cad.subpart_size;
viewport = cad.viewport;
hog_size = cad.hog_size;

% initialization
parts = cad.parts;
N = numel(parts);
parts2d_front(N).width = 0;
parts2d_front(N).height = 0;
parts2d_front(N).distance = 0;

for i = 1:N
    % determine the frontal view of the part
    view_index = parts(i).view_index;
    e = mean(cad.elevation);    
    if cad.roots(i) == 1
        d = cad.distance_front_root;        
        x3d = cad.parts(i).x3d;
    else
        d = cad.distance_front_part;
        parent = floor((i-1)/(subpart_num+1))*(subpart_num+1) + 1;
        x3d = cad.parts(parent).x3d;
    end
    
    % render the 3D object from different azimuth and take the mean shape
    count = 0;
    w = [];
    h = [];
    c = zeros(1,2);
    for a = 0:15:345
        index = find_interval(a, view_num);
        if index ~= view_index
            continue;
        end
        x2d = project_part_points(x3d, a, e, d, viewport);

        x1 = min(x2d(:,1));
        x2 = max(x2d(:,1));
        y1 = min(x2d(:,2));
        y2 = max(x2d(:,2));        
        
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
    width = sqrt(area/aspect);
    height = width*aspect;
    center = c ./ count;    
    
    x1 = center(1) - width/2;
    x2 = center(1) + width/2;
    y1 = center(2) - height/2;
    y2 = center(2) + height/2;
    
    if cad.roots(i) == 1
        part = [x1 y1; x1 y2; x2 y2; x2 y1; x1 y1];
        c = [(x1+x2)/2 (y1+y2)/2];
    else
        index = mod(i-1, subpart_num+1);
        index_x = floor((index-1)/subpart_size(1));
        index_y = mod(index-1, subpart_size(1));
        width = x2 - x1;
        height = y2 - y1;
        x1_part = x1 + (width / subpart_size(1)) * index_y;
        x2_part = x1_part + width / subpart_size(1);
        y1_part = y1 + (height / subpart_size(2)) * index_x;
        y2_part = y1_part + height / subpart_size(2);
        part = [x1_part y1_part; x1_part y2_part; x2_part y2_part; x2_part y1_part; x1_part y1_part];
        c = [(x1_part+x2_part)/2 (y1_part+y2_part)/2];        
    end
    
    % assign the front part
    parts2d_front(i).vertices = part - repmat(c, size(part,1), 1);
    parts2d_front(i).center = c;
    width = round(max(part(:,1))-min(part(:,1)));
    if mod(width, hog_size) >= hog_size/2
        width = width + hog_size - mod(width, hog_size);
    else
        width = width - mod(width, hog_size);
    end
    parts2d_front(i).width = width;
    height = round(max(part(:,2))-min(part(:,2)));
    if mod(height, hog_size) >= hog_size/2
        height = height + hog_size - mod(height, hog_size);
    else
        height = height - mod(height, hog_size);
    end
    parts2d_front(i).height = height;
    parts2d_front(i).distance = d;
    parts2d_front(i).viewport = viewport;
    parts2d_front(i).pname = cad.pnames{i};
end  