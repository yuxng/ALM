% render part from its frontal view
function parts2d_front = render_part_front(cad)

view_num = cad.view_num;
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
    if cad.roots(i) == 1
        view_index = parts(i).view_index;
        a = (360/view_num) * (view_index - 1);
        e = mean(cad.elevation);
        d = mean(cad.distance);        
    else
        % TO DO
    end
    x3d = cad.parts(i).x3d;
    x2d = project_part_points(x3d, a, e, d, viewport);
    
    % build the part shape
    x1 = min(x2d(:,1));
    x2 = max(x2d(:,1));
    y1 = min(x2d(:,2));
    y2 = max(x2d(:,2));
    part = [x1 y1; x1 y2; x2 y2; x2 y1; x1 y1];
    c = [(x1+x2)/2 (y1+y2)/2];
    
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