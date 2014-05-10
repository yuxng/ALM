% train a 3D model from a set of CAD models
function cad = cad_train_subtype(subtype)

root_path = pwd;
cd('../../3rd_Party/PROPACK');
addpath(pwd);
cd(root_path);

cd('../..');
addpath(pwd);
cd(root_path);

cad.cls = 'car';
cad.pnames = {'head', 'left', 'right', 'front', 'back', 'tail', 'roof'};
cad.distance_minpixel = 30;
cad.distance_maxpixel = 800;
cad.view_num = 8;
[azimuth, elevation, distance] = discretize_viewpoint(cad);
cad.azimuth = azimuth;
cad.elevation = elevation;
cad.distance = distance;
cad.distance_front_root = find_distance_front(cad, 19);
cad.distance_front_part = find_distance_front(cad, 13);
cad.occ_per = [0.6 0.6 0.6 0.9 0.9 0.6 0.6];
cad.viewport = 3000;
cad.tilt_threshold = [70 80 80 80 80 70 70];
cad.hog_size = 6;
cad.shinkage_root = 0.85;
cad.rescale = 1 / cad.shinkage_root;
cad.symmetry = [1 3 2 4 5 6 7];

shinkage_width = [0.8 0.8 0.8 0.8 0.8 0.8 0.8];
shinkage_height = [0.8 0.8 0.8 0.8 0.8 0.8 0.8];
part_direction = [0 -1 0; -1 0 0; 1 0 0; 0 0 1; 0 0 1; 0 1 0; 0 0 1];
switch subtype
    case {'hatchback', 'minivan', 'SUV'}
        M = 1:4;
    case 'truck'
        M = 1:4;
        cad.tilt_threshold(5) = -1;
    case 'sedan'
        M = [1:3 5];
    case 'wagon'
        M = 2:5;
    case 'all'
        M = [];
    otherwise
        return;
end

if isempty(M) == 0
    off_files = cell(numel(M),1);
    for i = 1:numel(M)
        off_files{i} = sprintf('%s_%02d', subtype, M(i));
    end    
else
    types = {'hatchback', 'minivan', 'sedan', 'SUV', 'wagon'};
    index = {1:4, 1:4, [1:3 5], 1:4, 2:5};
    count = 0;
    for i = 1:numel(types)
        count = count + numel(index{i});
    end
    off_files = cell(count, 1);
    count = 0;
    for i = 1:numel(types)
        for j = index{i}
            count = count + 1;
            off_files{count} = sprintf('%s_%02d', types{i}, j);
        end
    end
end

cls = cad.cls;
pnames = cad.pnames;
N = numel(pnames);
for i = 1:N
    part_vertices = [];
    for j = 1:numel(off_files)
        filename = sprintf('%s_subtype/%s_%s.off', cls, off_files{j}, pnames{i});
        disp(filename);
        vertices = load_off_file(filename);
        vertices = vertices * cad.rescale;
        part_vertices = [part_vertices; vertices];
    end

    [F, P, center, xaxis, yaxis] = fit_plane(part_vertices, shinkage_width(i), shinkage_height(i), part_direction(i,:));
    cad.parts(i).vertices = F;
    cad.parts(i).plane = P;
    cad.parts(i).center = center;
    cad.parts(i).xaxis = xaxis;
    cad.parts(i).yaxis = yaxis;
end

% render part from its frontal view
parts2d_front = render_part_front(cad);
cad.parts2d_front = parts2d_front;

% render parts to build the roots
parts2d = generate_2d_parts(cad);
cad.parts2d = parts2d;

% add root parts
cad = add_root_parts(cad);

% render parts with roots
parts2d = generate_2d_parts(cad);
cad.parts2d = parts2d;


% discretize the viewpoint
function [azimuth, elevation, distance] = discretize_viewpoint(cad)

azimuth = 0:15:345;

[~, e_imagenet, d_imagenet, w_imagenet] = viewpoint_distribution(cad.cls, 1);
[~, e_pascal, d_pascal, w_pascal] = viewpoint_distribution(cad.cls, 0);
e = [e_imagenet e_pascal];
d = [d_imagenet d_pascal];
w = [w_imagenet w_pascal];

elevation = mean(e);

% find minimum distance
diff = abs(w - cad.distance_minpixel);
[~, index] = min(diff);
dmin = d(index);

% find minimum distance
diff = abs(w - cad.distance_maxpixel);
[~, index] = min(diff);
dmax = d(index);

distance = exp(log(dmax):0.1:log(dmin));
% distance(end+1) = dmin;


% find the distance for frontal view
function distance_front = find_distance_front(cad, dis)

distance = cad.distance;
diff = abs(distance - dis);
[~, index] = min(diff);
distance_front = distance(index);