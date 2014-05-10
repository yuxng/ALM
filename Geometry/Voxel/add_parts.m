% add parts to the cad model
function cad = add_parts(cad)

% initialize parameters
cad.distance_minpixel = 30;
cad.distance_maxpixel = 800;
cad.view_num = 8;
cad.hog_size = 6;
cad.viewport = 3000;
cad.occ_per = 0.8;

% determine the azimuth, elevation and distance
% TO DO: mean-shift clustering to get the azimuth, elevation and distance
[azimuth, elevation, distance] = discretize_viewpoint(cad);
cad.azimuth = azimuth;
cad.elevation = elevation;
cad.distance = distance;
cad.distance_front_root = find_distance_front(cad, 19);

% first try the whole object as a part
view_num = cad.view_num;
cad.pnames = cell(1, view_num);
for i = 1:view_num
    cad.pnames{i} = sprintf('view%d', i);
    cad.parts(i).grid = cad.grid;
    cad.parts(i).x3d = compute_part_points(cad.grid);
    cad.parts(i).view_index = i;
end
cad.roots = ones(1, view_num);

% render part from its frontal view
fprintf('Render 2d part front\n');
parts2d_front = render_part_front(cad);
cad.parts2d_front = parts2d_front;

% render parts into an image
fprintf('generate 2d part\n');
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

% find maximum distance
diff = abs(w - cad.distance_maxpixel);
[~, index] = min(diff);
dmax = d(index);

distance = exp(log(dmax):0.1:log(dmin));

% find the distance for frontal view
function distance_front = find_distance_front(cad, dis)

distance = cad.distance;
diff = abs(distance - dis);
[~, index] = min(diff);
distance_front = distance(index);