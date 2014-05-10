% build the mean voxel model
function cad = cad_train(cls)

root_path = pwd;
cd('../..');
addpath(pwd);
cd(root_path);

opt = globals();
grid_size = 25;

% load CAD models
filename = sprintf(opt.path_cad, cls);
object = load(filename);
cads = object.(cls);
fprintf('Load CAD models done\n');

% compute the mean cad model
N = numel(cads);
grid = [];
for i = 1:N
    vertices = cads(i).vertices;
    voxel = simple_voxelization(vertices, grid_size);
    if isempty(grid) == 1
        grid = voxel;
    else
        grid = grid + voxel;
    end
end

cad.cls = cls;
cad.grid_size = grid_size;
cad.grid = double(grid > 2);
cad.x3d = compute_part_points(cad.grid);
fprintf('Build the mean voxel model done\n');

cad = add_parts(cad);