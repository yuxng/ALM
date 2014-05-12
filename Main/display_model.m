function model = display_model(filename, cls)

% load cad model
cad_file = sprintf('../Geometry/Voxel/%s.mat', cls);
cad = load(cad_file);
cad = cad.(cls);

% read data
fid = fopen(filename, 'r');

model.C = fscanf(fid, '%f', 1);
model.loss_function = fscanf(fid, '%d', 1);
model.loss_value = fscanf(fid, '%f', 1);
model.wxy = fscanf(fid, '%f', 1);
model.deep = fscanf(fid, '%d', 1);
model.padx = fscanf(fid, '%d', 1);
model.pady = fscanf(fid, '%d', 1);
% model.cad_index = fscanf(fid, '%d', 1);
% model.part_index = fscanf(fid, '%d', 1);
model.psi_size = fscanf(fid, '%d', 1);
model.weights = fscanf(fid, '%f', model.psi_size);

fclose(fid);

% draw HOG templates
figure;
count = 1;
pnames = cad.pnames;
part_num = numel(pnames);
length = 0;
for i = 1:part_num
    b0 = cad.parts2d_front(i).width / 6;
    b1 = cad.parts2d_front(i).height / 6;
    length = length + b0 * b1 *32;
    fprintf('part %s, b0 = %d, b1 = %d, length = %d, total = %d\n', pnames{i}, b0, b1, b0*b1*32, length);
    w = model.weights(count:count-1+b0*b1*32);
    w = reshape(w, b1, b0, 32);
    im = visualizeHOG(w, 0);
    subplot(8, 8, i);
    imagesc(im); 
    h = title(pnames{i});
    set(h, 'FontSize', 16);
    colormap gray;
    axis equal;
    axis off;
    count = count + b0*b1*32+1;
end