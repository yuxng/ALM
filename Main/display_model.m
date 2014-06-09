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
% part_num = numel(pnames);
length = 0;
view_num = cad.view_num;
for i = 1:view_num
    index = (i-1)*(1+cad.subpart_num) + 1;
    b0 = cad.parts2d_front(index).width / 6;
    b1 = cad.parts2d_front(index).height / 6;
    length = length + b0 * b1 *32;
    fprintf('part %s, b0 = %d, b1 = %d, length = %d, total = %d\n', pnames{index}, b0, b1, b0*b1*32, length);
    w = model.weights(count:count-1+b0*b1*32);
    w = reshape(w, b1, b0, 32);
    im = visualizeHOG(w, 0);
    subplot(view_num/2, 4, 2*i-1);
    imagesc(im); 
%     h = title(pnames{index});
%     set(h, 'FontSize', 16);
    colormap gray;
    axis equal;
    axis off;
    count = count + b0*b1*32+1;
    
    images = cell(cad.subpart_size(2), cad.subpart_size(1));
    for j = 1:cad.subpart_num
        b0 = cad.parts2d_front(index+j).width / 6;
        b1 = cad.parts2d_front(index+j).height / 6;
        length = length + b0 * b1 *32;
        fprintf('part %s, b0 = %d, b1 = %d, length = %d, total = %d\n', pnames{index+j}, b0, b1, b0*b1*32, length);
        w = model.weights(count:count-1+b0*b1*32);
        w = reshape(w, b1, b0, 32);
        im = visualizeHOG(w, 0);
        index_x = floor((j-1)/cad.subpart_size(1)) + 1;
        index_y = mod(j-1, cad.subpart_size(1)) + 1;
        images{index_x, index_y} = im;
        count = count + b0*b1*32+1;
    end
    
    % construct image
    A = cell(cad.subpart_size(2), 1);
    for j = 1:cad.subpart_size(2)
        A{j} = [];
        for k = 1:cad.subpart_size(1)
            A{j} = horzcat(A{j}, images{j,k});
        end
    end
    B = [];
    for j = 1:cad.subpart_size(2)
        B = vertcat(B, A{j});
    end
    
    subplot(view_num/2, 4, 2*i);
    imagesc(B); 
%     h = title([pnames{index} 'part']);
%     set(h, 'FontSize', 16);
    colormap gray;
    axis equal;
    axis off;    
end
model.pairwise = reshape(model.weights(count:end), numel(pnames), numel(pnames)-1);
