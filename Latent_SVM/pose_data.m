% Get training data from the PASCAL dataset.
function [pos, neg] = pose_data(cls, flippedpos, trainset)

globals;
pascal_init;

try
  load([cachedir cls '_' trainset '_pose']);
catch
  % positive examples from train+val
  fprintf('Read positive samples\n');
  pos = read_positive(cls, PASCAL3Dopts, VOCopts, flippedpos, trainset);

  % negative examples from train (this seems enough!)
  fid = fopen(sprintf(VOCopts.imgsetpath, 'train'), 'r');
  C = textscan(fid, '%s');
  ids = C{1};
  fclose(fid);
  
  neg = [];
  numneg = 0;
  for i = 1:length(ids);
    fprintf('%s: parsing negatives: %d/%d\n', cls, i, length(ids));
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    if isempty(clsinds)
      numneg = numneg+1;
      neg(numneg).im = [VOCopts.datadir rec.imgname];
      neg(numneg).flip = false;
    end
  end
  
  save([cachedir cls '_' trainset '_pose'], 'pos', 'neg');
end


% read positive training images
function pos = read_positive(cls, PASCAL3Dopts, VOCopts, flippedpos, trainset)

fid = fopen(sprintf(VOCopts.clsimgsetpath, cls, trainset), 'r');
C = textscan(fid, '%s%d');
fclose(fid);

ids = C{1}(C{2} == 1);
N = length(ids);
path_image = sprintf(PASCAL3Dopts.path_img_pascal, cls);
path_anno = sprintf(PASCAL3Dopts.path_ann_pascal, cls);

count = 0;
for i = 1:N
    fprintf('%s: parsing positives: %d/%d\n', cls, i, N);    
    file_ann = sprintf('%s/%s.mat', path_anno, ids{i});
    % load annotation
    image = load(file_ann);
    record = image.record;
    objects = record.objects;
    for j = 1:length(objects)
        if isfield(objects(j), 'viewpoint') == 0 || ~strcmp(cls, objects(j).class) || objects(j).difficult == 1
            continue;
        end
        viewpoint = objects(j).viewpoint;
        if isempty(viewpoint)
            continue;
        end
        bbox = objects(j).bbox;
        file_img = sprintf('%s/%s.jpg', path_image, ids{i});
        count = count + 1;
        pos(count).im = file_img;
        pos(count).x1 = bbox(1);
        pos(count).y1 = bbox(2);
        pos(count).x2 = bbox(3);
        pos(count).y2 = bbox(4);
        pos(count).flip = false;
        pos(count).trunc = objects(j).truncated;
        if viewpoint.distance == 0
            azimuth = viewpoint.azimuth_coarse;
        else
            azimuth = viewpoint.azimuth;
        end
        pos(count).azimuth = azimuth;
        % flip the positive example
        if flippedpos
            oldx1 = bbox(1);
            oldx2 = bbox(3);
            bbox(1) = record.imgsize(1) - oldx2 + 1;
            bbox(3) = record.imgsize(1) - oldx1 + 1;
            count = count + 1;
            pos(count).im = file_img;
            pos(count).x1 = bbox(1);
            pos(count).y1 = bbox(2);
            pos(count).x2 = bbox(3);
            pos(count).y2 = bbox(4);
            pos(count).flip = true;
            pos(count).trunc = objects(j).truncated;
            % flip viewpoint
            azimuth = 360 - azimuth;
            if azimuth >= 360
                azimuth = 360 - azimuth;
            end
            pos(count).azimuth = azimuth;            
        end
    end
end