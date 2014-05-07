function write_pascal_images

pascal_init;
ids = textread(sprintf(VOCopts.imgsetpath, 'val'), '%s');
count = 0;
for i = 1:length(ids)
    fprintf('parsing negatives: %d/%d\n', i, length(ids));    
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    filename = [VOCopts.datadir rec.imgname];
    I = imread(filename);
    count = count + 1;
    neg(count).image = I;
    neg(count).object_label = -1;
    neg(count).part_label = [];
    neg(count).cad_label = [];
    neg(count).view_label = [];
    neg(count).bbox = [];       
    neg(count).occlusion = [];
end

pos = [];

fprintf('Writing data\n');
filename = 'data/val.tst';
write_data(filename, pos, neg);