function write_pascal_negatives

cls = 'car';
maxnum = inf;

pascal_init;
ids = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');
count = 0;
for i = 1:length(ids)
%     fprintf('%s: parsing negatives: %d/%d\n', cls, i, length(ids));    
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    if isempty(clsinds)
        if count >= maxnum
            break;
        end        
        filename = [VOCopts.datadir rec.imgname];
        I = imread(filename);
        if size(I,1) > 500 || size(I,2) > 500
            disp(ids{i});
        end
        count = count + 1;
        neg(count).image = I;
        neg(count).object_label = -1;
        neg(count).part_label = [];
        neg(count).cad_label = [];
        neg(count).view_label = [];
        neg(count).bbox = [];       
        neg(count).occlusion = [];
    end
end

pos = [];
fprintf('Writing data, %d negatives\n', numel(neg));
filename = sprintf('data/%s_pascal_neg.dat', cls);
write_data(filename, pos, neg);