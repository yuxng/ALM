% project the CAD model to generate aspect part locations
function bbox = generate_bbox(cad, part2d, part_label)

pnames = cad.pnames;

x1 = inf;
x2 = -inf;
y1 = inf;
y2 = -inf;
for i = 1:numel(pnames)
    if i <= numel(cad.parts)
        continue;
    end
    part = part2d.(pnames{i});
    center = part_label(i,:);
    if isempty(part) == 0
        part = part + repmat(center, size(part,1), 1);
        x1 = min([x1; part(:,1)]);
        x2 = max([x2; part(:,1)]);
        y1 = min([y1; part(:,2)]);
        y2 = max([y2; part(:,2)]);
    end
end

% build the bounding box
bbox = [x1 y1 x2-x1 y2-y1];