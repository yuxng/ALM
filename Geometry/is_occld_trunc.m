% check if an object is occluded or truncated
function flag = is_occld_trunc(object)

anchors = object.anchors;
names = fieldnames(anchors);

flag = 0;
for i = 1:numel(names)
    if anchors.(names{i}).status == 3 || anchors.(names{i}).status == 4 || ...
            anchors.(names{i}).status == 5
        flag = 1;
        break;
    end
end