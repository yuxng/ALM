% find the closest discrete viewpoints
function view_label = find_closest_view(cad, object)

if object.viewpoint.distance == 0
    view_label = -1;
    return;
end

azimuth = object.viewpoint.azimuth;
elevation = object.viewpoint.elevation;
distance = object.viewpoint.distance;

a = cad.azimuth;
e = cad.elevation;
d = cad.distance;

[~, ind] = min(abs(azimuth - [a 360]));
if ind == numel(a) + 1
    ind = 1;
end
aind = ind - 1;
[~, ind] = min(abs(elevation - e));
eind = ind - 1;
[~, ind] = min(abs(distance - d));
dind = ind - 1;
view_label = aind*numel(e)*numel(d) + eind*numel(d) + dind + 1;
