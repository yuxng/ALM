% check the visibility of part
function flag = check_visibility_part(visibility, grid, threshold)

index = find(grid(:) == 1);
num_voxel_part = numel(index);
num_part_visible = sum(visibility(index));
ratio = num_part_visible /  num_voxel_part;

if ratio > threshold
    flag = 1;
else
    flag = 0;
end