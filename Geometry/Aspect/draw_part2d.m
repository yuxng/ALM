function draw_part2d(cad, part2d)

pnames = cad.pnames;
figure;
axis equal;
hold on;
for j = numel(pnames):-1:1
    a = part2d.azimuth;
    e = part2d.elevation;
    d = part2d.distance;
    part = part2d.(pnames{j});
    center = part2d.centers(j,:);
    if isempty(part) == 0
        part = part + repmat(center, size(part,1), 1);
        set(gca,'YDir','reverse');
        patch(part(:,1), part(:,2), 'r', 'FaceAlpha', 0.5);
        til = sprintf('azimuth=%d, elevation=%d, distance=%d', a, e, d);
        title(til);
        plot(center(1), center(2), 'o');
    end
end
hold off;