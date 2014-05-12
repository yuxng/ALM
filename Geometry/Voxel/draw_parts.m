function draw_parts(cad)

parts2d = cad.parts2d;
pnames = cad.pnames;
figure;
axis equal;
hold on;
M = numel(parts2d);
for i = 1:M
    disp(i);
    for j = 1:numel(pnames)
        a = parts2d(i).azimuth;
        e = parts2d(i).elevation;
        d = parts2d(i).distance;
        part = parts2d(i).(pnames{j});
        center = parts2d(i).centers(j,:);
        if isempty(part) == 0
            part = part + repmat(center, size(part,1), 1);
            set(gca,'YDir','reverse');
            if cad.roots(j) == 1
                patch(part(:,1), part(:,2), 'b', 'FaceAlpha', 0.3);
            else
                patch(part(:,1), part(:,2), 'r', 'FaceAlpha', 0.3);
            end
            til = sprintf('azimuth=%.2f, elevation=%.2f, distance=%.2f', a, e, d);
            title(til);
            plot(center(1), center(2), 'o');
        end
    end
    pause;
    clf;
    axis equal;
    hold on;
end