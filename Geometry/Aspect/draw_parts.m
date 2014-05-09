function draw_parts(cad)

parts2d = cad.parts2d;
pnames = cad.pnames;
figure;
axis equal;
hold on;
M = numel(parts2d);
for i = 1:M
    disp(i);
    for j = numel(pnames):-1:1
        a = parts2d(i).azimuth;
        e = parts2d(i).elevation;
        d = parts2d(i).distance;
        part = parts2d(i).(pnames{j});
        center = parts2d(i).centers(j,:);
        if isempty(part) == 0
            part = part + repmat(center, size(part,1), 1);
            set(gca,'YDir','reverse');
            patch(part(:,1), part(:,2), 'r', 'FaceAlpha', 0.5);
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