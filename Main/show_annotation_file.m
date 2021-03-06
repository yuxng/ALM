% show image with annotation
function show_annotation_file(cls)

path_file = sprintf('data_aspect/%s_wrap.dat', cls);

% load CAD model
object = load(sprintf('../Geometry/Aspect/%s.mat', cls));
cad = object.(cls);

fid = fopen(path_file, 'r');
N = fscanf(fid, '%d', 1);

figure;
for i = 1:N
    disp(i);
    example = read_sample(fid, cad, 1);
    % read original image and annotation
    I_origin = uint8(example.image);
    imagesc(I_origin);
    axis equal;
    hold on;

    if example.object_label == 1
        bbox = example.bbox;
        rectangle('Position', [bbox(1) bbox(2) bbox(3)-bbox(1) bbox(4)-bbox(2)], 'EdgeColor', 'g', 'LineWidth', 2);
        view_label = example.view_label + 1;
        cad_label = example.cad_label + 1;
        part_label = example.part_label;
        part2d = cad.parts2d(view_label);
        til = sprintf('cad=%d, a=%.2f, e=%.2f, d=%.2f', cad_label, part2d.azimuth, part2d.elevation, part2d.distance);
        title(til);          
        for k = 1:numel(cad.pnames)
            if part_label(k,1) ~= 0 
                % annotated part center
                center = [part_label(k,1), part_label(k,2)];
                if isempty(part2d.(cad.pnames{k})) == 0
                    plot(center(1), center(2), 'ro');
                    text(center(1), center(2), num2str(k), 'BackgroundColor', 'r');
                    part = part2d.(cad.pnames{k}) + repmat(center, 5, 1);
                    patch('Faces', [1 2 3 4 5], 'Vertices', part, 'EdgeColor', 'r', 'FaceColor', 'r', 'FaceAlpha', 0.1);
                end
            end
        end
    end
    hold off;
    pause;
end
