% project the cad model according to the viewpoint
function part2d = project_cad(cad, a, e, d)

part2d.azimuth = a;
part2d.elevation = e;
part2d.distance = d;
part2d.viewport = cad.viewport;
part2d.root = 0;
part2d.graph = zeros(numel(cad.pnames));
for i = 1:numel(cad.pnames)
    part2d.(cad.pnames{i}) = [];
end

% viewport size
R = cad.viewport * [1 0 0.5; 0 -1 0.5; 0 0 1];
R(3,3) = 1;

% render CAD model
[parts, occluded, parts_unoccluded] = render(cad.cls, cad, a, e, d);

% part number
N = numel(parts);
part2d.centers = zeros(N, 2);
part2d.homographies = cell(N, 1);
pnames = cad.pnames;
for i = 1:N
    % occluded percentage
    if occluded(i) > cad.occ_per(i)
        continue;
    end
    
    % compute the angle between the norm of the plane and the camera
    u = cad.parts(i).plane(1:3);
    [~, v] = projection(a, e, d);
    phi = acosd(dot(u,v) / (norm(u) * norm(v)));
    if phi >= 90
        continue;
    end

    % map to viewport
    p = R*[parts_unoccluded(i).x parts_unoccluded(i).y ones(numel(parts_unoccluded(i).x), 1)]';
    p = p(1:2,:)';
    c = R*[parts_unoccluded(i).center, 1]';
    c = c(1:2)';

    % translate the part center to the orignal
    part2d.(pnames{i}) = p - repmat(c, size(p,1), 1);
    part2d.centers(i,:) = c';

    % compute the homography for transfering current view of the part
    % to frontal view using four point correspondences
    % coefficient matrix
    A = zeros(8,9);
    % construct the coefficient matrix
    X = part2d.(pnames{i});
    xprim = cad.parts2d_front(i).vertices;
    for j = 1:4
        x = [X(j,:), 1];
        A(2*j-1,:) = [zeros(1,3), -x, xprim(j,2)*x];
        A(2*j, :) = [x, zeros(1,3), -xprim(j,1)*x];
    end
    [~, ~, V] = svd(A);
    % homography
    h = V(:,end);
    H = reshape(h, 3, 3)';
    % normalization
    H = H ./ H(3,3);
    part2d.homographies{i} = H;
end

% add root part
if numel(cad.pnames) == numel(cad.parts) + cad.view_num
    part_num = numel(cad.parts);
    x1 = inf;
    x2 = -inf;
    y1 = inf;
    y2 = -inf;
    for j = 1:part_num
        if isempty(part2d.(pnames{j})) == 0
            part = part2d.(pnames{j}) + repmat(part2d.centers(j,:), 5, 1);
            x1 = min([x1; part(:,1)]);
            x2 = max([x2; part(:,1)]);
            y1 = min([y1; part(:,2)]);
            y2 = max([y2; part(:,2)]);
        end
    end
    center = [(x1+x2)/2 (y1+y2)/2];
    width = (x2 - x1) * cad.shinkage_root;
    height = (y2 - y1) * cad.shinkage_root;
    x1 = center(1) - width/2;
    x2 = center(1) + width/2;
    y1 = center(2) - height/2;
    y2 = center(2) + height/2;
    part = [x1 y1;x1 y2; x2 y2; x2 y1; x1 y1];
    
    % assign the root part
    index = find_interval(part2d.azimuth, cad.view_num);
    view_name = sprintf('view%d', index);
    for j = 1:cad.view_num
        part_name = pnames{part_num+j};
        if strcmp(part_name, view_name) == 1
            part2d.(part_name) = part - repmat(center, size(part,1), 1);
            part2d.centers(part_num+j,:) = center;
            % compute the homography for transfering current view of the part
            % to frontal view using four point correspondences
            % coefficient matrix
            A = zeros(8,9);
            % construct the coefficient matrix
            X = part2d.(part_name);
            xprim = cad.parts2d_front(part_num+j).vertices;
            for k = 1:4
                x = [X(k,:), 1];
                A(2*k-1,:) = [zeros(1,3), -x, xprim(k,2)*x];
                A(2*k, :) = [x, zeros(1,3), -xprim(k,1)*x];
            end
            [~, ~, V] = svd(A);
            % homography
            h = V(:,end);
            H = reshape(h, 3, 3)';
            % normalization
            H = H ./ H(3,3);
            part2d.homographies{part_num+j} = H;
            part2d.root = part_num+j;
        else
            part2d.(part_name) = [];
            part2d.centers(part_num+j,:) = [0 0];
            part2d.homographies{part_num+j} = [];
        end
    end

    % remove large tilt part
    for i = 1:N
        H = part2d.homographies{i};
        if isempty(H) == 1
            continue;
        end

        % remove large tilt part
        A = H(1:2,1:2);
        [~, S] = svd(A);
        tilt = S(1,1)/S(2,2);
        theta = acosd(1/tilt);
        if theta > cad.tilt_threshold(i)
            part2d.(pnames{i}) = [];
            part2d.centers(i,:) = [0 0];
            part2d.homographies{i} = [];
        end
    end

    % construct graph, each row stores the parents of the node
    part2d.graph = zeros(numel(cad.pnames));
    root = part2d.root;
    for j = 1:numel(cad.pnames)
        if j ~= root && isempty(part2d.(cad.pnames{j})) == 0
            part2d.graph(j,root) = 1;
        end
    end
end