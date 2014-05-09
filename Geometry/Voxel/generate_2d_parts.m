% render parts into an image
function parts2d = generate_2d_parts(cad)

% initialization
a = cad.azimuth;
e = cad.elevation;
d = cad.distance;
na = numel(a);
ne = numel(e);
nd = numel(d);
parts2d(na*ne*nd).azimuth = a(end);
parts2d(na*ne*nd).elevation = e(end);
parts2d(na*ne*nd).distance = d(end);
pnames = cad.pnames;
parts = cad.parts;
N = numel(parts);

count = 0;
for n = 1:na
    for m = 1:ne
        % visibility of the cad model under the current viewpoint
        visibility = check_visibility(cad, a(n), e(m), mean(d));
        for o = 1:nd
            % initialize part
            count = count+1;
            parts2d(count).azimuth = a(n);
            parts2d(count).elevation = e(m);
            parts2d(count).distance = d(o);
            parts2d(count).theta = 0;
            parts2d(count).px = cad.viewport / 2;
            parts2d(count).py = cad.viewport / 2;
            parts2d(count).viewport = cad.viewport;
            parts2d(count).root = 0;
            parts2d(count).graph = zeros(N);
            for i = 1:numel(pnames)
                parts2d(count).(pnames{i}) = [];
            end
            parts2d(count).centers = zeros(N, 2);
            parts2d(count).homographies = cell(N, 1);
            
            % render parts
            for i = 1:N
                % check the visibility of the part
                if cad.roots(i) == 1
                    view_index = find_interval(a(n), cad.view_num);
                    if view_index == parts(i).view_index
                        flag = 1;
                        parts2d(count).root = i;
                    else
                        flag = 0;
                    end
                else
                    flag = check_visibility_part(visibility, parts(i).grid, cad.occ_per);
                end
                if flag == 0
                    continue;
                end
                % projection
                x3d = parts(i).x3d;
                x2d = project_part_points(x3d, a(n), e(m), d(o), cad.viewport);

                % build the part shape
                x1 = min(x2d(:,1));
                x2 = max(x2d(:,1));
                y1 = min(x2d(:,2));
                y2 = max(x2d(:,2));
                p = [x1 y1; x1 y2; x2 y2; x2 y1; x1 y1];
                c = [(x1+x2)/2 (y1+y2)/2];

                % translate the part center to the orignal
                parts2d(count).(pnames{i}) = p - repmat(c, size(p,1), 1);
                parts2d(count).centers(i,:) = c';

                % compute the homography for transfering current view of the part
                % to frontal view using four point correspondences
                % coefficient matrix
                A = zeros(8,9);
                % construct the coefficient matrix
                X = parts2d(count).(pnames{i});
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
                parts2d(count).homographies{i} = H;
            end
            % construct graph, each row stores the parents of the node
            root = parts2d(count).root;
            for i = 1:N
                if i ~= root && isempty(parts2d(count).(pnames{i})) == 0
                    if cad.roots(i) == 0
                        parts2d(count).graph(i,root) = 1;
                    end
                end
            end
            % end loop for one viewpoint
        end
    end
end