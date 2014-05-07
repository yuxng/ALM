% vertices: points in 3D
% P: fitted plane in 3D
% center: rectangle center in 3D
% xaxis: the direction of largest variance of vertices
% yaxis: the direction of second largest variance of vertices
% F: constructed part face in 3D
function [F, P, center, xaxis, yaxis] = fit_plane(vertices, shinkage_width, shinkage_height, part_direction)

% linear regression to fit a 2D plane
nv = size(vertices, 1);
vertices_homo = [vertices ones(nv, 1)];
[U,S,V] = lansvd(vertices_homo, 4, 'L');
P = V(:,end);
% [U,S,V] = svd(vertices - ones(nv,1)*mean(vertices));

angle = acos(dot(P(1:3), part_direction) / norm(P(1:3)));
if angle > pi/2
    P = -1 * P;
end

% project vertices to the plane
pnorm2 = P(1:3)'*P(1:3);
pvertices = zeros(nv, 3);
for i = 1:nv
    pvertices(i,:) = vertices(i,:) - ((vertices_homo(i,:)*P)/pnorm2)*P(1:3)';
end

% build a local coordinate system in the plane
origin = -(P(4)/pnorm2)*P(1:3)';
center = mean(pvertices);
temp = zeros(nv, 3);
for i = 1:nv
    temp(i,:) = pvertices(i,:) - center;
end

[U, S, V] = lansvd(temp, 3, 'L');
xaxis = V(:,1)';
yaxis = V(:,2)';

% form a right hand coordinate system
zaxis = cross(xaxis, yaxis);
if acos(dot(zaxis, P(1:3)/norm(P(1:3)))) > acos(dot(-zaxis, P(1:3)/norm(P(1:3))))
    xaxis = -1 * xaxis;
end

if part_direction(1) == 0,
    
    % find the closer one to [1 0 0] among xaxis and yaxis
    D = pdist([1 0 0; xaxis/norm(xaxis); yaxis/norm(yaxis)]);
    if D(1) < D(2),
        theta = acos(dot([1 0 0], xaxis) / norm(xaxis));
        u = cross([1 0 0], xaxis);
        u = u/norm(u);
    else
        theta = acos(dot([1 0 0], yaxis) / norm(xaxis));
        u = cross([1 0 0], yaxis);
        u = u/norm(u);
    end
    
    a = theta;
    R = [cos(a) + u(1)^2*(1-cos(a)) u(1)*u(2)*(1-cos(a))-u(3)*sin(a) u(1)*u(3)*(1-cos(a))+u(2)*sin(a);
        u(2)*u(1)*(1-cos(a))+u(3)*sin(a) cos(a)+u(2)^2*(1-cos(a)) u(2)*u(3)*(1-cos(a))-u(1)*sin(a);
        u(3)*u(1)*(1-cos(a))-u(2)*sin(a) u(3)*u(2)*(1-cos(a))+u(1)*sin(a) cos(a)+u(3)^2*(1-cos(a))];
    xaxis = xaxis*R;
    yaxis = yaxis*R;
end

% represent points in the plane using the local coordinates
v2d = zeros(nv, 2);
for i = 1:nv
    v2d(i,1) = dot(pvertices(i,:) - origin, xaxis);
    v2d(i,2) = dot(pvertices(i,:) - origin, yaxis);
end

% bounding box in the plane
% if part_direction(1) == 0,
%     center = [dot(center - origin, xaxis), dot(center - origin, yaxis)];
% else
center = [(min(v2d(:,1))+max(v2d(:,1)))/2, (min(v2d(:,2))+max(v2d(:,2)))/2];
width = (max(v2d(:,1)) - min(v2d(:,1))) * shinkage_width;
height = (max(v2d(:,2)) - min(v2d(:,2))) * shinkage_height;

r1 = center + [-width/2 -height/2];
r2 = center + [width/2 -height/2];
r3 = center + [width/2 height/2];
r4 = center + [-width/2 height/2];

% find the 3d corrdinates of the 4 cornors of the rectangle
p1 = r1(1)*xaxis + r1(2)*yaxis + origin;
p2 = r2(1)*xaxis + r2(2)*yaxis + origin;
p3 = r3(1)*xaxis + r3(2)*yaxis + origin;
p4 = r4(1)*xaxis + r4(2)*yaxis + origin;

% find the 3d coordinates of the rectangle center
center = center(1)*xaxis + center(2)*yaxis + origin;

% build the face
F = [p1; p2; p3; p4; p1];

% build the face for part
% F = zeros(numel(index), 3);
% for i = 1:numel(index)
%     F(i,:) = v2d(i,1)*xaxis + v2d(i,2)*yaxis + origin;
% end

% plot
% figure;
% plot3(pvertices(:,1), pvertices(:,2), pvertices(:,3), 'o');
% hold on;
% patch(F(:,1), F(:,2), F(:,3), 'r');
% axis equal;
