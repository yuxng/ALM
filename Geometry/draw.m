% display a CAD model

function draw(cad)

parts = cad.parts;
color = hsv(numel(parts)); 

figure;
hold on;
for i = 1:numel(parts)
    F = parts(i).vertices;
    if i == 11
        patch(F(:,1), F(:,2), F(:,3), 'b');
        center = parts(i).center;
        plot3(center(1), center(2), center(3), 'o');
    else
        patch(F(:,1), F(:,2), F(:,3), color(i,:), 'FaceAlpha', 0.5);
    end
end
axis equal;
axis tight;
% xlabel('x');
% ylabel('y');
% zlabel('z');
view(330, 20);
axis on;
hold off;