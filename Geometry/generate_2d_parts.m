% render parts into an image
function parts2d = generate_2d_parts(cad)

a = cad.azimuth;
e = cad.elevation;
d = cad.distance;
na = numel(a);
ne = numel(e);
nd = numel(d);

count = 0;
for n = 1:na
    for m = 1:ne
        for o = 1:nd
            count = count+1;
            parts2d(count) = project_cad(cad, a(n), e(m), d(o));
        end
    end
end