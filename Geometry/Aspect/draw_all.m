function draw_all

subtype = {'hatchback', 'minivan', 'sedan', 'SUV', 'truck', 'wagon'};

% load cad model
n = numel(subtype);
cads = cell(n, 1);
for i = 1:n
    object = load([subtype{i} '.mat']);
    cads{i} = object.cad;
    subplot(2, 3, i);
    draw(cads{i});
    title(subtype{i});
end