function draw_cad(cad)

L3 = cad.grid;
L3(L3 == 1) = 127;
L3(L3 == 0) = 1;
drawLabeledVoxel(L3);