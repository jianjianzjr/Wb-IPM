function output = compress(data)
    eps = 1.5;
    ori_shape = size(data)
    new_shape = round(ori_shape/eps)
    x_expand = zeros(new_shape);
    for i=1:new_shape(1)
        for j = 1:new_shape(2)
            for k = 1:new_shape(3)
                output(i,j,k) = data(floor(i*eps),floor(j*eps),floor(k*eps));
            end
        end
    end
end