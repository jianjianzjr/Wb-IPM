%% removepath.m
directory = pwd;
path(directory, path)

rmpath([directory, '/genHyBR'])
rmpath([directory, '/genHyBRrecycle'])
rmpath([directory, '/toeplitz'])
rmpath([directory, '/seismic'])
rmpath([directory, '/HyBR'])
rmpath([directory, '/HyBRrecycle'])
rmpath([directory, '/AIRToolsII-master'])
rmpath([directory, '/IRtools-master'])
clear directory
