function PS_C = PS_Select(PS,LABEL,REG)
%
% PS_C = PS_Select(PS,LABEL,REG)
%
% This function is to plot the selected results what user needs from 
% PS structure by setting 'LABEL' and 'REG'. These two variable have 
% to be cell including char type. The possible variables are:
%
% Input:
%   LABEL - [ 'HyBR' | 'rblw' | 'oas' | 'genHyBR' | 'mixHyBR',
%            'genHyBR1' | 'genHyBR2' | 'genHyBR-data-driven']
%           * Note that LABEL can have other types 
%             if user defines new LABEL in EX1_spherical.m
%     REG - ['optimal' | 'upre' | 'gcv' | 'wgcv']
%
% Output:
%    PS_C - includes the selected results from user LABEL and REG.
%
% For example: 
%   If user wants to plot results from 'mixHyBR' for LABEL and 'optimal', 
%   'gcv', 'upre' for REG, then put:
%       LABEL = {'mixHyBR'}
%         REG = {'optimal','gcv','upre'}
%   
%
% T.Cho  Nov/2019
n = size(PS,1);
PS_C = [];

for i = 1:n
    
    if isempty(LABEL) ~= 1 && isempty(REG) ~= 1
        if sum(strcmp(LABEL,PS{i,1}))>0 && sum(strcmp(REG,PS{i,2}))>0
            PS_C = copyStruct(PS,i,PS_C);
        end
    elseif isempty(LABEL) ~= 1 && isempty(REG) == 1
        if sum(strcmp(LABEL,PS{i,1}))>0 
            PS_C = copyStruct(PS,i,PS_C);
        end
    elseif isempty(LABEL) == 1 && isempty(REG) ~= 1
        if sum(strcmp(REG,PS{i,2}))>0 
            PS_C = copyStruct(PS,i,PS_C);
        end
    else 
        PS_C = PS;
    end
    
end

function PS_C = copyStruct(PS,i,PS_C)
    if isempty(PS_C) == 1
       j = 1; 
    else
       j = size(PS_C,1) + 1; 
    end
    PS_C{j,1} = PS{i,1};
    PS_C{j,2} = PS{i,2};
    PS_C{j,3} = PS{i,3};
    PS_C{j,4} = PS{i,4};
end
        
end 