load('Session_Object_2020.mat')

group = 'B6'
group = 'GAD2_ZI_Inh'

inds = [];
for i = 1:numel(S)
    if strcmp(S(i).group, group)
        inds(end+1) = i;
    end
end


T = 600;
s = 100;
for i = inds
   b = S(i).NovelObject.NovOld;
   for ii = 1:numel(b)
       exp = b{ii};
       save([group, '-', S(i).code, '-NovOld-', num2str(ii), '.mat'], '-struct', 'exp')
   end
end