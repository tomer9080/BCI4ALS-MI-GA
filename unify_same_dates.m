fileName = 'paths\paths_unify.txt';
FID = fopen(fileName);
data = textscan(FID,'%s');
fclose(FID);
paths = string(data{:});

for i = 1:length(paths):2
    
    MainScript(paths(i)); 
end

fprintf("FINISHED RUN\n");
