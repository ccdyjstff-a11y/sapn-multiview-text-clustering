function fssmsc_prepare_common(input_path,output_path)
here=fileparts(mfilename('fullpath'));addpath(fullfile(here,'..','literature','FSSMSC','master'));addpath(here);
payload=load(input_path);
assert(all(ismember(fieldnames(payload),{'X','k','m','seed'})),'Only features, k and seed enter anchor preparation.');
common=fssmsc_make_common(payload.X,payload.k,payload.m,payload.seed);
save(output_path,'-struct','common','-v7');
end
