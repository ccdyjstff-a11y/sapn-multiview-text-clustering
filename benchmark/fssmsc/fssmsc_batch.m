function fssmsc_batch(manifest_path)
% Serial MATLAB worker. Paths reference label-free payloads only.
here=fileparts(mfilename('fullpath'));addpath(here);
maxNumCompThreads(1);
jobs=jsondecode(fileread(manifest_path));
for i=1:numel(jobs)
    job=jobs(i);
    try
        fprintf('FSSMSC_START %d %s\n',i,job.input);
        fssmsc_pair_adapter(job.input,job.output);
        fprintf('FSSMSC_DONE %d %s\n',i,job.output);
    catch failure
        report=struct('status','exception','identifier',failure.identifier,'message',failure.message);
        fid=fopen([job.output '.error.json'],'w');fwrite(fid,jsonencode(report),'char');fclose(fid);
        fprintf('FSSMSC_FAILED %d %s\n',i,failure.message);
    end
end
end
