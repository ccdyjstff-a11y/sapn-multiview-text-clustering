function pcbg_clock_bridge(inputPath, outputDir, authorRoot, runId)
% Runtime instrumentation only. The original worker and author code are unchanged.
start = struct('schema',1,'run_id',runId,'phase','started','sequence',1);
atomicmarker(fullfile(outputDir,'worker_started.json'),start);
workerClock = tic;
try
    pcbg_formal_worker(inputPath,outputDir,authorRoot);
catch err
    finish = struct('schema',1,'run_id',runId,'phase','finished','sequence',2, ...
        'worker_call_wall_seconds',toc(workerClock),'outcome','threw', ...
        'error_identifier',err.identifier,'error_message',err.message);
    atomicmarker(fullfile(outputDir,'worker_finished.json'),finish);
    rethrow(err);
end
finish = struct('schema',1,'run_id',runId,'phase','finished','sequence',2, ...
    'worker_call_wall_seconds',toc(workerClock),'outcome','returned');
atomicmarker(fullfile(outputDir,'worker_finished.json'),finish);
end

function atomicmarker(target,value)
assert(~isfile(target),'PCBGClock:ExistingMarker','Refusing to overwrite marker');
temporary = [target '.tmp'];
assert(~isfile(temporary),'PCBGClock:ExistingTemporary','Refusing stale marker temporary');
handle = fopen(temporary,'w','n','UTF-8');
assert(handle>=0,'PCBGClock:MarkerWrite','Cannot open marker temporary');
try
    fprintf(handle,'%s\n',jsonencode(value));
    code = fclose(handle);
    assert(code==0,'PCBGClock:MarkerClose','Cannot close marker temporary');
catch err
    fclose(handle);
    rethrow(err);
end
% Same-directory atomic commit. Fail closed if this filesystem/JVM cannot
% provide ATOMIC_MOVE; never fall back to a partial copy visible to Python.
sourceFile = javaObject('java.io.File',temporary);
targetFile = javaObject('java.io.File',target);
options = javaArray('java.nio.file.CopyOption',1);
options(1) = javaMethod('valueOf','java.nio.file.StandardCopyOption','ATOMIC_MOVE');
javaMethod('move','java.nio.file.Files',sourceFile.toPath(),targetFile.toPath(),options);
end
