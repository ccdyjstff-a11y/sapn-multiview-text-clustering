function run_ssa_synthetic_qualification(outdir)
% Four predeclared tiny algorithm calls, one pre-fit zero-row rejection.
% No real corpus, true classes, candidate selection or clustering scores.
here=fileparts(mfilename('fullpath'));
src=fullfile(here,'private_generated');
addpath(src,'-begin');
names={'ssa_minimal_profile','Create_KNNgraph','NormalizationK','normalization', ...
       'Update_V_v','Update_W_v','Update_Teta','Cost_function'};
for i=1:numel(names)
    assert(strcmpi(which(names{i}),fullfile(src,[names{i} '.m'])),'SSA:Path','Unexpected function resolution.');
end
jsonpath=fullfile(outdir,'SSA_SYNTHETIC_PROFILE_RESULT.json');
matpath=fullfile(outdir,'SSA_SYNTHETIC_PROFILE_STATES.mat');
assert(~isfile(jsonpath) && ~isfile(matpath),'SSA:Output','Never overwrite qualification evidence.');
rng(90210,'twister');
fea={0.1+rand(24,3),0.1+rand(24,4),0.1+rand(24,5)};
ML=[1 2;3 4;5 6]; CL=[1 3;2 5;4 6]; empty=zeros(0,2);
specs={struct('name','mixed','ML',ML,'CL',CL), ...
       struct('name','ML_only','ML',ML,'CL',empty), ...
       struct('name','CL_only','ML',empty,'CL',CL), ...
       struct('name','mixed_repeat','ML',ML,'CL',CL)};
cases=cell(1,4); embeddings=cell(1,4); states=cell(1,4);
for i=1:4
    [embeddings{i},d,states{i}]=ssa_minimal_profile(fea,3,specs{i}.ML,specs{i}.CL,48272);
    cases{i}=struct('name',specs{i}.name,'diagnostics',d);
end
repeat_exact=isequaln(embeddings{1},embeddings{4}) && isequaln(states{1},states{4}) ...
    && isequaln(cases{1}.diagnostics,cases{4}.diagnostics);
bad=fea; bad{1}(1,:)=0; rejected=false; rejection_identifier='';
try
    ssa_minimal_profile(bad,3,ML,CL,48272);
catch err
    rejection_identifier=err.identifier;
    rejected=strcmp(err.identifier,'SSA:UnsupportedZeroRow');
end
% This isolated check documents the corrected pair semantics, not performance.
bad_index=ones(8); pair_index=ones(8); test_cl=[1 2;3 4];
bad_index(test_cl(:,1)',test_cl(:,2)')=0;
bad_index(test_cl(:,2)',test_cl(:,1)')=0;
pair_index(sub2ind([8 8],test_cl(:,1),test_cl(:,2)))=0;
pair_index(sub2ind([8 8],test_cl(:,2),test_cl(:,1)))=0;
pair_index_ok=nnz(bad_index==0)==8 && nnz(pair_index==0)==4;
interface_ok=repeat_exact && rejected && pair_index_ok ...
    && cases{1}.diagnostics.ML_count==3 && cases{1}.diagnostics.CL_count==3 ...
    && cases{2}.diagnostics.CL_count==0 && cases{3}.diagnostics.ML_count==0;
operational=cellfun(@(x)x.diagnostics.operational_contract,cases);
report=struct('kind','SSA_MINIMAL_PROFILE_SYNTHETIC_RESULT', ...
    'scope','Tiny fixed-profile interface and numerical diagnostics. Not a performance benchmark or faithful-equation qualification.', ...
    'matlab_version',version,'n',24,'view_dimensions',[3 4 5],'known_k',3, ...
    'algorithm_calls',4,'real_corpus_calls',0,'full_class_labels_used',false,'clustering_scores_computed',false, ...
    'interface_checks_passed',interface_ok,'mixed_same_seed_exact_repeat',repeat_exact, ...
    'zero_row_rejected_before_fit',rejected,'zero_row_rejection_identifier',rejection_identifier, ...
    'two_pair_index_check',struct('original_zero_entries',nnz(bad_index==0),'repaired_zero_entries',nnz(pair_index==0),'passed',pair_index_ok), ...
    'case_operational_contracts',operational,'all_case_operational_contracts',all(operational), ...
    'cases',{cases},'NaN_Inf_JSON_representation','MATLAB jsonencode converts nonfinite numeric diagnostics to null; raw values remain in MAT state report.', ...
    'synthetic_pass_is_not_real_data_or_novelty_or_performance_validation',true);
save(matpath,'report','embeddings','states','fea','ML','CL','-v7');
f=fopen(jsonpath,'w','n','UTF-8'); assert(f>=0,'SSA:Output','Cannot open report.');
c=onCleanup(@()fclose(f));
fprintf(f,'%s',jsonencode(report,'PrettyPrint',true));
disp(jsonpath);
assert(interface_ok,'SSA:Interface','A declared interface check failed; report retained.');
end
