from nipype.interfaces.io import DataFinder

df = DataFinder()
df.inputs.root_paths = '/tmp/exp_01'
df.inputs.match_regex = '.+/(?P<series_dir>.+)/(?P<basename>.+)\.nii.gz'
result = df.run()
print(result.outputs.out_paths)
