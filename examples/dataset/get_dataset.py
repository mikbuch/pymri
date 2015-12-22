from pymri.dataset.datasets import DatasetManager

mvpa_directory = '/tmp/Maestro_Project1/GK011RZJA/Right_Hand/mvpa'
roi_path = '/tmp/Maestro_Project1/GK011RZJA/Right_Hand/mvpa/ROIs/pSMG.nii.gz'
runs = 5
volumes = 145
n_time = 0

# Load the dataset
print('Loading database from %s' % mvpa_directory)
dataset = DatasetManager(
    mvpa_directory=mvpa_directory,
    # conditions has to be tuples
    contrast=(
        ('PlanTool_0', 'PlanTool_5'),
        ('PlanCtrl_0', 'PlanCtrl_5')
        )
    )

dataset.feature_reduction(
    roi_path=roi_path,
    k_features=784,
    reduction_method='SKB'
    )

training_data, test_data, validation_data = dataset.leave_one_run_out(
    runs=runs,
    volumes=volumes,
    n_time=n_time
    )

print('Saving file to /tmp/fmri.pkl')
import pickle
pickle.dump(
    (training_data, test_data),
    open("/tmp/fmri.pkl", "wb")
    )
