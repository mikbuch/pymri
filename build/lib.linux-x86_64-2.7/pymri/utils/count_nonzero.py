def count_nonzero(image):
    # GET DATA FROM NIFTI
    import nibabel
    img = nibabel.load(image)
    data = img.get_data()

    # COUNT NON ZERO VOXELS
    import numpy as np
    nonzero = np.count_nonzero(data)

    return nonzero

