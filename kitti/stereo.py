import os

from kitti.data import data_dir


def load_image(index, test=False, right=False, frame=10, multiview=False):
    import scipy.ndimage
    path = os.path.join(
        data_dir,
        'data_stereo_flow_multiview' if multiview else 'data_stereo_flow',
        'testing' if test else 'training',
        'image_1' if right else 'image_0',
        "%06d_%02d.png" % (index, frame))
    return scipy.ndimage.imread(path)


def load_pair(index, test=False, frame=10, multiview=False):
    left = load_image(index, right=False, test=test, frame=frame, multiview=multiview)
    right = load_image(index, right=True, test=test, frame=frame, multiview=multiview)
    return left, right


def load_disp(index, test=False, occluded=False):
    assert not test, "disparity not available for test data"

    import scipy.ndimage
    path = os.path.join(
        data_dir,
        'data_stereo_flow',
        'testing' if test else 'training',
        'disp_occ' if occluded else 'disp_noc',
        "%06d_10.png" % (index))
    return scipy.ndimage.imread(path)
