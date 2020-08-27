import os
import sys
import glob
import numpy as np
import transformations as tr


def accuracy_thres_curve(error_list, thres_list):

    num_error = len(error_list)
    num_thres = len(thres_list)

    accuracy_list = [0] * num_thres

    error_list = sorted(error_list)

    error_idx = 0
    thres_idx = 0

    while error_idx < num_error and thres_idx < num_thres:
        if error_list[error_idx] <= thres_list[thres_idx]:
            accuracy_list[thres_idx] = error_idx + 1
            error_idx += 1
        else:
            accuracy_list[thres_idx] = error_idx
            thres_idx += 1

    while thres_idx < num_thres:
        accuracy_list[thres_idx] = error_idx + 1
        thres_idx += 1

    accuracy_list = list(np.array(accuracy_list) / num_error)

    return accuracy_list, thres_list


def axis_angle_from_quat(q):
    if q[0] > 1:
        norm = np.sqrt(np.sum(q ** 2))
        q /= norm

    angle = 2 * np.arccos(q[0])

    s = np.sqrt(1 - q[0] ** 2)
    if s < 1e-4:
        axis = q[1:]
    else:
        axis = q[1:] / s

    return axis, angle


def normalize_q(q):
    q = np.array(q)
    assert q.shape == (4,)
    norm = np.sqrt(np.sum(q ** 2))
    return q / norm


def tf_to_quat(T):
    if T.shape == (12,):
        T.shape = (3,4)
    if T.shape == (3,4):
        T = np.concatenate((T, np.array([[0.0,0.0,0.0,1.0]])), axis=0)
    assert T.shape == (4, 4)
    translation = tr.translation_from_matrix(T)
    # quaternions = np.array(tr.quaternion_from_matrix(T[0:3, 0:3]))
    quaternions = np.array(tr.quaternion_from_matrix(T))
    return np.concatenate((translation, quaternions), axis=0)


def tf_to_dof(T):
    if T.shape == (12,):
        T.shape = (3,4)
    if T.shape == (3,4):
        T = np.concatenate((T, np.array([[0.0,0.0,0.0,1.0]])), axis=0)
    assert T.shape == (4,4), 'T.shape is {}, not (4,4)'.format(T.shape)
    translation = tr.translation_from_matrix(T)
    # euler_angles = np.array(tr.euler_from_matrix(T[0:3, 0:3]))
    euler_angles = np.array(tr.euler_from_matrix(T))
    return np.concatenate((translation, euler_angles), axis=0)


def dof_to_tf(x, y, z, roll, pitch, yaw):
    T = tr.euler_matrix(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))
    T[:3, 3] = [x, y, z]
    return T


def get_stem(path):
    basename = os.path.basename(path)
    stem, _ = os.path.splitext(basename)
    return stem


def get_extension(path):
    basename = os.path.basename(path)
    _, ext = os.path.splitext(basename)
    return ext


def get_files(my_dir, ext):
    if my_dir[-1] != '/':
        my_dir += '/'
    assert os.path.isdir(my_dir), '{} is not a valid directory!'.format(my_dir)

    if ext[0] != '.':
        ext = '.' + ext

    files = glob.glob(my_dir + '*' + ext)
    return files


def make_dir(my_dir):
    if os.path.isdir(my_dir):
        pass  # dir already exists
    else:
        os.makedirs(my_dir)
        print('Make directory {}'.format(my_dir))


# Make directory if directory does not exist.
# Ask to remove content if directory exists.
def check_dir(my_dir):
    if my_dir[-1] == '/':
        my_dir = my_dir.rstrip('/')

    make_dir(my_dir)

    files = os.listdir(my_dir)
    if files:  # if my_dir is not empty
        while True:
            is_delete = input("Files found in " + my_dir + ". They will be removed if you continue. Continue? [ENTER/n]")
            if is_delete == '':
                for file in files:
                    os.remove(my_dir + '/' + file)
                break
            elif is_delete == 'n':
                sys.exit(-1)
