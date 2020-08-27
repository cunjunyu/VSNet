import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import tensorboard_logger as tb

import dataset
from model import VSNet, combined_loss_quat
from utils import check_dir, axis_angle_from_quat, normalize_q, get_stem, accuracy_thres_curve
from transformations import angle_between_vectors, euler_from_quaternion

model_name = 'VSNet-rotate-3900train'
model_pretrained = None
num_classes = 7

root_dir = '/home/xinmatrix/data/fyp_ycj'
pattern = 'six_dof_1cm5deg_'
#set_list = [#'new',
 # 'White',
# 'A2']
#  'White_small',
#   'A3',
#  'B2',
#  'C1',
#  'C2',
#   'C3']
set_list = ['rotate']

img_dir_list = [root_dir + '/' + pattern + my_set + '/img' for my_set in set_list]
label_dir_list = [root_dir + '/' + pattern + my_set + '/label' for my_set in set_list]

save_root_dir = './'
log_name = save_root_dir + '/' + model_name + '/' + 'log.txt'
img_size = (640, 480)

train_size_list = [3900] * len(set_list)
dev_size_list = [50] * len(set_list)
test_size_list = [50] * len(set_list)

num_epochs = 10
batch_size = 256
aug_factor = 0.25
num_workers = 8

learning_rate = 1e-4
milestones = [int(num_epochs * 0.4), int(num_epochs * 0.6), int(num_epochs * 0.8)]
# milestones = list(range(num_epochs))
gamma = 0.5
momentum = 0.9
# gamma = 0.3
limits = None
weights = [0.99, 0.01]

mode = ('train', 'train')  # train train set
# mode = ('eval', 'train')  # eval train set
# mode = ('eval', 'dev')  # eval dev set
# mode = ('eval', 'test')  # eval test set

random_seed = 2


def prepare_loaders(return_path=False):
    train_set_paths, dev_set_paths, test_set_paths = fyp_dataset.split_sets(img_dir_list=img_dir_list,
                                                                            label_dir_list=label_dir_list,
                                                                            train_size_list=train_size_list,
                                                                            dev_size_list=dev_size_list,
                                                                            test_size_list=test_size_list,
                                                                            random_seed=random_seed,
                                                                            aug_factor=aug_factor,
                                                                            limits=limits, weights=weights)

    train_set = dataset.VSDataset(set_paths=train_set_paths, img_size=img_size, return_path=return_path)
    dev_set = dataset.VSDataset(set_paths=dev_set_paths, img_size=img_size, return_path=return_path)
    test_set = dataset.VSDataset(set_paths=test_set_paths, img_size=img_size, return_path=return_path)

    train_size_aug = len(train_set)
    dev_size_aug = len(dev_set)
    test_size_aug = len(test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, dev_loader, test_loader, train_size_aug, dev_size_aug, test_size_aug


def mode_train(train_loader, dev_loader, train_size_aug, dev_size_aug):
    check_dir(save_root_dir + '/' + model_name)

    device = torch.device('cuda')

    if model_pretrained:
        print('Loading pretrained model from {}'.format(save_root_dir + '/' + model_pretrained + '/model.pth'))
        model = torch.load(save_root_dir + '/' + model_pretrained + '/model.pth', map_location=device)
    else:
        model = VSNet(num_classes=num_classes)
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    model.to(device)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    tb.configure(save_root_dir + '/' + model_name)

    start_time = time.time()

    tb_count = 0
    for epoch in range(num_epochs):

        scheduler.step()

        # Training
        model.train()
        running_loss = 0.0
        for i, sample in enumerate(train_loader, 0):
            if i == 1 and epoch == 0:
                start_time = time.time()
            img_a, img_b, label = sample

            optimizer.zero_grad()

            img_a = img_a.to(device)
            img_b = img_b.to(device)
            label = label.to(device)

            output = model(img_a, img_b)

            loss = combined_loss_quat(output, label, weights=weights)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * output.shape[0]

            output = output.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            error = np.zeros(8)

            for j in range(output.shape[0]):
                error[:3] += np.abs(output[j, :3] - label[j, :3])

                quat_output = normalize_q(output[j, 3:])
                quat_label = label[j, 3:]

                axis_output, angle_output = axis_angle_from_quat(quat_output)
                axis_label, angle_label = axis_angle_from_quat(quat_label)

                error_mag = np.abs(angle_output - angle_label)
                error_mag = error_mag if error_mag < np.pi else error_mag - np.pi
                error_dir = angle_between_vectors(axis_output, axis_label)
                error[3] += np.nan_to_num(error_mag)
                error[4] += np.nan_to_num(error_dir)

                rpy_output = np.array(euler_from_quaternion(quat_output))
                rpy_label = np.array(euler_from_quaternion(quat_label))
                error[5:] += np.abs(rpy_output - rpy_label)

            error /= output.shape[0]
            error[:3] *= 1000
            error[3:] = np.rad2deg(error[3:])
            est_time = (time.time() - start_time) / (epoch * len(train_loader) + i + 1) * (
                    num_epochs * len(train_loader))
            est_time = str(datetime.timedelta(seconds=est_time))
            print(
                '[TRAIN][{}][EST:{}] Epoch {}, Batch {}, Loss = {:0.7f}, error: x={:0.2f}mm,y={:0.2f}mm,z={:0.2f}mm,mag={:0.2f}deg,dir={:0.2f}deg,roll={:0.2f}deg,pitch={:0.2f}deg,yaw={:0.2f}deg'.format(
                    time.time() - start_time, est_time, epoch + 1, i + 1,
                    loss.item(), *error))

            tb.log_value(name='Loss', value=loss.item(), step=tb_count)
            tb.log_value(name='x/mm', value=error[0], step=tb_count)
            tb.log_value(name='y/mm', value=error[1], step=tb_count)
            tb.log_value(name='z/mm', value=error[2], step=tb_count)
            tb.log_value(name='mag/deg', value=error[3], step=tb_count)
            tb.log_value(name='dir/deg', value=error[4], step=tb_count)
            tb.log_value(name='roll/deg', value=error[5], step=tb_count)
            tb.log_value(name='pitch/deg', value=error[6], step=tb_count)
            tb.log_value(name='yaw/deg', value=error[7], step=tb_count)
            tb_count += 1

        # Dev eval
        model.eval()
        with torch.no_grad():
            running_error_dev = np.zeros(8)
            # running_error_dev = np.zeros(2)
            for i, sample in enumerate(dev_loader, 0):
                img_a, img_b, label = sample

                img_a = img_a.to(device)
                img_b = img_b.to(device)

                output = model(img_a, img_b)

                output = output.cpu().detach().numpy()

                label = label.numpy()

                error = np.zeros(8)
                # error = np.zeros(2)

                for j in range(output.shape[0]):
                    error[:3] += np.abs(output[j, :3] - label[j, :3])

                    quat_output = normalize_q(output[j, 3:])
                    quat_label = label[j, 3:]

                    axis_output, angle_output = axis_angle_from_quat(quat_output)
                    axis_label, angle_label = axis_angle_from_quat(quat_label)

                    error_mag = np.abs(angle_output - angle_label)
                    error_mag = error_mag if error_mag < np.pi else error_mag - np.pi
                    error_dir = angle_between_vectors(axis_output, axis_label)
                    error[3] += np.nan_to_num(error_mag)
                    error[4] += np.nan_to_num(error_dir)

                    rpy_output = np.array(euler_from_quaternion(quat_output))
                    rpy_label = np.array(euler_from_quaternion(quat_label))
                    error[5:] += np.abs(rpy_output - rpy_label)

                error[:3] *= 1000
                error[3:] = np.rad2deg(error[3:])

                running_error_dev += error
                error /= output.shape[0]

                print(
                    '[EVAL][{}] Epoch {}, Batch {}, error: x={:0.2f}mm,y={:0.2f}mm,z={:0.2f}mm,mag={:0.2f}deg,dir={:0.2f}deg'.format(
                        time.time() - start_time, epoch + 1, i + 1, *error))

        average_loss = running_loss / train_size_aug
        average_error = running_error_dev / dev_size_aug
        print(
            '[SUMMARY][{}] Summary: Epoch {}, loss = {:0.7f}, dev_eval: x={:0.2f}mm,y={:0.2f}mm,z={:0.2f}mm,mag={:0.2f}deg,dir={:0.2f}deg,roll={:0.2f}deg,pitch={:0.2f}deg,yaw={:0.2f}deg\n\n'.format(
                time.time() - start_time, epoch + 1, average_loss, *average_error))

        tb.log_value(name='Dev loss', value=average_loss, step=epoch)
        tb.log_value(name='Dev x/mm', value=average_error[0], step=epoch)
        tb.log_value(name='Dev y/mm', value=average_error[1], step=epoch)
        tb.log_value(name='Dev z/mm', value=average_error[2], step=epoch)
        tb.log_value(name='Dev mag/deg', value=average_error[3], step=epoch)
        tb.log_value(name='Dev dir/deg', value=average_error[4], step=epoch)
        tb.log_value(name='Dev roll/deg', value=average_error[5], step=epoch)
        tb.log_value(name='Dev pitch/deg', value=average_error[6], step=epoch)
        tb.log_value(name='Dev yaw/deg', value=average_error[7], step=epoch)

        torch.save(model, save_root_dir + '/' + model_name + '/model.pth')
        print('Model saved at {}/{}/model.pth'.format(save_root_dir, model_name))


def mode_eval(loader, size_aug):
    # image checker
    check_image = False
    if check_image:
        xyz_thres = 3  # mm
        rpy_thres = 2  # deg
        paths = []  # for display of bad image pairs

    # accuracy-threshold curve
    make_curve = True
    if make_curve:
        xyz_error_max = 10.0  # mm
        xyz_error_reso = 0.01  # mm
        rpy_error_max = 5.0  # deg
        rpy_error_reso = 0.01  # deg

    device = torch.device('cuda')

    model = torch.load(save_root_dir + '/' + model_name + '/model.pth')

    model.eval()
    model.to(device)

    data = [[] for i in range(8)]  # for box plotting
    running_error_test = np.zeros(8)
    start_time = time.time()
    for i, sample in enumerate(loader):

        img_a, img_b, label, img_a_path, img_b_path, label_a_path, label_b_path = sample
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        label = label.to(device)

        output = model(img_a, img_b)
        output = output.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        # print('output = \n{}\nlabel = \n{}'.format(output, label))

        error = np.zeros(8)
        for j in range(output.shape[0]):

            # print('{} vs {}'.format(output[j], label[j]))

            xyz_error = np.abs(output[j, :3] - label[j, :3]) * 1000
            error[:3] += xyz_error
            # error[:2] += np.abs(output[j, :2] - label[j, :2]
            quat_output = normalize_q(output[j, 3:])
            quat_label = label[j, 3:]

            axis_output, angle_output = axis_angle_from_quat(quat_output)
            axis_label, angle_label = axis_angle_from_quat(quat_label)

            # print('output[j, 3] = {}, label[j, 3] = {}'.format(output[j, 3], label[j, 3]))
            error_mag = np.abs(angle_output - angle_label)
            error_mag = error_mag if error_mag < np.pi else error_mag - np.pi
            error_dir = angle_between_vectors(axis_output, axis_label)
            error_mag = np.rad2deg(np.nan_to_num(error_mag))
            error_dir = np.rad2deg(np.nan_to_num(error_dir))
            error[3] += error_mag
            error[4] += error_dir

            rpy_output = np.array(euler_from_quaternion(quat_output))
            rpy_label = np.array(euler_from_quaternion(quat_label))
            rpy_error = np.rad2deg(np.abs(rpy_output - rpy_label))
            error[5:] += rpy_error

            data[0].append(xyz_error[0])
            data[1].append(xyz_error[1])
            data[2].append(xyz_error[2])
            data[3].append(error_mag)
            data[4].append(error_dir)
            data[5].append(rpy_error[0])
            data[6].append(rpy_error[1])
            data[7].append(rpy_error[2])

            if check_image:
                if np.any(xyz_error > xyz_thres) or np.any(rpy_error > rpy_thres):
                    checker_output = np.ones(8) * -1
                    checker_output[:3] = output[j, :3] * 1000
                    checker_output[5:] = np.rad2deg(rpy_output)

                    checker_label = np.ones(8) * -1
                    checker_label[:3] = label[j, :3] * 1000
                    checker_label[5:] = np.rad2deg(rpy_label)

                    paths.append((img_a_path[j], img_b_path[j], checker_output, checker_label, error))

        running_error_test += error
        error /= output.shape[0]

        print(
            '[EVAL][{}] Batch {}, error: x={}mm,y={}mm,z={}mm,mag={}deg,dir={}deg,roll={}deg,pitch={}deg,yaw={}deg'.format(
                time.time() - start_time, i + 1, *error))

    average_error = running_error_test / test_size_aug
    print(
        'Summary: test_eval: x={:0.2f}mm,y={:0.2f}mm,z={:0.2f}mm,mag={:0.2f}deg,dir={:0.2f}deg,roll={:0.2f}deg,pitch={:0.2f}deg,yaw={:0.2f}deg\n\n'.format(
        *average_error))
    fig1 = plt.figure(0)
    ax11 = fig1.add_subplot(121)
    ax11.set_title('Translation Errors (mm)')
    bp11 = ax11.boxplot(data[:3])
    ax11.set_xticklabels(['x', 'y', 'z'])
    # only show max and min outlier
    for outliers in bp11['fliers']:
        # outliers.set_data([[outliers.get_xdata()[0],outliers.get_xdata()[0]],[np.min(outliers.get_ydata()),‌​np.max(outliers.get_ydata())]])
        outliers.set_data([[outliers.get_xdata()[0]], [[np.max(outliers.get_ydata())]]])

    ax12 = fig1.add_subplot(122)
    ax12.set_title('Rotation Errors (deg)')
    bp12 = ax12.boxplot(data[5:])
    ax12.set_xticklabels(['roll', 'pitch', 'yaw'])
    # only show max and min outlier
    for outliers in bp12['fliers']:
        # outliers.set_data([[outliers.get_xdata()[0],outliers.get_xdata()[0]],[np.min(outliers.get_ydata()),‌​np.max(outliers.get_ydata())]])
        outliers.set_data([[outliers.get_xdata()[0]], [[np.max(outliers.get_ydata())]]])

    plt.savefig(save_root_dir + '/' + model_name + '/error_distribution.png')
    plt.show()

    if check_image:
        idx = 0
        idx_prev = -1
        print(len(paths))
        while paths:

            if idx is not idx_prev:
                idx_prev = idx

                print('img_a: {}, img_b: {}'.format(get_stem(paths[idx][0]), get_stem(paths[idx][1])))
                print('output: {}\nlabel: {}\nerror:{}'.format(paths[idx][2], paths[idx][3], paths[idx][4]))

                img_a = cv2.imread(paths[idx][0])
                img_b = cv2.imread(paths[idx][1])

            cv2.imshow('a', img_a)
            cv2.imshow('b', img_b)

            key = cv2.waitKeyEx(0)

            if key == 97:  # a
                idx -= 1
                idx = max(idx, 0)
            elif key == 100:  # d
                idx += 1
                idx = min(idx, len(paths) - 1)
            elif key == 27:  # exit
                break
            else:
                print('Unknown key: {}'.format(key))

    if make_curve:
        xyz_thres_list = list(np.arange(0, xyz_error_max, xyz_error_reso))
        rpy_thres_list = list(np.arange(0, rpy_error_max, rpy_error_reso))

        x_accuracy_list, x_thres_list = accuracy_thres_curve(data[0], xyz_thres_list)
        y_accuracy_list, y_thres_list = accuracy_thres_curve(data[1], xyz_thres_list)
        z_accuracy_list, z_thres_list = accuracy_thres_curve(data[2], xyz_thres_list)
        ro_accuracy_list, ro_thres_list = accuracy_thres_curve(data[5], rpy_thres_list)
        pi_accuracy_list, pi_thres_list = accuracy_thres_curve(data[6], rpy_thres_list)
        ya_accuracy_list, ya_thres_list = accuracy_thres_curve(data[7], rpy_thres_list)

        fig2 = plt.figure()
        ax21 = fig2.add_subplot(211)
        ax21.set_xlabel('Threshold (mm)')
        ax21.set_ylabel('Fraction of pass')
        lines21 = ax21.plot(x_thres_list, x_accuracy_list, 'r-', y_thres_list, y_accuracy_list, 'g-', z_thres_list,
                  z_accuracy_list, 'b-')
        ax21.set_xticks(np.arange(min(x_thres_list), max(x_thres_list) + 1, 1.0))
        ax21.legend(lines21, ('x', 'y', 'z'))

        ax22 = fig2.add_subplot(212)
        ax22.set_xlabel('Threshold (deg)')
        ax22.set_ylabel('Fraction of pass')
        lines22 = ax22.plot(ro_thres_list, ro_accuracy_list, 'r-', pi_thres_list, pi_accuracy_list, 'g-', ya_thres_list,
                  ya_accuracy_list, 'b-')
        ax22.set_xticks(np.arange(min(ro_thres_list), max(ro_thres_list) + 1, 0.5))
        ax22.legend(lines22, ('roll', 'pitch', 'yaw'))

        plt.tight_layout()
        plt.savefig(save_root_dir + '/' + model_name + '/error_curve.png')
        plt.show()


if __name__ == '__main__':

    if mode[0] == 'train':

        train_loader, dev_loader, test_loader, train_size_aug, dev_size_aug, test_size_aug = prepare_loaders(
            return_path=False)

        if mode[1] == 'train':
            mode_train(train_loader, dev_loader, train_size_aug, dev_size_aug)
        else:
            raise Exception('Cannot train ' + mode[1] + ' set.')

    elif mode[0] == 'eval':

        train_loader, dev_loader, test_loader, train_size_aug, dev_size_aug, test_size_aug = prepare_loaders(
            return_path=True)

        if mode[1] == 'train':
            mode_eval(train_loader, train_size_aug)
        elif mode[1] == 'dev':
            mode_eval(dev_loader, dev_size_aug)
        elif mode[1] == 'test':
            mode_eval(test_loader, test_size_aug)
        else:
            raise Exception('Cannot eval ' + mode[1] + ' set.')

    else:
        raise Exception('Unknown mode ' + mode[0])
