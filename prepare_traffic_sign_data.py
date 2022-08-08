import os
import os.path as osp



def create_img_paths_as_train_val_lists(classes_known_train:list, classes_known_train_path:str,
                                        classes_background_train:list, classes_background_train_path:str,
                                        empty_training_classes:list, training_classes_with_subfolders:list,
                                        exclude_GTSRB_data:list, VAL_SPLIT_PERCENTAGE:float,
                                        include_background_training_classes:bool):

    '''
    Function gets full image paths of all training images, make train-validation split, and add to respective list.
    :param classes_known_train:
    :param classes_known_train_path:
    :param classes_background_train:
    :param classes_background_train_path:
    :param empty_training_classes:
    :param training_classes_with_subfolders:
    :param exclude_GTSRB_data:
    :param VAL_SPLIT_PERCENTAGE:
    :param include_background_training_classes:
    :return: img_paths_train, img_paths_val
    '''

    # create empty lists to save img paths
    img_paths_train = []
    img_paths_val = []

    #add filenames of GTSRB dataset to list
    for cls_k in classes_known_train:
        if cls_k in empty_training_classes: #Skip folders that do not contain training images (defined some cells above)
            continue

        # if class has a subfolder (defined some cells above) iterate over those folders and then add image paths
        elif cls_k in training_classes_with_subfolders:
            subfolders = os.listdir(osp.join(classes_known_train_path, cls_k))
            subfolders = [sf for sf in subfolders if sf != '.DS_Store'] #remove unwanted files
            for sf in subfolders:
                tmp_img_paths = os.listdir(osp.join(classes_known_train_path, cls_k, sf))
                # exclude GTSRB data if specified above. Excluding means not considering .ppm files (all GTSRB images are .ppm files)
                if cls_k in exclude_GTSRB_data:
                    tmp_img_paths = [osp.join(classes_known_train_path, cls_k, sf, img_path) for img_path in tmp_img_paths if img_path.endswith(

                    )] #filter unwanted files
                else:
                    tmp_img_paths = [osp.join(classes_known_train_path, cls_k, sf, img_path) for img_path in tmp_img_paths if img_path.endswith(('jpeg', 'jpg', 'ppm', 'png'))] #filter unwanted files

                # train-validation split
                img_paths_train += tmp_img_paths[:round(len(tmp_img_paths)*(1-VAL_SPLIT_PERCENTAGE))]
                img_paths_val += tmp_img_paths[round(len(tmp_img_paths)*(1-VAL_SPLIT_PERCENTAGE)):]

        # folder has training images and no subfolders
        else:
            tmp_img_paths = os.listdir(osp.join(classes_known_train_path, cls_k))
            # exclude GTSRB data if specified above. Excluding means not considering .ppm files (all GTSRB images are .ppm files)
            if cls_k in exclude_GTSRB_data:
                tmp_img_paths = [osp.join(classes_known_train_path, cls_k, img_path) for img_path in tmp_img_paths if img_path.endswith(('jpeg', 'jpg', 'png'))] #filter unwanted files
            else:
                tmp_img_paths = [osp.join(classes_known_train_path, cls_k, img_path) for img_path in tmp_img_paths if img_path.endswith(('jpeg', 'jpg', 'ppm', 'png'))] #filter unwanted files

            # train-validation split
            img_paths_train += tmp_img_paths[:round(len(tmp_img_paths)*(1-VAL_SPLIT_PERCENTAGE))]
            img_paths_val += tmp_img_paths[round(len(tmp_img_paths)*(1-VAL_SPLIT_PERCENTAGE)):]


    # background classes are only considered for training if specified
    if include_background_training_classes:
        for cls_b in classes_background_train:
            tmp_img_paths = os.listdir(osp.join(classes_background_train_path, cls_b))
            tmp_img_paths = [osp.join(classes_background_train_path, cls_b, img_path) for img_path in tmp_img_paths if img_path.endswith(('jpeg', 'jpg', 'ppm', 'png'))] #filter unwanted files

            # train-validation split
            img_paths_train += tmp_img_paths[:round(len(tmp_img_paths)*(1-VAL_SPLIT_PERCENTAGE))]
            img_paths_val += tmp_img_paths[round(len(tmp_img_paths)*(1-VAL_SPLIT_PERCENTAGE)):]


    return img_paths_train, img_paths_val


def create_test_img_paths_as_list(classes_known_test:list, classes_known_test_path:str,
                                  classes_unknown_test:list, classes_unknown_test_path:str,
                                  classes_background_test:list, classes_background_test_path:str,
                                  empty_test_classes:list, test_classes_with_subfolders:list):

    """
    Function gets full image paths of all test images and adds them to one list.
    :param classes_known_test:
    :param classes_known_test_path:
    :param classes_unknown_test:
    :param classes_unknown_test_path:
    :param classes_background_test:
    :param classes_background_test_path:
    :param empty_test_classes:
    :param test_classes_with_subfolders:
    :return:
    """

    # create empty list to save img paths
    img_paths_test = []

    # known classes
    for cls_K in classes_known_test:
        tmp_img_paths = os.listdir(osp.join(classes_known_test_path, cls_K))
        tmp_img_paths = [osp.join(classes_known_test_path, cls_K, img_path) for img_path in tmp_img_paths if img_path.endswith(('jpeg', 'jpg', 'ppm', 'png'))]
        img_paths_test += tmp_img_paths

    # background classes (should be predicted as unknown)
    for cls_b in classes_background_test:
        tmp_img_paths = os.listdir(osp.join(classes_background_test_path, cls_b))
        tmp_img_paths = [osp.join(classes_background_test_path, cls_b, img_path) for img_path in tmp_img_paths if img_path.endswith(('jpeg', 'jpg', 'ppm', 'png'))]
        img_paths_test += tmp_img_paths

    # unknown classes
    for cls_u in classes_unknown_test:
        # unknown classes with empty folders
        if cls_u in empty_test_classes:
            continue

        # unknown classes with subfolders
        elif cls_u in test_classes_with_subfolders:
            subfiles_folders = os.listdir(osp.join(classes_unknown_test_path, cls_u))
            subfiles_folders = [sf for sf in subfiles_folders if sf != '.DS_Store']
            for f in subfiles_folders:
                # if file is a folder, then get images from the folder
                if osp.isdir(osp.join(classes_unknown_test_path, cls_u, f)):
                    tmp_img_paths = os.listdir(osp.join(classes_unknown_test_path, cls_u, f))
                    tmp_img_paths = [osp.join(classes_unknown_test_path, cls_u, f, img_path) for img_path in tmp_img_paths if img_path.endswith(('jpeg', 'jpg', 'ppm', 'png'))]
                    img_paths_test += tmp_img_paths
                # if file is an image, append the img path to image_paths_test
                elif f.endswith(('jpeg', 'jpg', 'ppm', 'png')):
                    img_paths_test.append(osp.join(classes_unknown_test_path, cls_u, f))

        else:
            # unknown classes without subfolders
            tmp_img_paths = os.listdir(osp.join(classes_unknown_test_path, cls_u))
            tmp_img_paths = [osp.join(classes_unknown_test_path, cls_u, img_path) for img_path in tmp_img_paths if img_path.endswith(('jpeg', 'jpg', 'ppm', 'png'))]
            img_paths_test += tmp_img_paths

    return img_paths_test