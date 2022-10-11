from .mydataset import ImageFolder
from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as transform_lib
from .dsml_aug import BYOLDataTransform, MOCODataTransform, BYOLDataTransform_amlp

def get_loader(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False, val=False, val_data=None):
    
    train_transform = BYOLDataTransform(
        crop_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        blur_prob=[.0, .0],
        solarize_prob=[.0, .2],
        GaussianBlur=1,
        )

    ori_trans = transform_lib.Compose([
        transform_lib.Resize((256, 256)),
        transform_lib.RandomHorizontalFlip(),
        transform_lib.RandomCrop(224),
        transform_lib.ToTensor(),
        transform_lib.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_transform.transforms.append(ori_trans)

    source_folder = ImageFolder(os.path.join(source_path),
                                train_transform,
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                  transform = train_transform,
                                  return_paths = False, return_id=return_id)
    if val:
        source_val_train = ImageFolder(val_data, train_transform, return_id=return_id)
        target_folder_train = torch.utils.data.ConcatDataset([target_folder_train, source_val_train])
        source_val_test = ImageFolder(val_data, transforms[evaluation_path], return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms["eval"],
                                   return_paths=True)

    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,            
            )
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,
            )

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        # persistent_workers=True,
        pin_memory=True,
        )
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        # persistent_workers=True,
        pin_memory=True,
        )
    if val:
        test_loader_source = torch.utils.data.DataLoader(
            source_val_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,
            )
        return source_loader, target_loader, test_loader, test_loader_source

    return source_loader, target_loader, test_loader, target_folder_train



def get_loader_amlp(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False, val=False, val_data=None,
               data_aug_crop=0.8, aug_type=0,
               ):
    
    # train_transform_sourse = BYOLDataTransform_amlp(
    #     crop_size=224,
    #     mean=(0.485, 0.456, 0.406),
    #     std=(0.229, 0.224, 0.225),
    #     blur_prob=[.0, .0],
    #     solarize_prob=[.0, .2],
    #     aug_type=aug_type,
    #     data_aug_crop=data_aug_crop,
    #     )
    train_transform_target = BYOLDataTransform_amlp(
        crop_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        blur_prob=[.0, .0],
        solarize_prob=[.0, .2],
        aug_type=aug_type,
        data_aug_crop=data_aug_crop,
        )
    
    train_transform_sourse = transform_lib.Compose([
        transform_lib.Resize((256, 256)),
        transform_lib.RandomHorizontalFlip(),
        transform_lib.RandomCrop(224),
        transform_lib.ToTensor(),
        transform_lib.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # train_transform_sourse.transforms.append(ori_trans)

    source_folder = ImageFolder(os.path.join(source_path),
                                train_transform_sourse,
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                  transform = train_transform_target,
                                  return_paths = False, return_id=return_id)
    if val:
        source_val_train = ImageFolder(val_data, train_transform_sourse, return_id=return_id)
        target_folder_train = torch.utils.data.ConcatDataset([target_folder_train, source_val_train])
        source_val_test = ImageFolder(val_data, transforms[evaluation_path], return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms["eval"],
                                   return_paths=True)
    eval_folder_test_s = ImageFolder(os.path.join(source_path),
                                   transform=transforms["eval"],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,            
            )
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,
            )

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        # persistent_workers=True,
        pin_memory=True,
        )
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        # persistent_workers=True,
        pin_memory=True,
        )
    test_loader_s = torch.utils.data.DataLoader(
        eval_folder_test_s,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        # persistent_workers=True,
        pin_memory=True,
        )
    print('val', val)
    if val:
        test_loader_source = torch.utils.data.DataLoader(
            source_val_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,
            )
        return source_loader, target_loader, test_loader, test_loader_source, test_loader_s

    return source_loader, target_loader, test_loader, target_folder_train, test_loader_s


def get_loader_ad(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False, val=False, val_data=None):
    
    train_transform = BYOLDataTransform(
        crop_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        blur_prob=[.0, .0],
        solarize_prob=[.0, .2],
        GaussianBlur=1,
        )

    ori_trans = transform_lib.Compose([
        transform_lib.Resize((256, 256)),
        transform_lib.RandomHorizontalFlip(),
        transform_lib.RandomCrop(224),
        transform_lib.ToTensor(),
        transform_lib.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_transform.transforms.append(ori_trans)

    source_folder = ImageFolder(os.path.join(source_path),
                                train_transform,
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                  transform = train_transform,
                                  return_paths = False, return_id=return_id)
    if val:
        source_val_train = ImageFolder(val_data, train_transform, return_id=return_id)
        target_folder_train = torch.utils.data.ConcatDataset([target_folder_train, source_val_train])
        source_val_test = ImageFolder(val_data, transforms[evaluation_path], return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms["eval"],
                                   return_paths=True)

    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=6,
            # persistent_workers=True,
            pin_memory=True,            
            )
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
            # persistent_workers=True,
            pin_memory=True,
            )

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        # persistent_workers=True,
        pin_memory=True,
        )
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        # persistent_workers=True,
        pin_memory=True,
        )
    if val:
        test_loader_source = torch.utils.data.DataLoader(
            source_val_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=6,
            # persistent_workers=True,
            pin_memory=True,
            )
        return source_loader, target_loader, test_loader, test_loader_source

    return source_loader, target_loader, test_loader, target_folder_train, source_folder


def get_loader_label(source_path, target_path, target_path_label, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    target_folder_label = ImageFolder(os.path.join(target_path_label),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    target_loader_label = torch.utils.data.DataLoader(
        target_folder_label,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, target_loader_label, test_loader, target_folder_train



