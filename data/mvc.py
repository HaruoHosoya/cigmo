#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:21:53 2021

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

Based on torchvision.datasets.caltech
@author: hahosoya
"""

from PIL import Image
import os
import os.path
from typing import Any, Callable, List, Optional, Union, Tuple
import json
import scipy.io
import urllib
import http
import time

from torchvision.datasets.vision import VisionDataset



class MVC(VisionDataset):
    """`MVC <https://github.com/MVC-Datasets/MVC>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``MVC`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = 'MVC'
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(MVC, self).__init__(os.path.join(root, self.base_folder),
                                  transform=transform,
                                  target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' First manually download *.json files and use download=True to download images')

        with open(os.path.join(self.root, 'attribute_labels.json'), 'r') as fin:
            self.attrs = json.load(fin)
            
        self.index: List[int] = list(range(len(self.attrs)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) 
        """

        attr = self.attrs[index]
        img = Image.open(os.path.join(self.root,
                                      'image_data',
                                      attr['filename'] + '.jpg'))

        target = attr['itemN']

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_attr(self, index):
        (_, _, attr) = self.attr[index]
        attr.pop('filename')
        attr.pop('itemN')
        return attr

    def _check_integrity(self) -> bool:
        return (os.path.exists(os.path.join(self.root, 'image_links.json')) and
                os.path.exists(os.path.join(self.root, 'attribute_labels.json')) and
                os.path.exists(os.path.join(self.root, 'mvc_info.json')) and
                os.path.exists(os.path.join(self.root, 'image_data')))
    
    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        
        with open(os.path.join(self.root, 'image_links.json'), 'r') as fin:
            links = json.load(fin)

        with open(os.path.join(self.root, 'attribute_labels.json'), 'rb') as fin:
            attrs = json.load(fin)
            
        os.makedirs(os.path.join(self.root, 'image_data'), exist_ok=True)

        for i in range(len(links)):
            link = links[i]['image_url_4x']
            fname = attrs[i]['filename']
            filepath = os.path.join(self.root, 'image_data', fname + '.jpg')
            
            if os.path.exists(filepath):
                continue
            
            print('downloading', link, 'into', fname)
            
            headers = { 'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0' } 
            request = urllib.request.Request(link, headers=headers) 
            while True:
                try:
                    data = urllib.request.urlopen(request).read()
                    break
                except urllib.error.HTTPError as e:
                    print('Error:', str(e))
                    print('Retry after 5 secs...')
                    time.sleep(5)
                except http.client.IncompleteRead as e:
                    print('Error:', str(e))
                    print('Retry after 5 secs...')
                    time.sleep(5)
                
            with open(filepath, mode='wb') as f:
                f.write(data)
            

