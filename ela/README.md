ela is a Python package for reproducable data-augmentations. In difference to most other libraries random parameters for transformations are drawn seperate from the transformations. This makes it very easy apply the same transformations to several images. An example where this behaviour is useful is semantic segmentation, when you need to modify the input and the mask in the same way.


This library is currently in it's early stages so interfaces may break quite often. Python2 support is planed, but currently only Python3.6 is developed.

##Installation
1. Clone this repository
2. Run 
``` pip install folderyouclonedelainto/ela ```