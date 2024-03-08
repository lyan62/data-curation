# Data-curation
codebase for The Role of Data Curation in Image Captioning

## Data curation with BEiT-3

[(BEiT-3) Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)

### Pretrained models
We used the BEiT3-base model from the original unilm repo. We used the base size checkpoint---`BEiT3-base`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #parameters: 276M => [download checkpoint](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)

### Setup
Set up accordingly as in 
https://github.com/microsoft/unilm/tree/master/beit3

### Finetuning on COCO and Flickr30k with dynamic data curation
Following instructions at [`get_started_for_image_captioning.md`](get_started/get_started_for_captioning.md) for downloading datasets and preprocessing. 


## Citation
```
@misc{li2024role,
      title={The Role of Data Curation in Image Captioning}, 
      author={Wenyan Li and Jonas F. Lotz and Chen Qiu and Desmond Elliott},
      year={2024},
      eprint={2305.03610},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgement

This repository is built using the [BEiT3](https://github.com/microsoft/unilm/tree/master/beit3) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using BEiT-3 models, please submit a GitHub issue.
