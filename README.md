### Spatial-Separated Curve Rendering Network for Efficient and High-Resolution Image Harmonization

<b>Jingtang Liang*, <a href='vinthony.github.io'>Xiaodong Cun*</a> and Chi-Man Pun</b>

Paper: https://arxiv.org/pdf/2109.04760.pdf

<i>Image harmonization aims to modify the color of the composited region with respect to the specific background. Previous works model this task as a pixel-wise image-to-image translation using UNet family structures. However, the model size and computational cost limit the performability of their models on edge devices and higher-resolution images. To this end, we propose a novel spatial-separated curve rendering network(S<sup>2</sup>CRNet) for efficient and high-resolution image harmonization for the first time. In S<sup>2</sup>CRNet, we firstly extract the spatial-separated embeddings from the thumbnails of the masked foreground and background individually. Then, we design a curve rendering module(CRM), which learns and combines the spatial-specific knowledge using linear layers to generate the parameters of the pixel-wise curve mapping in the foreground region. Finally, we directly render the original high-resolution images using the learned color curve. Besides, we also make two extensions of the proposed framework via the Cascaded-CRM and Semantic-CRM for cascaded refinement and semantic guidance, respectively. Experiments show that the proposed method reduces more than 90% parameters compared with previous methods but still achieves the state-of-the-art performance on both synthesized iHarmony4 and real-world DIH test set. Moreover, our method can work smoothly on higher resolution images in real-time which is more than 10× faster than the existing methods. </i> 

> The source codes and pre-trained models will be released soon.


##### Citation
```
@misc{liang2021spatialseparated,
      title={Spatial-Separated Curve Rendering Network for Efficient and High-Resolution Image Harmonization}, 
      author={Jingtang Liang and Xiaodong Cun and Chi-Man Pun},
      year={2021},
      eprint={2109.05750},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


##### Related Work

* [TIP 2020] [S<sup>2</sup>AM: Improving the Harmony of the Composite Image by Spatial-Separated Attention Module.](https://github.com/vinthony/s2am)
* [AAAI 2021] [Split then Refine: Sequential Attention-guided ResUNets for Blind Single Image Visible Watermark Removal](https://github.com/vinthony/deep-blind-watermark-removal)
