# PV-S3
<p align="center">

  <h2 align="center">PV-S3: Advancing Automatic Photovoltaic Defect Detection using Semi-Supervised Semantic Segmentation of Electroluminescence Images</h2>
  <p align="center">
    <a href="https://abj247.github.io/"><strong>Abhishek Jha</strong></a><sup>1</sup>
    ·
    <a href="https://www.crcv.ucf.edu/person/rawat/"><strong>Yogesh Rawat</strong></a><sup>2</sup>
    ·
    <a href="https://ai.ucf.edu/person/shruti-vyas/"><strong>Shruti Vyas</strong></a><sup>2</sup>
    
</p>

<p align="center">
    <sup>1</sup> Delhi Technological University · <sup>2</sup>University of Central Florida
</p>
   <h3 align="center">

   [![arXiv](https://img.shields.io/badge/arXiv-2404.13693-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2404.13693/)
  <div align="center"></div>
</p>

## Abstract:

Photovoltaic (PV) systems allow us to tap into all abundant solar energy, however they require regular maintenance for high efficiency and to prevent degradation. Traditional manual health check, using Electroluminescence (EL) imaging, is expensive and logistically challenging which makes automated defect detection essential. Current automation approaches require extensive manual expert labeling, which is time-consuming, expensive, and prone to errors. We propose PV-S3 (Photovoltaic-Semi Supervised Segmentation), a Semi-Supervised Learning approach for semantic segmentation of defects in EL images that reduces reliance on extensive labeling. PV-S3 is a Deep learning model trained using a few labeled images along with numerous unlabeled images. We evaluate PV-S3 on multiple datasets and demonstrate its effectiveness and adaptability. With merely 20% labeled samples, we achieve an absolute improvement of 9.7% in IoU, 13.5% in Precision, 29.15% in Recall, and 20.42% in F1-Score over prior state-of-the-art supervised method (which uses 100% labeled samples) on UCF-EL dataset (largest dataset available for semantic segmentation of EL images)showing improvement in performance while reducing the annotation costs by 80%.
