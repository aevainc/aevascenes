<div align="center">
  <img src="media/aeva_logo.png" alt="Aeva Logo" width="120" height="120">
  <h1>AevaScenes Python SDK</h1>
  <p><em>A Dataset and Benchmark for FMCW LiDAR Perception</em></p>
  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/aevascenes-v0.1-green?style=flat-square" alt="AevaScenes Versions">
    <img src="https://img.shields.io/badge/python->=3.10-green?style=flat-square" alt="Python Versions">
    <img src="https://img.shields.io/github/license/aevainc/aevascenes?style=flat-square&color=orange" alt="License">
    <img src="https://img.shields.io/badge/license-AevaScenes%20Dataset%20License-orange?style=flat-square" alt="License">
  </p>
  <!-- Navigation Links -->
  <p>
    <a href="https://scenes.aeva.com/"><strong>Website</strong></a> |
    <a href="https://scenes.aeva.com/dataset"><strong>Dataset</strong></a> |
    <a href="https://scenes.aeva.com/download"><strong>Download</strong></a> |
    <a href="#license"><strong>License</strong></a> |
    <a href="#citation"><strong>Citation</strong></a>
  </p>
</div>

<!-- ## Overview -->

**AevaScenes** is a comprehensive multi-modal dataset designed to advance research in FMCW (Frequency-Modulated Continuous Wave) LiDAR perception. The dataset features synchronized data from **6 FMCW LiDARs** and **6 high-resolution cameras** mounted on a vehicle, captured across diverse urban and highway environments in the San Francisco Bay Area.

#### Instantaneous Velocity Measurements
AevaScenes features the longest-range FMCW LiDAR data ever released to the public, delivering over 400 meters of range with per-point velocity—enabling researchers and developers to explore perception capabilities beyond the limits of traditional datasets.

<p align="center">
  <img src="media/velocity.gif" width="600">
</p>


#### Ultra-Long Range Detections
For the first time, FMCW LiDAR enables detections up to **400** meters by measuring instantaneous radial velocity per point directly at the sensor. Aeva's Doppler-based approach enhances long-range perception, enabling earlier detection of moving objects and improving overall safety for autonomous navigation.

<p align="center">
  <img src="media/long_range.gif" width="600">
</p>


#### High Fidelity Perception Labels
AevaScenes provides annotations—including 3D bounding boxes, lane lines, and semantic segmentation—at ranges up to 400 meters. This unprecedented depth of annotation empowers research in long-range perception, planning, and tracking beyond the limits of existing datasets.

<p align="center">
  <img src="media/semseg.gif" width="600">
</p>


#### High Dynamic Range Reflectivity
Unlike traditional LiDARs, FMCW LiDAR delivers high dynamic range with negligible blooming around retroreflective surfaces such as road signs, botts-dots and license plates, resulting in sharper object boundaries and more accurate perception, even in challenging high-reflectivity scenarios.

<p align="center">
  <img src="media/reflectivity.gif" width="600">
</p>

#### Interactive Web Visualizer

A small subset of the highway/city/day/night sequences are available to see using the web visualizer here
**[AevaScenes Web Visualizer](https://scenes.aeva.com/visualize)**.

<p align="center">
  <img src="media/web_visualizer.gif" width="600">
</p>


## Download Dataset
1. Visit [scenes.aeva.com/download](https://scenes.aeva.com/download).
2. Register and agree to the license terms.
3. We will email you the signed_urls.txt file which contains the download links.
4. Download the dataset using the provided download script and signed_urls.txt:

```bash
# Download the dataset (after signing up and obtaining access)
mkdir -p data/aevascenes_v2
bash scripts/download_dataset.sh --url-file signed_urls_train.txt --output data/aevascenes_v2
bash scripts/download_dataset.sh --url-file signed_urls_validation.txt --output data/aevascenes_v2
bash scripts/download_dataset.sh --url-file signed_urls_test.txt --output data/aevascenes_v2

# Extract the dataset
cd data/aevascenes_v2/   # split = train | validation | test
for split in train validation test; do [ -d "$split" ] && for f in "$split"/*.tar.gz; do tar -xzf "$f" -C "$split"; done; done
```

## Getting Started

Please see [Dataset.md](./docs/Dataset.md) for details about the dataset and schema.

Please see [Getting Started.md](./docs/Getting_Started.md) to get started with using the Python SDK. Here's a quick overview.

```bash
# List available sequences
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2 --list-sequences

# Visualize both raw and compensated point clouds
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2 --sequence-uuid <UUID> --pcd-type raw_and_compensated

# Visualize a single sequence (defaults: compensated clouds, vehicle frame, velocity coloring)
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2 --sequence-uuid <UUID>

# Project LiDAR points onto camera images
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2 --sequence-uuid <UUID> --color-mode semantic --project-points

# Raw point clouds in world frame, LiDAR-only
python examples/visualize_aevascenes.py --dataroot data/aevascenes_v2 --sequence-uuid <UUID> --pcd-type raw --coordinate-frame world --no-images
```

Options: `--color-mode` (`velocity` | `reflectivity` | `semantic`), `--pcd-type` (`compensated` | `raw` | `raw_and_compensated`), `--coordinate-frame` (`vehicle` | `world`), `--project-points`, `--no-images`, `--image-downsample-factor` (`1` | `2` | `4` | `8`), `--no-keep-alive`, `--list-sequences`.



## License

The AevaScenes dataset is provided under the **AevaScenes Dataset License Agreement** for non-commercial use only. The AevaScenes Python SDK is licensed under the MIT License.

- **Dataset License**: [AevaScenes Dataset License](https://scenes.aeva.com/license)
- **Code License**: [MIT License](LICENSE-CODE)


## Citation

If you use AevaScenes in your research, please cite our work using the following BibTeX.

```bibtex
@misc{aevascenes,
  title        = {AevaScenes: A Dataset and Benchmark for FMCW LiDAR Perception},
  author       = {Narasimhan, Gautham Narayan and Vhavle, Heethesh and Vishvanatha, Kumar Bhargav and Reuther, James},
  year         = {2025},
  url          = {https://scenes.aeva.com/},
}
```

## Contributing

We welcome contributions to improve the AevaScenes dataset and toolkit! Please see our contributing guidelines and submit pull requests for:

- Requests for data diversity.
- Bug fixes and performance improvements.
- New visualization features.
- Dataset utilities and analysis tools.

## Support

- **Website**: [scenes.aeva.com](https://scenes.aeva.com/)
- **Issues**: [GitHub Issues](https://github.com/aevainc/aevascenes/issues)
- **Email**: [research@aeva.ai](mailto:research@aeva.ai)
