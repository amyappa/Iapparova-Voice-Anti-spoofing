# Audio Deepfake Detection with FFT-LCNN

LCNN-based audio anti-spoofing system trained on the **ASVspoof 2019 Logical Access** dataset. The model classifies recordings as **bona fide** or **spoof**.

The input consists of log-power spectrograms of size `1 × 863 × 600`. The final model was trained with FFT size `1724`, hop length `130`, Adam (`lr=3e-4`), batch size `8`, for `15` epochs with seed `100`.

The work was completed as part of an educational deep learning internship.

## Results

| Split | EER | Accuracy |
|---|---:|---:|
| Development | 0.0398% | 99.96% |
| Evaluation | 8.4051% | 85.54% |

## Acknowledgements
This repository is based on a [project template](https://github.com/Blinorot/pytorch_project_template) provided by the course instructor.

## References

- [A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection](https://arxiv.org/abs/2103.11326)
- [STC Antispoofing Systems for the ASVspoof2019 Challenge](https://arxiv.org/abs/1904.05576)

## Author

Amina Yapparova, HSE University
