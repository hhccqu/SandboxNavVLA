# SandboxNav-VLA

**Release preparation draft — not yet public.** This repository is being assembled from the research code used for the paper. The source files under `src/` are copied from the authors' experiment tree; paths, dependencies, checkpoints, and hardware interfaces still need a clean-install reproduction pass before a public release.

SandboxNav-VLA adapts a SimLingo vision-language driving backbone for semantic navigation and closed-loop control on a physical miniature-vehicle sandbox. It adds history-conditioned trajectory prediction and interfaces the predicted local path with vehicle control. See the [project site](docs/index.html) draft and [evaluation notes](evaluation/PAPER_TABLE_DATA.md).

## Repository status

- `src/`: model, dataset, coordinate conversion, collation, training and open-loop evaluation modules.
- `configs/`: intended home for the minimal public training and evaluation configs; not packaged yet.
- `evaluation/`: paper table data and, when cleared, reproducible result artifacts.
- `docs/`: static GitHub Pages site draft.
- `hardware/`: intended home for public interface descriptions and example configurations; hardware-control release needs an additional security and reproducibility review.

## Before reproducing

This draft does not yet include the full training corpus, model checkpoints, credentials, or a verified install environment. Do not interpret copied scripts as a turnkey release. Follow the upstream SimLingo setup and license; this project does not redistribute its pretrained weights.

## Attribution and license

The project is built on the publicly released [SimLingo](https://github.com/RenzKa/simlingo) code and pretrained model. Preserve the upstream license and notices for every retained upstream-derived component. Project-specific code needs an author-approved license before publication. Third-party code and model weights have separate terms.

## Media and privacy

The website embeds an author-provided [project demonstration video](https://www.youtube.com/watch?v=n9HuocT_FBs). When adding further recordings, retain complete clips for trials represented in aggregate metrics and review faces, voices, school/lab signs, account names, and network/device details. Do not upload raw datasets or logs until consent, ownership, and privacy have been checked.

## Citation

Citation metadata will be added after the author list and public paper version are finalized.
