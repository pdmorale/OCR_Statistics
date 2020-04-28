# OCR Statistics tool

A repository that fascilitates the extraction of statistics from a given OCR engines. This is a UiPath Studio and Python project which targets the Studio built-in engines, however, feel free any custom engine by modifying the powershell script. At its core, this workflow generates synthetic data according to a given set of parameters and evaluates OCR engine resilience to the effects by computing the similarity of the outcome and what it should read through its [Levishtein measure](https://en.wikipedia.org/wiki/Levenshtein_distance). And ultimately allows you to a graph the resilience to a given feature.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
UiPath Studio
matplotlib
beautifulsoup4==4.6.0
numpy==1.15.1
opencv-python==4.0.0.21
tqdm==4.23.4
Pillow==5.1.0
requests==2.20.0
```

The text-image generation part was largely based and inspired on Belval's [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) project, so please make sure to check that project out as well

## Deployment

To run it on another windows machine you can pack the python runtime and enviroment with the requirements especified at [requirements.txt](requirements.txt) with

* [Conda-Pack](https://conda.github.io/conda-pack/)
* [Embedded Python](https://docs.python.org/3/extending/embedding.html)

## Built With

* [UiPath Studio](https://www.uipath.com/) - 

## Authors

* **Pablo Morales** - *Initial work* - [pdmorale](https://github.com/pdmorale)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
