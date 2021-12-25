# RST Parsing from Scratch
This repository contains the source code of our paper "RST Parsing from Scratch" in NAACL 2021.
## Requirements
* `python`: 3.7
* `pytorch`: 1.4
* `transformers`: 3.0

## Usage

To train a discourse parser:

    ./*_train.sh
    
To predict discourse tree:
 
    ./*_predict.sh

## Data Format

* For end-to-end parsing from scratch (no sentence guidance): 
we need to create the data with dummy edu_break and doc_structure. Refer to `create_sample_dummy_format_data.py` and `dummy_format_data/sample_rawtext_data_format`
* For other parsing models:
Refer to `create_sample_dummy_format_data.py` and `dummy_format_data/sample_full_data_format`
  
## Citation
Please cite our paper if you found the resources in this repository useful.

    @inproceedings{nguyen-etal-2021-rst-scratch,
    title = "RST Parsing from Scratch",
    author = "Nguyen, Thanh-Tung  and
      Nguyen, Xuan-Phi  and
      Joty, Shafiq  and
      Li, Xiaoli",
    booktitle = "Proceedings of the 2021 Conference of the North {A}merican Chapter 
    of the Association for Computational Linguistics: Human Language Technologies, 
    Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = "xx--xx",}
    }	