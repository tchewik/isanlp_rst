![Python](https://img.shields.io/badge/python-3.10%2B-blue) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dFItacO_fiOczrVno-hXEn2HTTN6ldH5?usp=sharing)


# IsaNLP RST Parser

This library provides several versions of the Rhetorical Structure (RST) parser for multiple languages. Below, you will find instructions on how to set up and run the parser either locally or using Docker.

## Performance

The parser supports multiple languages and corpora. The end-to-end performance metrics for different model versions across corpora are as follows:

### Tags

Supported languages (all): English (eng), Czech (ces), German (deu), Basque (eus), Persian (fas), French (fra), Dutch (nld), Brazilian Portuguese (por), Russian (rus), Spanish (spa), and Chinese (zho).

| Tag / Version | Languages   | Train Data          | Test Data       | Seg  | S    | N    | R    | Full  |
|-------------- |------------ |---------------------|-----------------|------|------|------|------|-------|
| `rstdt`       | eng         | eng.rst.rstdt       | eng.rst.rstdt       | 97.8 | 75.6 | 65.0 | 55.6 | 53.9  |
| `gumrrg`      | eng, rus    | eng.erst.gum, rus.rst.rrg    | eng.erst.gum        | 95.5 | 67.4 | 56.2 | 49.6 | 48.7  |
|               |             |                     | rus.rst.rrg         | 97.0 | 67.1 | 54.6 | 46.5 | 45.4  |
| `rstreebank`  | rus         | rus.rrt             | rus.rst.rrt         | 92.1 | 66.2 | 53.1 | 46.1 | 46.2  |
| `unirst`      | all         | all                 | ces.rst.crdt     | 94.5 | 59.1 | 41.2 | 28.6 | 28.0 |
|               |             |                     | deu.rst.pcc      | 96.5 | 67.3 | 47.4 | 34.1 | 32.1 |
|               |             |                     | eng.erst.gum     | 95.3 | 67.3 | 55.6 | 48.5 | 47.4 |
|               |             |                     | eng.rst.oll      | 92.5 | 55.7 | 39.0 | 27.5 | 26.3 |
|               |             |                     | eng.rst.rstdt    | 98.1 | 76.7 | 65.5 | 55.2 | 53.6 |
|               |             |                     | eng.rst.sts      | 91.2 | 43.3 | 31.3 | 19.4 | 18.7 |
|               |             |                     | eng.rst.umuc     | 88.8 | 52.6 | 40.6 | 26.2 | 25.8 |
|               |             |                     | eus.rst.ert      | 92.5 | 66.0 | 50.3 | 34.9 | 34.7 |
|               |             |                     | fas.rst.prstc    | 94.7 | 63.0 | 50.2 | 40.8 | 40.7 |
|               |             |                     | fra.sdrt.annodis | 91.3 | 58.6 | 48.9 | 30.6 | 30.3 |
|               |             |                     | nld.rst.nldt     | 98.0 | 61.8 | 49.8 | 36.8 | 35.8 |
|               |             |                     | por.rst.cstn     | 93.9 | 68.4 | 52.8 | 44.9 | 44.5 |
|               |             |                     | rus.rst.rrg      | 96.4 | 67.4 | 54.0 | 46.3 | 45.1 |
|               |             |                     | rus.rst.rrt      | 90.7 | 63.0 | 49.0 | 42.3 | 42.2 |
|               |             |                     | spa.rst.rststb   | 93.4 | 63.5 | 50.3 | 36.0 | 36.0 |
|               |             |                     | spa.rst.sctb     | 85.5 | 55.1 | 46.8 | 39.1 | 39.1 |
|               |             |                     | zho.rst.gcdt     | 93.0 | 64.5 | 50.7 | 45.9 | 44.6 |
|               |             |                     | zho.rst.sctb     | 95.4 | 67.5 | 51.5 | 39.9 | 39.9 |


## Local Setup

To use the IsaNLP RST Parser locally, follow these steps:

1. **Installation:**

   First, install the `isanlp` and `isanlp_rst` libraries using pip:

   ```bash
   pip install git+https://github.com/iinemo/isanlp.git
   pip install isanlp_rst
   ```

2. **Usage:**

    Below is an example of how to run a specific version of the parser using the library:

   ```python
   from isanlp_rst.parser import Parser

   # Define the version of the model you want to use
   version = 'gumrrg'  # Choose from {'gumrrg', 'rstdt', 'rstreebank'}
   
   # Initialize the parser with the desired version
   parser = Parser(hf_model_name='tchewik/isanlp_rst_v3', hf_model_version=version, cuda_device=0)

   # Example text for parsing
   text = """
   On Saturday, in the ninth edition of the T20 Men's Cricket World Cup, Team India won against South Africa by seven runs. 
   The final match was played at the Kensington Oval Stadium in Barbados. This marks India's second win in the T20 World Cup, 
   which was co-hosted by the West Indies and the USA between June 2 and June 29.

   After winning the toss, India decided to bat first and scored 176 runs for the loss of seven wickets. 
   Virat Kohli top-scored with 76 runs, followed by Axar Patel with 47 runs. Hardik Pandya took three wickets, 
   and Jasprit Bumrah took two wickets.
   """

   # Parse the text to obtain the RST tree
   res = parser(text)  # res['rst'] contains the binary discourse tree

   # Display the structure of the RST tree
   vars(res['rst'][0])
   ```

   To use the multilingual UniRST model, you can specify the required relation inventory with `relinventory='lang.code.dataset'`, as listed in the performance table. The default inventory for UniRST is `eng.rst.rstdt`. 
   
   ```python
   parser = Parser(hf_model_name='tchewik/isanlp_rst_v3',
                   hf_model_version='unirst',
                   cuda_device=0,
                   relinventory='eng.erst.gum')
   ```
   
   The output is an RST tree with the following structure:

   ```python
   {'id': 21,
    'left': (id=14, start=1, end=323),
    'right': (id=20, start=324, end=570),
    'relation': 'elaboration',
    'nuclearity': 'NS',
    'proba': 1.0,
    'start': 1,
    'end': 570,
    'text': "On Saturday, ... took two wickets.",
   }
   ```

   - **id**: Unique identifier for the discourse unit.
   - **left** and **right**: Children of the current discourse unit.
   - **relation**: Rhetorical relation between sub-units (e.g., "elaboration").
   - **nuclearity**: Indicates nuclearity of the relation (e.g., "NS" for nucleus-satellite).
   - **start** and **end**: Character offsets in the text for this discourse unit.
   - **text**: Text span corresponding to this discourse unit.

4. **(Optional) Save the result in RS3 format:**

   You can save the resulting RST tree in an RS3 file using the following command:

   ```python
   res['rst'][0].to_rs3('filename.rs3')
   ```

   The `filename.rs3` file can be opened in RSTTool or rstWeb for visualization or editing.
   <img src="examples/example-image.png" alt="Illustration of En parsing" width="600">


## (Optional) Visualize the output inline or export in PNG/PDF

You can preview the RST tree right inside a Jupyter notebook.

```python
import io, contextlib
import isanlp_rst

# Suppress the HTML string from being printed to stdout —
# the widget will render inline instead.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    isanlp_rst.render("filename.rs3")
```

If you’re in **Google Colab**, pass `colab=True` to keep the output cell height in sync with the rendered tree:

```python
isanlp_rst.render("output.rs3", colab=True)
```

<img src="examples/example-inline.png" alt="Illustration of the parsing visualization" width="600">

You can also export the `.rs3` file to **PNG** or **PDF**:

```python
import isanlp_rst

# PNG
isanlp_rst.to_png("filename.rs3", "filename.png")

# PDF
isanlp_rst.to_pdf("filename.rs3", "filename.pdf")
```

> Tip: If you plan to use PNG/PDF export, make sure Playwright is installed:
>
> ```bash
> pip install playwright
> playwright install chromium
> ```


## (Optional) Docker Setup

For now, Docker container is available for tags: `rstdt`, `gumrrg`, `rstreebank`. 

To run the IsaNLP RST Parser using Docker, follow these steps:

1. **Run the Docker container:**

   Pull and run the Docker container with the desired model version tag:

   ```bash
   docker run --rm -p 3335:3333 --name rst_rrt tchewik/isanlp_rst:3.0-rstreebank
   ```

2. **Connect using the IsaNLP Python library:**

   Install the `isanlp` library. The `isanlp_rst` library is not required for dockerized parsers:

   ```bash
   pip install git+https://github.com/iinemo/isanlp.git
   ```

   Then connect to the running Docker container:

   ```python
   from isanlp import PipelineCommon
   from isanlp.processor_remote import ProcessorRemote

   # Put the container address here
   address_rst = ('127.0.0.1', 3335)

   ppl = PipelineCommon([
       (ProcessorRemote(address_rst[0], address_rst[1], 'default'),
        ['text'],
        {'rst': 'rst'})
   ])

   res = ppl(text)
   # res['rst'] will contain the binary discourse tree, similar to the previous example
   ```


   
## Citation

If you use the IsaNLP RST Parser in your research, please cite our work as follows:

  ```bibtex
  @inproceedings{
   chistova-2024-bilingual,
   title = "Bilingual Rhetorical Structure Parsing with Large Parallel Annotations",
   author = "Chistova, Elena",
   booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
   month = aug,
   year = "2024",
   address = "Bangkok, Thailand and virtual meeting",
   publisher = "Association for Computational Linguistics",
   url = "https://aclanthology.org/2024.findings-acl.577",
   pages = "9689--9706"
  }
  ```
