# IDR Evolution Manuscript

This is the repo for the IDR Evolution manuscript. It contains the formatting and text of the manuscript in the form of LaTeX files as well as the Python code used to generate the figures. To keep the presentation of data separate from their generation, this project was spun off from the main [IDR Evolution](https://github.com/marcsingleton/IDR_evolution2023) repo. The two, however, are closely linked, and updates to the organization and content of the main repo are ultimately reflected here as well. Accordingly, the figure generation code expects links to certain portions of the main repo, which is described in more detail in the Project Organization section.

This manuscript is now published and available in its final form [here](https://doi.org/10.1371/journal.pcbi.1012028). Note, as the published manuscript underwent some light copy editing, its text may differ slightly from the version archived here.

## Project Organization

As this is a small repo, its organization is largely self-explanatory. To give a brief overview, however, the LaTeX files are stored in the project root, where `main.tex` controls the overall formatting and organization of the manuscript, and the other files contain the text itself. The code for generating each figure is stored in a subdirectory under the `figures/` directory. To access utility functions and data from the main repo, these scripts expect links in the root of this repo to a local copy of the main repo as well as to the main repo's `src/` directory. (The directory of this repo should also be in your environment's PYTHONPATH to allow Python to import this `src/` "directory" as a module in the figure generation scripts.) In other words, the directory structure of this project should look like:

```
IDR_evolution_manuscript/
	├── figures/
	├── IDR_evolution/   (link to /local/path/to/IDR_evolution/)
	├── src/             (link to /local/path/to/IDR_evolution/src/)
	└── ...              (other files omitted for brevity)
```
