
Run Gaussian Splatting workflow on folders automatically.

## What is this?

The script watches a .yml file for jobs.

Each job is the name of a sub-folder in a source directory.

If the sub-folder exists, contains images and no .ply file, the GS workflow is started:

The images are copied into a processing directory.

A Gaussian Splatting workflow (GLOMAP/COLMAP and Brush) is run on these images.

The output (.ply) is copied back into the image source directory.

## Requirements

COLMAP and GLOMAP, Brush

## How to run

Make a virtual environment.

Install dependencies:

`pip install -r requirements.txt`

Create a `queue.yml` file in the directory with image sub-folders (e.g. `/tmp/GSjobs`):

```{.yaml}
config:
  brush: "~/Downloads/brush-app-x86_64-unknown-linux-gnu"
  processing_dir: "/tmp/splatWatch"
queue:
 - folder: foo
 - folder: bar
 - folder: baz
```

Run this script and specify the image source directory:

```{.sh}
python3 main.py /tmp/GSjobs
```


