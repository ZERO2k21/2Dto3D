# 2D Image to 3D Mesh Converter

Hey there! This is a simple yet powerful Python script that transforms your ordinary 2D images into detailed 3D meshes. It leverages the impressive MiDaS model for depth estimation and Open3D for generating the final mesh. All wrapped in a friendly interface that even asks you nicely to pick your files.

## Quick Start

### Step 1: Download the project files or just clone the Repo


### Step 2: Set Up the Environment

It's always good to keep things tidy. Let's create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### Step 3: Install Dependencies

Run this command to install all the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

### Option A (Command Line Arguments)

```bash
python main.py -i path/to/your/image.jpg -o path/to/save/mesh.ply
```

### Option B (Interactive Mode)

Just run without arguments, and you'll get friendly prompts:

```bash
python main.py
```

- It'll ask you to pick an image.
- Then, it'll ask where you want to save the generated mesh.

## Viewing Your Mesh

Once done, the script will automatically pop up an Open3D window to show your new 3D creation. You can rotate and zoom in to explore it.

## What’s Under the Hood?

- Depth Estimation: MiDaS (DPT Large model)
- 3D Processing: Open3D for point cloud creation, mesh reconstruction, and visualization
- Friendly GUI: Tkinter for easy file selection

## Tips for Best Results

- High-quality images with good lighting and contrast yield the best meshes.
- Images of objects or scenes with clear foreground-background separation typically convert well.