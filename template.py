import os
from pathlib import Path

package_name = "irisclassification"

list_of_files = [
    ".github/workflows/.gitkeep", # for ensuring we can push empty folder
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/components/data_ingestion.py",
    f"src/{package_name}/components/data_transformation.py",
    f"src/{package_name}/components/model_trainer.py",
    f"src/{package_name}/pipelines/__init__.py",
    f"src/{package_name}/pipelines/training_pipeline.py",
    f"src/{package_name}/pipelines/prediction_pipeline.py",
    f"src/{package_name}/logger.py",
    f"src/{package_name}/exception.py",
    f"src/{package_name}/utils/__init__.py",
    "notebooks/research.ipynb",
    "notebooks/data/.gitkeep",
    "requirements.txt",
    "setup.py",
    "init_setup.sh"
]


# Here we will create a directory

for filepath in list_of_files:
    filepath = Path(filepath) # it will generate system compatible path
    filedir, filename = os.path.split(filepath)
    
    """
    how exist_ok works: if "directory" already exists,
    os.makedirs() will not raise an error, and it will do nothing.
    If "my_directory" doesnt exist, it will create the directory
    """
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True) # if "directory" already exists, os.makedirs() will not raise an error, and it will do nothing.
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0): # file is not present or file size is 0
        with open(filepath, "w") as f:
            pass
    else:
        print("file already exists")    
            