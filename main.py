import time
import sys
import yaml
import os
from shutil import rmtree, copy
import atexit
import subprocess

class SplatWatch():
    '''
    Checks for sub-folders (from queue.yml in source dir) with images in a source directory (specified as sys.arg to this script).

    Creates a processing folder (specified in yaml).
    Copies images into the processing folder and runs Colmap/Glomap and Brush (Gaussian Splatting).

    Output from Brush (.ply files) is copied back into source images folder.

    A folder is considered processed, if it contains a .ply file.

    '''
    def __init__(self, path_to_source_images):
        self.source_images=path_to_source_images
        self.processing_dir=""
        self.path_to_brush_app=""
        self.sub_p = None  # the subprocess
        atexit.register(self.__cleanup)

    def run(self):
        done = False
        while not done:
            print("\n\n Reading Queue")
            self.process()
            time.sleep(5)

    def process(self):
        ''' Go through each job in the queue and run the first job that has not been processed'''

        valid_path = os.path.isdir(self.source_images)
        if not valid_path:
            print("Source path invalid:", self.source_images)
            return False

        data = None
        try:
            with open(os.path.join(self.source_images, 'queue.yml'), 'r') as file:
                data = yaml.safe_load(file)
                file.close()
        except OSError:
            print("Could not open/read file:", self.source_images)
            return
        
        if not "config" in data:
            print("No config in yaml")
            return False

        if not "brush" in data["config"]:
            print("Path to Brush app not set")
            return False

        if not "processing_dir" in data["config"]:
            print("Path to Processing Dir not set")
            return False

        if not "queue" in data:
            print("Dict key Queue not found")
            return False 

        # Settings from config
        config = data["config"]
        self.path_to_brush_app = os.path.abspath(config["brush"])
        self.processing_dir = os.path.abspath(config["processing_dir"])

        print(self.path_to_brush_app)
        print(self.processing_dir)

        # Go through each "job" and do sanity checks
        for el in data["queue"]:
            print("\n-----------")
            if not "folder" in el:
                print("Entry has no folder, skipping")
                continue

            job_source_images = os.path.join(self.source_images, el["folder"])
            valid_source_path = os.path.isdir(job_source_images)

            # Folder exists?
            if not valid_source_path:
                print("Source image path does not exist:", job_source_images)
                continue

            # Folder has list of images?
            has_images = self.list_images(job_source_images)
            if not has_images:
                print("Folder has no images:", job_source_images)
                continue

            # Check this jobs folder, if done already
            has_ply = self.list_ply(job_source_images)
            if has_ply:
                print("Folder already done:", job_source_images)
                continue
            
            # Folder valid and not processed 
            print("Processing Folder", job_source_images)

            # Create a new dir in processing_dir and /images
            # copy images over
            job_processing_dir = os.path.join(self.processing_dir, el["folder"])

            # Delete existing processing dir
            if os.path.isdir(job_processing_dir):
                rmtree(job_processing_dir)

            job_processing_images_dir = os.path.join(job_processing_dir,"images")

            os.makedirs(job_processing_images_dir, exist_ok=True)

            # Copy images over
            src_files = os.listdir(job_source_images)
            for file_name in src_files:
                full_file_name = os.path.join(job_source_images, file_name)
                if os.path.isfile(full_file_name):
                    print("Copy", full_file_name, "to", job_processing_images_dir)
                    copy(full_file_name, job_processing_images_dir)

            job_cmds = self.build_job_cmd(job_processing_dir, job_source_images)
            self.run_subprocess(job_cmds)


        return True

    def build_job_cmd(self, job_workspace, output_path):

        dataset_cmd="DATASET_PATH={}".format(job_workspace)
        brush_app="BRUSH={}".format(self.path_to_brush_app)
        brush_output="BRUSH_OUTPUT_PATH={}".format(output_path)

        # Sparse reconstruction
        feature_extractor_cmd = "colmap feature_extractor --image_path {}/images --database_path {}/database.db".format(job_workspace, job_workspace)

        exhaustive_matcher_cmd = "colmap exhaustive_matcher --database_path {}/database.db".format(job_workspace)

        mapper_cmd = "glomap mapper --database_path {}/database.db --image_path {}/images --output_path {}/sparse".format(
            job_workspace, job_workspace, job_workspace
        )

        # Gaussian Splatting with Brush
        brush_cmd = "{}/brush_app {} --export-path {}".format(
            self.path_to_brush_app, job_workspace, output_path
        )
        print("BRUSH", brush_cmd)
        all_cmds = [
                # dataset_cmd, 
                # brush_app, 
                # brush_output,
                feature_extractor_cmd, 
                exhaustive_matcher_cmd, 
                mapper_cmd, 
                brush_cmd
                ]

        return all_cmds



    def run_subprocess(self, cmds):
        for cmd in cmds:
            c = cmd.split()
            self.sub_p = subprocess.Popen(
                c,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            while True:
                line = self.sub_p.stdout.readline()
                if not line:
                    break
                print(line)

            output, errors = self.sub_p.communicate()


    def list_images(self, folder):

        for file in os.listdir(folder):
            if file.split(".")[-1] in ["png", "jpeg", "jpg"]:
                # Early return if an image was found
                return True

        return False

    def list_ply(self, folder):
        # Returns true if the folder contains at least one .ply file
        for file in os.listdir(folder):
            if file.split(".")[-1] in ["ply"]:
                # Early return if an ply was found
                return True

        return False

    def __cleanup(self):
        pass
        print("Cleanup...")
        try:
            self.sub_p.kill()
        except:
            print("No subprocess running")

if __name__ == "__main__":
    path_to_images = sys.argv[1] if len(sys.argv)>1 else "."
    print("Path is", path_to_images)
    sw = SplatWatch(path_to_images)
    sw.run()
