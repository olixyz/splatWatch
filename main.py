import time
import datetime
import sys
import yaml
import os
from shutil import rmtree, copy, move
import atexit
import subprocess
import logging


class SplatWatch():
    '''
    Checks for sub-folders (from yaml file "queue" in source dir) with images in a source directory (specified as sys.arg to this script).

    Creates a processing folder (specified in yaml).
    Copies images into the processing folder and runs Colmap/Glomap sparse, dense and Brush (Gaussian Splatting).


    After processing a file named "done" is written into the processing folder.
    Folder with a "done" file are skipped.


    '''
    def __init__(self, path_to_source_images):
        self.source_images=path_to_source_images
        self.processing_dir=""
        self.path_to_brush_app=""
        self.sub_p = None  # the subprocess
        self.logger = logging.getLogger(__name__)
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
            with open(os.path.join(self.source_images, 'queue'), 'r') as file:
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

        if not os.path.exists(self.processing_dir):
            os.makedirs(self.processing_dir, exist_ok=True)

        # Setup logging
        log_file = os.path.join(self.processing_dir,"capture.log")
        if not os.path.exists(log_file):
            open(log_file, 'w+').close()

        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="w",
            encoding="utf-8",
            format='%(asctime)s %(message)s'
        )

        print(self.path_to_brush_app)
        print("processing Dir:" , self.processing_dir)


        # Go through each "job" and do sanity checks
        for el in data["queue"]:


            # Job defaults
            dense= False
            splat= False
            output="" # set default output
            every = 1 # use every x image


            # Job settings

            if not "folder" in el:
                # set output from image source folder
                print("Entry has no folder, skipping")
                continue

            output = el["folder"] 

            if "output" in el:
                output = el["output"]

            if "every" in el:
                every = el["every"]

            if "dense" in el:
                dense = el["dense"]

            if "splat" in el:
                splat = el["splat"]


            job_source_images = os.path.join(self.source_images, el["folder"])
            valid_source_path = os.path.isdir(job_source_images)

            # Folder exists?
            if not valid_source_path:
                print("Source image path does not exist: " , job_source_images)
                continue

            # Folder has list of images?
            has_images = self.list_images(job_source_images)
            if not has_images:
                print("Folder has no images: " , job_source_images)
                continue


            # Create a new dir in processing_dir and /images
            job_processing_dir = os.path.join(self.processing_dir, output)

            # Check this folder if done already
            if os.path.exists(os.path.join(job_processing_dir, "done")):
                print("Folder already done:" , job_processing_dir)
                continue

            # Folder valid and not processed 

            self.logger.info("-----------")
            self.logger.info("Folder not processed: " + job_source_images)
            self.logger.info("Processing Folder: " + el["folder"])
            self.logger.info("Output set from user: "+ output)
            self.logger.info("Every nth image set from user: "+ str(every))
            self.logger.info("Dense set from user: " + str(dense))
            self.logger.info("Splat set from user: " + str(splat))


            # Delete existing processing dir
            if os.path.isdir(job_processing_dir):
                rmtree(job_processing_dir)

            job_processing_images_dir = os.path.join(job_processing_dir,"images")

            os.makedirs(job_processing_images_dir, exist_ok=True)

            # Copy images to processing folder 
            src_files = os.listdir(job_source_images)
            # name_counter = 0
            for count, file_name in enumerate(src_files):
                if (count % every) == 0:
                    full_file_name = os.path.join(job_source_images, file_name)
                    target_file_name = os.path.join(job_processing_images_dir, file_name)
                    # target_file_name = os.path.join(job_processing_images_dir, str(name_counter).zfill(4) + os.path.splitext(file_name)[1])
                    # name_counter += 1

                    if os.path.isfile(full_file_name):
                        self.logger.info("Copy    " + full_file_name + "  to  " + target_file_name)
                        copy(full_file_name, target_file_name)


            job_cmds = self.build_job_cmd(job_processing_dir, job_source_images, dense, splat)
            self.run_subprocess(job_cmds)

            done_file = os.path.join(job_processing_dir, "done")
            if not os.path.exists(done_file):
                open(done_file, 'w').close()

            self.logger.info("done")


        return True

    def build_job_cmd(self, job_workspace, output_path, dense, splat):

        dataset_cmd="DATASET_PATH={}".format(job_workspace)
        brush_app="BRUSH={}".format(self.path_to_brush_app)
        brush_output="BRUSH_OUTPUT_PATH={}".format(output_path)

        # Sparse reconstruction
        feature_extractor_cmd = "colmap feature_extractor --image_path {}/images --database_path {}/database.db".format(job_workspace, job_workspace)

        exhaustive_matcher_cmd = "colmap exhaustive_matcher --database_path {}/database.db".format(job_workspace)

        mapper_cmd = "glomap mapper --database_path {}/database.db --image_path {}/images --output_path {}/sparse".format(
            job_workspace, job_workspace, job_workspace
        )

        # Undistort images
        image_undistorter_cmd="colmap image_undistorter --image_path {}/images --input_path {}/sparse/0 --output_path {}/dense --output_type COLMAP --max_image_size 2000".format(job_workspace, job_workspace, job_workspace)


        # Dense reconstruction:
        patch_match_stereo_cmd ="colmap patch_match_stereo --workspace_path {}/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size 2000".format(job_workspace)

        # This creates the dense colored pointcloud (fuse.ply):
        stereo_fusion_cmd = "colmap stereo_fusion --workspace_path {}/dense --workspace_format COLMAP --input_type geometric --output_path {}/dense_fused.ply".format(job_workspace, job_workspace)



        # Gaussian Splatting with Brush
        brush_cmd = "{}/brush_app {} --export-path {}".format(
            self.path_to_brush_app, job_workspace, job_workspace
        )

        all_cmds = [
                # dataset_cmd, 
                # brush_app, 
                # brush_output,
                feature_extractor_cmd, 
                exhaustive_matcher_cmd, 
                mapper_cmd, 
                image_undistorter_cmd
                ]
        if dense:
            all_cmds.append(patch_match_stereo_cmd)
            all_cmds.append(stereo_fusion_cmd)
        if splat:
            all_cmds.append(brush_cmd)

        return all_cmds



    def run_subprocess(self, cmds):
        for cmd in cmds:
            self.logger.info(cmd)
            c = cmd.split()
            print("Running command", cmd)
            # continue
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
                self.logger.info(line)
                # print(line)

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
