# PhysCap: Physically Plausible Monocular 3D Motion Capture in Real Time
The implementation is based on [SIGGRAPH Aisa'20](https://vcai.mpi-inf.mpg.de/projects/PhysCap/). 
 
## Dependencies
- Python 3.7
- Ubuntu 18.04  (The system should run on other Ubuntu versions and Windows, however not tested.)
- RBDL: Rigid Body Dynamics Library v.2.6.0 (https://github.com/rbdl/rbdl) (Important: set "RBDL_BUILD_ADDON_URDFREADER " to be "ON" when you compile. Also don't forget to add the compiled rbdl library in your python path use it.)
- PyTorch 1.8.1 with GPU support (cuda 10.2 is tested to work)
- For other python packages, please check requirements.txt

## Installation
- Download and install Python binded RBDL from  https://github.com/rbdl/rbdl
- Install Pytorch 1.8.1 with GPU support (https://pytorch.org/) (other versions should also work but not tested)
- Install python packages by:

		pip install -r requirements.txt

## How to Run on the Sample Data
We provide a sample data taken from DeepCap dataset [CVPR'20](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/). To run the code on the sample data, first go to physcap_release directory and run:

    python pipeline.py --contact_estimation 0 --floor_known 1 --floor_frame  data/floor_frame.npy  --humanoid_path asset/physcap.urdf --skeleton_filename asset/physcap.skeleton --motion_filename data/sample.motion --contact_path data/sample_contacts.npy --stationary_path data/sample_stationary.npy --save_path './results/'

To visualize the prediction, run:

    python visualizer.py --q_path ./results/PhyCap_q.npy

To run PhysCap with its full functionality, the floor position should be given as 4x4 matrix (rotation and translation). In case you don't know the floor position, you can still run PhysCap with "--floor_known 0" option:

    python pipeline.py --contact_estimation 0 --floor_known 0  --humanoid_path asset/physcap.urdf --skeleton_filename asset/physcap.skeleton --motion_filename data/sample.motion --save_path './results/'

## How to Run on Your Data

1) Run Stage I: 

	we employ [VNect](http://gvv.mpi-inf.mpg.de/projects/VNect/) for the stage I of PhysCap pipeline.  Please install the VNect C++ library and use its prediction to run PhysCap. When running VNect, please replace "default.skeleton" with "physcap.skeleton" in asset folder that is compatible with PhysCap skeletion definition (physcap.urdf). After running VNect on your sequence, the predictions (motion.motion and ddd.mdd) will be saved under the specified folder. For this example, we assuem the predictions are saved under "data/VNect_data" folder.

2) Run Stage II and III:
	
	First, run the following command to apply preprocessing on the 2D keypoints:

		python process_2Ds.py --input ./data/VNect_data/ddd.mdd --output ./data/VNect_data/ --smoothing 0

	The processed keypoints will be stored as "vnect_2ds.npy". Then run the following command to run Stage II and III:
		
		python pipeline.py --contact_estimation 1 --vnect_2d_path ./data/VNect_data/vnect_2ds.npy --save_path './results/' --floor_known 0 --humanoid_path asset/physcap.urdf --skeleton_filename asset/physcap.skeleton --motion_filename ./data/VNect_data/motion.motion --contact_path results/contacts.npy --stationary_path results/stationary.npy  
	In case you know the exact floor position, you can use the options --floor_known 1 --floor_frame /Path/To/FloorFrameFile

	To visualize the results, run:

		python visualizer.py --q_path ./results/PhyCap_q.npy


## License Terms
Permission is hereby granted, free of charge, to any person or company obtaining a copy of this software and associated documentation files (the "Software") from the copyright holders to use the Software for any non-commercial purpose. Publication, redistribution and (re)selling of the software, of modifications, extensions, and derivates of it, and of other software containing portions of the licensed Software, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee. 

Packaging or distributing parts or whole of the provided software (including code, models and data) as is or as part of other software is prohibited. Commercial use of parts or whole of the provided software (including code, models and data) is strictly prohibited. Using the provided software for promotion of a commercial entity or product, or in any other manner which directly or indirectly results in commercial gains is strictly prohibited. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation
If the code is used, the licesnee is required to cite the use of VNect and the following publication in any documentation 
or publication that results from the work:
```
@article{
	PhysCapTOG2020,
	author = {Shimada, Soshi and Golyanik, Vladislav and Xu, Weipeng and Theobalt, Christian},
	title = {PhysCap: Physically Plausible Monocular 3D Motion Capture in Real Time},
	journal = {ACM Transactions on Graphics}, 
	month = {dec},
	volume = {39},
	number = {6}, 
	articleno = {235},
	year = {2020}, 
	publisher = {ACM}, 
	keywords = {physics-based, 3D, motion capture, real time}
} 
```
