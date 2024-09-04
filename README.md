# camera placement simulation for timber crane 



## Getting started

1. Create conda environment `crane` based on `environment.yml`
```bash
conda env create -f environment.yml
```
2. activate environment `conda activate crane`
3. pip install -r requirements.txt
4. run simulation `python main.py`

99. deactivate environment `conda deactivate`

## How it works
Each component of the simulation (environment, crane, cameras, ...) is modelled as a standalone mujoco xml file. The components are assembled via the mesh manager utilizing `dm_control` and combiled resulting in a mj_model and mj_data structure. See `main.py` function `create_forestry_crane_mujoco_models`. A combiled model can be saved as a preset to export to resulting xml file and all meshes, see `main.py``env.export_with_assets("forestry_crane", os.path.abspath('.'))`.

## camera placement 
In `main.py` `create_forestry_crane_mujoco_models(..)` update the pos and rpy properties of each camera object to alter the positioning of each camera w.r.t. to the site where it is attached to. 


## Control
The crane is velocity controlled either via keyboard or gamepad (xbox one controller). 


## Troubleshooting

