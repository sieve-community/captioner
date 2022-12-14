# Video Captioning

Deployed on Sieve at `developer-sievedata-com/captioner`. You can also deploy it yourself using `sieve model` inside this repo's directory.

### Create a workflow using this model

`pip install https://storage.googleapis.com/sieve-client-package/sievedata-0.0.1.3-py3-none-any.whl`

#### Python
```Python
from sieve.api.client import SieveClient, SieveProject
from sieve.types.api import *
cli = SieveClient()

proj = SieveProject(
    name="captioning",
    fps=1,
    store_data=True,
    workflow=SieveWorkflow([
        SieveLayer(
            iteration_type=SieveLayerIterationType.video,
            models=[
                SieveModel(
                    name="developer-sievedata-com/captioner",
                )
            ]
        )
    ])
)
proj.create()
```

#### YAML
```
# workflow.yaml
fps: 1
store_data: true
layers:
- iteration: "video"
  models: 
  - model_name: developer-sievedata-com/captioner
```
`sieve projects scene_recognition create -wf workflow.yaml`
