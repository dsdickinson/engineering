***THIS IS A DEMO OF A WORK IN PROGRESS.***

# cap_infer_play.py

# Purpose
This script runs computer vision inference requests on a Jetson Orin Nano Dev Kit to a local Triton server for video streams, mp4 videos, etc.<br/><br/>
The _cap_infer_play.py_ script is going to be part of a bigger tool set, eventually. For now, I just want to demonstrate how object detection inferencing works on my Jetson Orin Nano.

## <ins>Network Diagram</ins>
This network diagram shows the intended full build-out of the required infrastructure for the related _cv-infer-py_ project that I have in the works. This demo currently only includes functionality for steps 1-3.<br/><br/>
![cv_infer_py_backend_diagram_0001 drawio](https://github.com/user-attachments/assets/fbed0465-113d-437a-936a-afd45de48051)

## <ins>System Setup</ins>

_NOTE: For this demo, all the following should happen directly on the Jetson Orin Nano Dev Kit._

### Setup Python Environment.
```
> mkdir ~/git
> cd ~/git
> sudo apt install python3.10-venv
> python -m venv infer_env_jetson
> source infer_env_jetson/bin/activate
> cd infer_env_jetson/
```
### Setup this demo's repo
```
> sudo apt-get install git-lfs
> git clone git@github.com:dsdickinson/engineering.git
> cd engineering/python/ai/computer_vision/demo-01/
> git lfs fetch --all
> git lfs pull
> sudo apt-get install libhdf5-dev (for hdf5 Python package)
> pip3 install --upgrade pip setuptools wheel # (will help w/ requirements.txt installs)
> pip3 install -r ./requirements.txt --no-cache-dir > requirements_install.txt
```

### Get Triton Client/Server bits.
```
> cd ~/git/
> mkdir triton-inference-server
> cd triton-inference-server/
> git clone -b r24.12 https://github.com/triton-inference-server/server.git
> git clone -b r24.12 https://github.com/triton-inference-server/client.git
```

### Setup Base Triton Models.
```
> cd server/docs/examples/
> ./fetch_models.sh
> sudo cp -rf model_repository /models
```

### Setup object detection model.
###### Get the model and compile the .proto files.
```
> cd ~/git/
> git clone git@github.com:tensorflow/models.git
> cd models/research/
> sudo apt install protobuf-compile
> protoc object_detection/protos/*.proto --python_out=.
> cp object_detection/packages/tf2/setup.py .
```

#### Fix some object detection model issues.
We need to fix a couple of things here to make the object detection model work on our system.

> ISSUE:
>
> ```
> ImportError: cannot import name 'string_int_label_map_pb2'
> ```
>
>
> FIX:<br/>
> https://github.com/tensorflow/models/issues/6148
> 
> Overwrite with a working version of the file.
> ```
> > wget -O ~/git/infer_env_jetson/lib/python3.10/site-packages/object_detection/protos/string_int_label_map_pb2.py \
> https://github.com/datitran/object_detector_app/blob/master/object_detection/protos/string_int_label_map_pb2.py
> ```

</br>

> ISSUE:
> ```
> https://stackoverflow.com/questions/55591437/attributeerror-module-tensorflow-has-no-attribute-gfile
> Traceback (most recent call last):
>   File "/home/steve/git/infer_env_jetson/cv-infer-py/gpu/capture/./02_cap_infer.py", line 387, in <module>
>     category_index = label_map_util.create_category_index_from_labelmap("./labels.txt", use_display_name=True)
>   File "/home/steve/git/infer_env_jetson/lib/python3.10/site-packages/object_detection/utils/label_map_util.py", line 229, in create_category_index_from_labelmap
>     categories = create_categories_from_labelmap(label_map_path, use_display_name)
>   File "/home/steve/git/infer_env_jetson/lib/python3.10/site-packages/object_detection/utils/label_map_util.py", line 209, in create_categories_from_labelmap
>     label_map = load_labelmap(label_map_path)
>   File "/home/steve/git/infer_env_jetson/lib/python3.10/site-packages/object_detection/utils/label_map_util.py", line 132, in load_labelmap
>     with tf.gfile.GFile(path, 'r') as fid:
> AttributeError: module 'tensorflow' has no attribute 'gfile'. Did you mean: 'fill'?
> ```
> FIX:<br/>
> Change tf.gfile.GFile to tf.io.gfile.GFile.
> ```
> > vi ~/git/infer_env_jetson/lib/python3.10/site-packages/object_detection/utils/label_map_util.py
> ```

###### Deploy model
```
> sudo cp -rf object_detection /models
```

###### Setup Tensorflow definition file for object detection model.
<details>
<summary>config.pbtxt</summary>
  
```
> sudo vi /models/object_detection/config.pbtxt
name: "detection"
platform: "tensorflow_graphdef"
max_batch_size: 1
input [
  {
    name: "image_tensor"
    data_type: TYPE_UINT8
    format: FORMAT_NHWC
    dims: [ 600, 1024, 3 ]
  }
]
output [
  {
    name: "detection_boxes"
    data_type: TYPE_FP32
    dims: [ 100, 4]
    reshape { shape: [100,4] }
  },
  {
    name: "detection_classes"
    data_type: TYPE_FP32
    dims: [ 100 ]
    reshape { shape: [ 1, 100 ] }
  },
  {
    name: "detection_scores"
    data_type: TYPE_FP32
    dims: [ 100 ]

  },
  {
    name: "num_detections"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape { shape: [] }
  }
]
```
</details>

###### Add labels file for object detection.
<details>
<summary>labels.txt</summary>

```
> sudo vi /models/object_detection/labels.txt
item {
  name: "/m/01g317"
  id: 1
  display_name: "person"
}
item {
  name: "/m/0199g"
  id: 2
  display_name: "bicycle"
}
item {
  name: "/m/0k4j"
  id: 3
  display_name: "car"
}
item {
  name: "/m/04_sv"
  id: 4
  display_name: "motorcycle"
}
item {
  name: "/m/05czz6l"
  id: 5
  display_name: "airplane"
}
item {
  name: "/m/01bjv"
  id: 6
  display_name: "bus"
}
item {
  name: "/m/07jdr"
  id: 7
  display_name: "train"
}
item {
  name: "/m/07r04"
  id: 8
  display_name: "truck"
}
item {
  name: "/m/019jd"
  id: 9
  display_name: "boat"
}
item {
  name: "/m/015qff"
  id: 10
  display_name: "traffic light"
}
item {
  name: "/m/01pns0"
  id: 11
  display_name: "fire hydrant"
}
item {
  name: "/m/02pv19"
  id: 13
  display_name: "stop sign"
}
item {
  name: "/m/015qbp"
  id: 14
  display_name: "parking meter"
}
item {
  name: "/m/0cvnqh"
  id: 15
  display_name: "bench"
}
item {
  name: "/m/015p6"
  id: 16
  display_name: "bird"
}
item {
  name: "/m/01yrx"
  id: 17
  display_name: "cat"
}
item {
  name: "/m/0bt9lr"
  id: 18
  display_name: "dog"
}
item {
  name: "/m/03k3r"
  id: 19
  display_name: "horse"
}
item {
  name: "/m/07bgp"
  id: 20
  display_name: "sheep"
}
item {
  name: "/m/01xq0k1"
  id: 21
  display_name: "cow"
}
item {
  name: "/m/0bwd_0j"
  id: 22
  display_name: "elephant"
}
item {
  name: "/m/01dws"
  id: 23
  display_name: "bear"
}
item {
  name: "/m/0898b"
  id: 24
  display_name: "zebra"
}
item {
  name: "/m/03bk1"
  id: 25
  display_name: "giraffe"
}
item {
  name: "/m/01940j"
  id: 27
  display_name: "backpack"
}
item {
  name: "/m/0hnnb"
  id: 28
  display_name: "umbrella"
}
item {
  name: "/m/080hkjn"
  id: 31
  display_name: "handbag"
}
item {
  name: "/m/01rkbr"
  id: 32
  display_name: "tie"
}
item {
  name: "/m/01s55n"
  id: 33
  display_name: "suitcase"
}
item {
  name: "/m/02wmf"
  id: 34
  display_name: "frisbee"
}
item {
  name: "/m/071p9"
  id: 35
  display_name: "skis"
}
item {
  name: "/m/06__v"
  id: 36
  display_name: "snowboard"
}
item {
  name: "/m/018xm"
  id: 37
  display_name: "sports ball"
}
item {
  name: "/m/02zt3"
  id: 38
  display_name: "kite"
}
item {
  name: "/m/03g8mr"
  id: 39
  display_name: "baseball bat"
}
item {
  name: "/m/03grzl"
  id: 40
  display_name: "baseball glove"
}
item {
  name: "/m/06_fw"
  id: 41
  display_name: "skateboard"
}
item {
  name: "/m/019w40"
  id: 42
  display_name: "surfboard"
}
item {
  name: "/m/0dv9c"
  id: 43
  display_name: "tennis racket"
}
item {
  name: "/m/04dr76w"
  id: 44
  display_name: "bottle"
}
item {
  name: "/m/09tvcd"
  id: 46
  display_name: "wine glass"
}
item {
  name: "/m/08gqpm"
  id: 47
  display_name: "cup"
}
item {
  name: "/m/0dt3t"
  id: 48
  display_name: "fork"
}
item {
  name: "/m/04ctx"
  id: 49
  display_name: "knife"
}
item {
  name: "/m/0cmx8"
  id: 50
  display_name: "spoon"
}
item {
  name: "/m/04kkgm"
  id: 51
  display_name: "bowl"
}
item {
  name: "/m/09qck"
  id: 52
  display_name: "banana"
}
item {
  name: "/m/014j1m"
  id: 53
  display_name: "apple"
}
item {
  name: "/m/0l515"
  id: 54
  display_name: "sandwich"
}
item {
  name: "/m/0cyhj_"
  id: 55
  display_name: "orange"
}
item {
  name: "/m/0hkxq"
  id: 56
  display_name: "broccoli"
}
item {
  name: "/m/0fj52s"
  id: 57
  display_name: "carrot"
}
item {
  name: "/m/01b9xk"
  id: 58
  display_name: "hot dog"
}
item {
  name: "/m/0663v"
  id: 59
  display_name: "pizza"
}
item {
  name: "/m/0jy4k"
  id: 60
  display_name: "donut"
}
item {
  name: "/m/0fszt"
  id: 61
  display_name: "cake"
}
item {
  name: "/m/01mzpv"
  id: 62
  display_name: "chair"
}
item {
  name: "/m/02crq1"
  id: 63
  display_name: "couch"
}
item {
  name: "/m/03fp41"
  id: 64
  display_name: "potted plant"
}
item {
  name: "/m/03ssj5"
  id: 65
  display_name: "bed"
}
item {
  name: "/m/04bcr3"
  id: 67
  display_name: "dining table"
}
item {
  name: "/m/09g1w"
  id: 70
  display_name: "toilet"
}
item {
  name: "/m/07c52"
  id: 72
  display_name: "tv"
}
item {
  name: "/m/01c648"
  id: 73
  display_name: "laptop"
}
item {
  name: "/m/020lf"
  id: 74
  display_name: "mouse"
}
item {
  name: "/m/0qjjc"
  id: 75
  display_name: "remote"
}
item {
  name: "/m/01m2v"
  id: 76
  display_name: "keyboard"
}
item {
  name: "/m/050k8"
  id: 77
  display_name: "cell phone"
}
item {
  name: "/m/0fx9l"
  id: 78
  display_name: "microwave"
}
item {
  name: "/m/029bxz"
  id: 79
  display_name: "oven"
}
item {
  name: "/m/01k6s3"
  id: 80
  display_name: "toaster"
}
item {
  name: "/m/0130jx"
  id: 81
  display_name: "sink"
}
item {
  name: "/m/040b_t"
  id: 82
  display_name: "refrigerator"
}
item {
  name: "/m/0bt_c3"
  id: 84
  display_name: "book"
}
item {
  name: "/m/01x3z"
  id: 85
  display_name: "clock"
}
item {
  name: "/m/02s195"
  id: 86
  display_name: "vase"
}
item {
  name: "/m/01lsmm"
  id: 87
  display_name: "scissors"
}
item {
  name: "/m/0kmg4"
  id: 88
  display_name: "teddy bear"
}
item {
  name: "/m/03wvsk"
  id: 89
  display_name: "hair drier"
}
item {
  name: "/m/012xff"
  id: 90
  display_name: "toothbrush"
}
```
</details>

### Triton Server
##### Start server
```
> sudo docker run -d --gpus=1 --runtime=nvidia --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/models:/models nvcr.io/nvidia/tritonserver:24.01-py3-igpu tritonserver --model-repository=/models --strict-model-config=false
```

##### Validations
###### Check Triton server is ready.
```
> curl -v http://localhost:8000/v2/health/ready
```

###### Check model config.
```
> curl http://localhost:8000/v2/models/object_detection/config | jq
```

###### Run a test inference request against an image.
```
> cd ~/git/triton-inference-server/client/src/python/examples/
> ./image_client.py -m densenet_onnx -c 3 -s INCEPTION ../../../../server/qa/images/mug.jpg
Request 1, batch size 1
    13.916380 (504) = COFFEE MUG
    12.018959 (968) = CUP
    9.840457 (967) = ESPRESSO
PASS
```

## <ins>Demo Execution</ins>
Run Triton inference against a test video.
```
> cd ~/git/engineering/python/ai/computer_vision/demo-01/
> ./cap_infer_play.py -s videos/4791734-hd_1920_1080_30fps.mp4
```
![Screenshot from 2025-02-24 15-01-55](https://github.com/user-attachments/assets/ebea5dd3-a51b-4a96-90aa-56df03ad2f53)
