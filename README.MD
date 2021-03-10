### Non-Autoregressive streaming speech recognation 

#### Model backbone: rezero-transformer with directional mask
#### Core streaming module: ctc trigger
#### Training tools: amp & pytorch-lightning 
#### Core module is written in batched tensor operation rather than for-loop for speeding up: See src/model/module/mask.py and src/model/layer/spec_augment.py
#### Model main structure: See src/model/lightning_module/
#### Model training and inference with half precision
#### Dataset module is hdfs based rather than original dataset for fast data preprocess and fast training
#### There are some task remained to be done: add grpc client, grpc server and a streamlit webapp as my former stream asr project to make it truly usable. And model to be tested for tensor rt and onnx/onnxruntime and converted, there could be some operator to be written with cuda c and binding into pytorch. Use other transformer backbone for fast inference such as sparse transformer/ reformer etc.

#### environment: docker , dockerfile see my github resp "docker_image"
