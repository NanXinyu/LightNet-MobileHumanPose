import onnx
import torch
import argparse
import numpy
import imageio
import onnxruntime as ort
#import tensorflow as tf
# python main/pytorch2onnx.py --gpu 0-1 --joint 18 --modelpath "./output/model_dump/snapshot_15.pth.tar" --backbone LPSKI
from config import cfg
from torchsummary import summary
from base import Transformer
#from onnx_tf.backend import prepare

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--joint', type=int, dest='joint')
    parser.add_argument('--modelpath', type=str, dest='modelpath')
    parser.add_argument('--backbone', type=str, dest='backbone')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

args = parse_args()

dummy_input = torch.randn(1, 3, 256, 256, device='cuda')

# modelpath as definite path
transformer = Transformer(args.backbone, args.joint, args.modelpath)
transformer._make_model()

single_pytorch_model = transformer.model

summary(single_pytorch_model, (3, 256, 256))

ONNX_PATH="./output/out/mobilehumanpose.onnx"#"../output/baseline.onnx"

torch.onnx.export(
    model=single_pytorch_model,
    args=dummy_input,
    f=ONNX_PATH, # where should it be saved
    verbose=False,
    export_params=True,
    do_constant_folding=False,  # fold constant values for optimization
    # do_constant_folding=True,   # fold constant values for optimization
    input_names=['input'],
    output_names=['output'],
    opset_version=11#9
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)

pytorch_result = single_pytorch_model(dummy_input)
pytorch_result = pytorch_result.cpu().detach().numpy()
print("pytorch_model output {}".format(pytorch_result.shape), pytorch_result)

ort_session = ort.InferenceSession(ONNX_PATH)
outputs = ort_session.run(None, {'input': dummy_input.cpu().numpy()})
outputs = numpy.array(outputs[0])
print("onnx_model ouput size{}".format(outputs.shape), outputs)

print("difference", numpy.linalg.norm(pytorch_result-outputs))

#TF_PATH = "../output/baseline" # where the representation of tensorflow model will be stored

# prepare function converts an ONNX model to an internel representation
# of the computational graph called TensorflowRep and returns
# the converted representation.
#tf_rep = prepare(onnx_model)  # creating TensorflowRep object

# export_graph function obtains the graph proto corresponding to the ONNX
# model associated with the backend representation and serializes
# to a protobuf file.
# tf_rep.export_graph(TF_PATH)

# TFLITE_PATH = "../output/baseline.tflite"

# PB_PATH = "../output/baseline/saved_model.pb"

# make a converter object from the saved tensorflow file
# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(PB_PATH, input_arrays=['input'], output_arrays=['output'])
#converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

# tell converter which type of optimization techniques to use
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
# converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

# converter.experimental_new_converter = True
#
# # I had to explicitly state the ops
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]

# def representative_dataset():

#     dataset_size = 10

#     for i in range(dataset_size):
#         print(i)
#         data = imageio.imread("../sample_images/" + "00000" + str(i) + ".jpg")
#         data = numpy.resize(data, [1, 3, 256, 256])
#         yield [data.astype(numpy.float32)]


# converter.experimental_new_converter = True
# converter.experimental_new_quantizer = True

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

# # input_arrays = converter.get_input_arrays()
# # converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}

# tf_lite_model = converter.convert()
# # Save the model.
# with open(TFLITE_PATH, 'wb') as f:
#     f.write(tf_lite_model)
