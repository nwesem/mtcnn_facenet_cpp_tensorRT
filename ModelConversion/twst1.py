import inference as inf

TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
trt_runtime = trt.Runtime(TRT_LOGGER)
engine_path = './facenet_engine.plan'
engine = eng.load_engine(trt_runtime, engine_path)
print('Engine loaded successfully...')

h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
out = inf.do_inference(engine, samples, h_input, d_input, h_output, d_output, stream, 1, 160, 160)
