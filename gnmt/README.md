

Add the following code to inference.py

```
  out_dir = "/home/abduld/mlperf/inference/v0.5/translation/gnmt/tensorflow/savedmodel"
  # Create savedmodel
  with  tf.Session(graph=infer_model.graph) as sess:
    loaded_model, global_step = model_helper.create_or_load_model(
        infer_model.model, out_dir, sess, "infer_name")
    # tf.saved_model.save(loaded_model, out_dir)
    # ckpt_path = loaded_model.saver.sa