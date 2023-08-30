---
dataset_info:
  config_name: taco-multi
  features:
  - name: image_id
    dtype: int64
  - name: image
    dtype: image
  - name: width
    dtype: int32
  - name: height
    dtype: int32
  - name: objects
    sequence:
    - name: id
      dtype: int64
    - name: area
      dtype: int64
    - name: bbox
      sequence: float32
      length: 4
    - name: category
      dtype:
        class_label:
          names:
            '0': metals_and_plastic
            '1': other
            '2': non_recyclable
            '3': glass
            '4': paper
            '5': bio
            '6': unknown
  splits:
  - name: train
    num_bytes: 1006510
    num_examples: 3647
  - name: test
    num_bytes: 248312
    num_examples: 915
  download_size: 10265127938
  dataset_size: 1254822
---
