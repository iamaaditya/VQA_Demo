# VQA Model 

## Block Diagram

<img src='https://raw.githubusercontent.com/iamaaditya/VQA_Demo/master/model_vqa.png'>

## Model JSON


```json
{
    "layers": [
        {
            "cache_enabled": true,
            "concat_axis": 1,
            "dot_axes": -1,
            "layers": [
                {
                    "layers": [
                        {
                            "activation": "tanh",
                            "cache_enabled": true,
                            "forget_bias_init": "one",
                            "go_backwards": false,
                            "init": "glorot_uniform",
                            "inner_activation": "hard_sigmoid",
                            "inner_init": "orthogonal",
                            "input_dim": 300,
                            "input_length": null,
                            "input_shape": [
                                30,
                                300
                            ],
                            "name": "LSTM",
                            "output_dim": 512,
                            "return_sequences": true,
                            "stateful": false
                        },
                        {
                            "activation": "tanh",
                            "cache_enabled": true,
                            "forget_bias_init": "one",
                            "go_backwards": false,
                            "init": "glorot_uniform",
                            "inner_activation": "hard_sigmoid",
                            "inner_init": "orthogonal",
                            "input_dim": 512,
                            "input_length": null,
                            "name": "LSTM",
                            "output_dim": 512,
                            "return_sequences": true,
                            "stateful": false
                        },
                        {
                            "activation": "tanh",
                            "cache_enabled": true,
                            "forget_bias_init": "one",
                            "go_backwards": false,
                            "init": "glorot_uniform",
                            "inner_activation": "hard_sigmoid",
                            "inner_init": "orthogonal",
                            "input_dim": 512,
                            "input_length": null,
                            "name": "LSTM",
                            "output_dim": 512,
                            "return_sequences": false,
                            "stateful": false
                        }
                    ],
                    "name": "Sequential"
                },
                {
                    "layers": [
                        {
                            "cache_enabled": true,
                            "dims": [
                                4096
                            ],
                            "input_shape": [
                                4096
                            ],
                            "name": "Reshape"
                        }
                    ],
                    "name": "Sequential"
                }
            ],
            "mode": "concat",
            "name": "Merge"
        },
        {
            "W_constraint": null,
            "W_regularizer": null,
            "activation": "linear",
            "activity_regularizer": null,
            "b_constraint": null,
            "b_regularizer": null,
            "cache_enabled": true,
            "init": "uniform",
            "input_dim": null,
            "name": "Dense",
            "output_dim": 1024
        },
        {
            "activation": "tanh",
            "cache_enabled": true,
            "name": "Activation"
        },
        {
            "cache_enabled": true,
            "name": "Dropout",
            "p": 0.5
        },
        {
            "W_constraint": null,
            "W_regularizer": null,
            "activation": "linear",
            "activity_regularizer": null,
            "b_constraint": null,
            "b_regularizer": null,
            "cache_enabled": true,
            "init": "uniform",
            "input_dim": null,
            "name": "Dense",
            "output_dim": 1024
        },
        {
            "activation": "tanh",
            "cache_enabled": true,
            "name": "Activation"
        },
        {
            "cache_enabled": true,
            "name": "Dropout",
            "p": 0.5
        },
        {
            "W_constraint": null,
            "W_regularizer": null,
            "activation": "linear",
            "activity_regularizer": null,
            "b_constraint": null,
            "b_regularizer": null,
            "cache_enabled": true,
            "init": "uniform",
            "input_dim": null,
            "name": "Dense",
            "output_dim": 1024
        },
        {
            "activation": "tanh",
            "cache_enabled": true,
            "name": "Activation"
        },
        {
            "cache_enabled": true,
            "name": "Dropout",
            "p": 0.5
        },
        {
            "W_constraint": null,
            "W_regularizer": null,
            "activation": "linear",
            "activity_regularizer": null,
            "b_constraint": null,
            "b_regularizer": null,
            "cache_enabled": true,
            "init": "glorot_uniform",
            "input_dim": null,
            "name": "Dense",
            "output_dim": 1000
        },
        {
            "activation": "softmax",
            "cache_enabled": true,
            "name": "Activation"
        }
    ],
    "name": "Sequential"
}
```
