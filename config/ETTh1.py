exp_conf = dict(
    model_name="FDCNet",
    dataset_name='ETTh1',
    exp_runner='exp_with_aux_loss',

    hist_len=96,
    pred_len=96,

    use_repr=False,
    hidden_dim=96,
    layer_config=[("TaylorKAN", 4), ("TaylorKAN", 4), ("WavKAN", None), ("WavKAN", None)],

    lr=0.01,
    dropout=0.1,
    construction_loss_weight=0.5,
    max_epochs=20,
    batch_size=128
)
