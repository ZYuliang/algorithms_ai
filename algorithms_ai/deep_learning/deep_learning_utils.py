def get_train_date(output_dir,mode='last'):
    import os
    import json

    ckpt_dirs = os.listdir(output_dir)
    ckpt_dirs = [i for i in ckpt_dirs if 'checkpoint-' in i]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
    last_ckpt = ckpt_dirs[-1]

    if mode=='best':
        with open(f"{output_dir}/{last_ckpt}/trainer_state.json", "r", encoding="utf-8") as f:
            train_state = json.loads(f.read())
        best_ckpt = train_state["best_model_checkpoint"]
        print(f'best_checkpoint:{best_ckpt}')  # your best ckpoint.
        return best_ckpt
    else:
        print(f'load last checkpoint:{last_ckpt}')
        return last_ckpt