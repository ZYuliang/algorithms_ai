import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
data_dir = os.path.join(root_dir,'my_datasets')
raw_data_dir = os.path.join(data_dir,'raw_data')
ner_dataset_dir = os.path.join(data_dir,'ner')

test_dataset_dir = os.path.join(data_dir,'test')
model_dir = os.path.join(root_dir,'my_models')
test_model_dir = os.path.join(model_dir,'test')

#
cadec2_raw_data = os.path.join(raw_data_dir,'cadec2')
cadec2_ner_results = os.path.join(cadec2_raw_data,'ner_results.jsonl')
cadec2_dataset_dict = os.path.join(ner_dataset_dir,'cadec2/dataset_dict')
