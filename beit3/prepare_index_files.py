from datasets import CaptioningDataset
from transformers import XLMRobertaTokenizer
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--data", type=str, default="coco")
args = arg_parser.parse_args()



tokenizer = XLMRobertaTokenizer("/projects/nlp/people/rdp455/beit3/pretrained_model/beit3.spm")
if args.data == "coco":
    CaptioningDataset.make_coco_captioning_dataset_index(
        data_path="/projects/nlp/people/rdp455/beit3_data",
        tokenizer=tokenizer,
    )
elif args.data == "flickr30k":
    CaptioningDataset.make_flickr_captioning_curation_train_dataset_index(
        data_path="/projects/nlp/people/rdp455/beit3_data", 
        img_dir="flickr30k/images", 
        tokenizer=tokenizer,
    )

elif args.data == "coco_curation":
    CaptioningDataset.make_coco_captioning_curation_train_dataset_index(
        data_path="/projects/nlp/people/rdp455/beit3_data",
        tokenizer=tokenizer,
    )
