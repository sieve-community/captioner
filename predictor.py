from typing import List, Dict
from sieve.types import FrameSingleObject, SingleObject, BoundingBox, Temporal
from sieve.predictors import TemporalPredictor

import torch
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 

class Captioner(TemporalPredictor):
    def setup(self):
        device='cuda'
        encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)
    
    def predict(self, frame: FrameSingleObject) -> List[SingleObject]:
        frame_number = frame.get_temporal().frame_number
        frame_data = frame.get_temporal().get_array()
        image = self.feature_extractor(frame_data, return_tensors="pt").pixel_values.to('cuda')
        clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
        caption_ids = self.model.generate(image, max_length = 64)[0]
        caption_text = clean_text(self.tokenizer.decode(caption_ids))
        
        return [SingleObject(
            cls='caption',
            temporal=Temporal(
                frame_number=frame_number,
                bounding_box=BoundingBox.from_array([0, 0, frame.width, frame.height]),
                text=caption_text,
                score=1.0
            )
        )]
