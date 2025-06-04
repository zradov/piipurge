from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


#TEXT = 'THIS CO-BRANDING AND ADVERTISING AGREEMENT (the "Agreement") is made as of [DATE] (the "Effective Date") by and between [ORG1], with its principal place of business at [ADDRESS1] ("[ORG1_SHORT]"), and [ORG2] having its principal place of business at [ADDRESS2] ("[ORG2_SHORT]").'
#TEXT = '(a) "CONTENT" means all content or information, in any medium, provided by a party to the other party for use in conjunction with the performance of its obligations hereunder, including without limitation any text, music, sound, photographs, video, graphics, data or software. Content provided by [ORG1] is referred to herein as "[ORG1] Content" and Content provided by [ORG2] is referred to herein as "[ORG2] Content."'
#TEXT = '(b) "CO-BRANDED SITE" means the web-site accessible through Domain Name, for the Services implemented by [ORG1].'
#TEXT = 'The homepage of this web-site will visibly display both [ORG2] Marks and [ORG1] Marks.'
#TEXT = '(f) "INFORMATION TRANSFER MECHANISM" means the mechanism by which [ORG1] transfers to [ORG2] information to populate the applicable [ORG2] transaction and user registration forms.'
TEXT = '2.1 OVERVIEW.  As set forth herein, [ORG1] will promote Services to its auction users (buyers and sellers), and [ORG2] shall develop Co-Branded Site, and develop the Information Transfer Mechanism working with [ORG1] to make Services available seamlessly to Customers.'
MAX_SENTENCES = 5

class Paraphraser:
    
    def __init__(self, 
                 model_name: str="Vamsi/T5_Paraphrase_Paws", 
                 max_sentences: int=5,
                 max_length: int=256,
                 top_k: int=50,
                 top_p: float=0.95,
                 temperature: float=1.0):
        
        self.model_name = model_name
        self.max_sentences = max_sentences
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    def rephrase(self, text: str) -> List[str]:
        prompt_text = f"rephrase: {text}</s>"

        encoding = self.tokenizer([prompt_text], 
                                add_special_tokens=True, 
                                return_tensors="pt", 
                                padding="max_length", 
                                truncation=True)
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = self.model.generate(input_ids=input_ids, 
                                      attention_mask=attention_masks,
                                      top_k=self.top_k,
                                      top_p=self.top_p,
                                      temperature=self.temperature,
                                      max_length=self.max_length,
                                      num_return_sequences=self.max_sentences,
                                      do_sample=True)  
        new_paragraphs = []
        for i in range(self.max_sentences):
            paragraph = self.tokenizer.decode(outputs[i], 
                                              skip_special_tokens=True, 
                                              clean_up_tokenization_spaces=True)
            new_paragraphs.append(paragraph)

        return new_paragraphs
