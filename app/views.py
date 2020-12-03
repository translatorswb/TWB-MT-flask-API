from app import app
from flask import Flask, request, jsonify
import re
import os
from subword_nmt import apply_bpe
from ctranslate2 import Translator
from sacremoses import MosesTokenizer, MosesDetokenizer
import json

#constants
MOSES_TOKENIZER_DEFAULT_LANG = 'en'

#processors
lowercaser =  lambda x: x.lower() #string IN -> string OUT
desegmenter = lambda x: re.sub('(@@ )|(@@ ?$)', '', ' '.join(x)) #list IN -> string OUT
capitalizer = lambda x: x.capitalize() #string IN -> string OUT
token_segmenter = lambda x: x.strip().split()  #string IN -> list OUT
token_desegmenter = lambda x: ' '.join(x) #list IN -> string OUT
dummy_translator = lambda x: x

#models
loaded_models = {}

def get_tokenizer(lang):
    try:
        moses_tokenizer = MosesTokenizer(lang=lang)
    except:
        print("WARNING: Moses doesn't have tokenizer for", lang)
        moses_tokenizer = MosesTokenizer(lang='en')
        
    tokenizer = lambda x: moses_tokenizer.tokenize(x, return_str=True) #string IN -> string OUT
    return tokenizer

def get_segmenter(bpe_codes_path):
    bpe = apply_bpe.BPE(codes=open(bpe_codes_path, 'r'))
    segmenter = lambda x: bpe.process_line(x.strip()).split() #string IN -> list OUT
    return segmenter

def get_translator(ctranslator_model_path):
    ctranslator = Translator(ctranslator_model_path)
    translator = lambda x: ctranslator.translate_batch([x])[0][0]['tokens']  #list IN -> list OUT
    return translator

def get_detokenizer(lang):
    try:
        moses_detokenizer = MosesDetokenizer(lang=lang)
    except:
        print("WARNING: Moses doesn't have tokenizer for", lang)
        moses_detokenizer = MosesDetokenizer(lang=MOSES_TOKENIZER_DEFAULT_LANG)
        
    tokenizer = lambda x: moses_detokenizer.detokenize(x.split(), return_str=True) #string IN -> string OUT
    return tokenizer

def load_models(config_path):
    config_data = read_config(config_path)

    for model_config in config_data['models']:
        if model_config['load']:
            model_id = model_config['src'] + model_config['tgt']
            loaded_models[model_id] = {}
            loaded_models[model_id]['src'] = model_config['src']
            loaded_models[model_id]['tgt'] = model_config['tgt']
            
            model_dir = os.path.join(config_data['models_root'], model_config['model_path'])
            print(model_dir)
            
            #Load model pipeline
            print("Pipeline:", end=" ")
            loaded_models[model_id]['pipeline'] = []
            if 'lowercase' in model_config['pipeline'] and model_config['pipeline']['lowercase']:
                print("lowercase", end=" ")
                loaded_models[model_id]['pipeline'].append(lowercaser)

            if 'tokenize' in model_config['pipeline'] and model_config['pipeline']['tokenize']:
                print("tokenize", end=" ")
                loaded_models[model_id]['pipeline'].append(get_tokenizer(model_config['src']))

            if 'bpe' in model_config['pipeline'] and model_config['pipeline']['bpe']:
                print("bpe", end=" ")
                loaded_models[model_id]['pipeline'].append(get_segmenter(os.path.join(model_dir, model_config['bpe_file'])))
            else:
                loaded_models[model_id]['pipeline'].append(token_segmenter)

            if 'translate' in model_config['pipeline'] and model_config['pipeline']['translate']:
                print("translate", end=" ")
                loaded_models[model_id]['pipeline'].append(get_translator(model_dir))
            else:
            	loaded_models[model_id]['pipeline'].append(dummy_translator)

            if 'bpe' in model_config['pipeline'] and model_config['pipeline']['bpe']:
                print("unbpe", end=" ")
                loaded_models[model_id]['pipeline'].append(desegmenter)
            else:
                loaded_models[model_id]['pipeline'].append(token_desegmenter)

            if 'tokenize' in model_config['pipeline'] and model_config['pipeline']['tokenize']:
                print("detokenize", end=" ")
                loaded_models[model_id]['pipeline'].append(get_detokenizer(model_config['tgt']))

            if 'recase' in model_config['pipeline'] and model_config['pipeline']['recase']:
                print("racase", end=" ")
                loaded_models[model_id]['pipeline'].append(capitalizer)

            print()

def read_config(config_file):
    with open(config_file, "r") as jsonfile: 
        data = json.load(jsonfile) 
        print("Config Read successful") 
        print(data)
    return data

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    data = request.get_json(force=True)
    
    src_lang = data['src']
    tgt_lang = data['tgt']

    model_id = src_lang + tgt_lang

    if model_id in loaded_models:

        src_sentence = data['text']

        tgt_sentence = src_sentence
        for step in loaded_models[model_id]['pipeline']:
            tgt_sentence = step(tgt_sentence)

        out = {'text': src_sentence, 'translation':tgt_sentence}     

        return jsonify(out)  
    else:
        return "No model named %s\n"%model_id

@app.route('/reload', methods=['GET', 'POST'])
def reload():
    load_models(CONFIG_FILE)

    return "Reloaded models\n"

CONFIG_FILE = "models_config.json"
load_models(CONFIG_FILE)
