from app import app
from flask import Flask, request, jsonify, render_template
from flask_wtf import FlaskForm
from flask_pagedown import PageDown
from flask_pagedown.fields import PageDownField
from wtforms.fields import SubmitField
import re
import os
from subword_nmt import apply_bpe
from ctranslate2 import Translator   #COMMENT ON PC
from sacremoses import MosesTokenizer, MosesDetokenizer
import json

#constants
MOSES_TOKENIZER_DEFAULT_LANG = 'en'
CONFIG_FILE = "models_config.json"
# CONFIG_FILE = "models_config_mac.json"     #COMMENT OFF PC

#GUI elements
app.config['SECRET_KEY'] = 'secret!'
pagedown = PageDown(app)

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

def get_translator(ctranslator_model_path):   #COMMENT ON PC
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
            model_id = model_config['src'] + "-" + model_config['tgt']
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
                loaded_models[model_id]['pipeline'].append(get_translator(model_dir))   #COMMENT ON PC
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

def translate(src_lang, tgt_lang, text):
    model_id = src_lang + "-" + tgt_lang
    print(model_id)
    print(loaded_models)

    if model_id in loaded_models:

        tgt_sentence = text
        for step in loaded_models[model_id]['pipeline']:
            tgt_sentence = step(tgt_sentence)

        return tgt_sentence
    else:
        return 0

def read_config(config_file):
    with open(config_file, "r") as jsonfile: 
        data = json.load(jsonfile) 
        print("Config Read successful") 
        print(data)
    return data

@app.route('/translate', methods=['GET', 'POST'])
def translate_api():
    data = request.get_json(force=True)
    
    translation = translate(data['src'], data['tgt'], data['text'])

    if translation:
        out = {'text': data['text'], 'translation':translation}     
        return jsonify(out)  
    else:
        return "Language pair not supported %s\n"%(data['src'] + "-" + data['tgt'])


@app.route('/reload', methods=['GET', 'POST'])
def reload():
    load_models(CONFIG_FILE)

    return "Reloaded models\n"

class PageDownFormExample(FlaskForm):
    pagedown = PageDownField('Type the text you want to translate and click "Translate".')
    submit = SubmitField('Translate')

@app.route('/', methods=['GET', 'POST'])
def gui():
    form = PageDownFormExample()
    language = str(request.form.get('lang'))
    translated_text = ""
    if form.validate_on_submit():
        source = form.pagedown.data

        src_language = language.split("-")[0]
        tgt_language = language.split("-")[1]

        print("Request %s-%s"%(src_language, tgt_language))
        print(source)

        translated_text = translate(src_language, tgt_language, source)

        if not translated_text:
            translated_text = "Something went wrong"
    else:
        form.pagedown.data = ('This is a very simple test.')

    print("exit language", language)
    return render_template('index.html', form=form, language=language, text=translated_text)


load_models(CONFIG_FILE)
